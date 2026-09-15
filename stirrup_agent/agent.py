"""Stirrup as an ACP agent, so hosted Harbor can run it from this repo."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import NamedTuple

import acp
from acp.schema import (
    AgentCapabilities,
    Cost,
    Implementation,
    PromptCapabilities,
    Usage,
    UsageUpdate,
)

from . import runner
from .trajectory import convert_trajectory

AGENT_NAME = "stirrup"


def _agent_version() -> str:
    """pyproject is the source of truth; harbor-agent.json is checked against it.

    A source checkout with nothing installed has no metadata to read, and the
    version only ever reaches Harbor from an installed agent.
    """
    try:
        return version("stirrup-agent")
    except PackageNotFoundError:
        return "0.0.0"


AGENT_VERSION = _agent_version()


def _instruction(blocks: list) -> str:
    parts = []
    for block in blocks:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def _kwargs_from_env() -> dict:
    """Hosted passes agent kwargs as env, since ACP has no kwargs channel."""
    raw = os.environ.get("STIRRUP_AGENT_KWARGS")
    if not raw:
        return {}
    try:
        loaded = json.loads(raw)
    except ValueError as error:
        raise RuntimeError("STIRRUP_AGENT_KWARGS is not valid JSON") from error
    if not isinstance(loaded, dict):
        raise RuntimeError("STIRRUP_AGENT_KWARGS must be a JSON object")
    return loaded


class StirrupAgent(acp.Agent):
    def __init__(self, connection=None) -> None:
        # the connection arrives by factory, on_connect is never called
        self._conn = connection
        self._sessions: dict[str, str] = {}
        self._kwargs = _kwargs_from_env()

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities=None,
        client_info=None,
        **_: object,
    ) -> acp.InitializeResponse:
        return acp.InitializeResponse(
            protocol_version=min(protocol_version, acp.PROTOCOL_VERSION),
            agent_capabilities=AgentCapabilities(
                load_session=False,
                prompt_capabilities=PromptCapabilities(image=True, embedded_context=True),
            ),
            agent_info=Implementation(name=AGENT_NAME, version=AGENT_VERSION),
        )

    async def new_session(
        self,
        cwd: str,
        additional_directories=None,
        mcp_servers=None,
        **_: object,
    ) -> acp.NewSessionResponse:
        runner.assert_runner_present()
        session_id = f"stirrup-{uuid.uuid4().hex[:12]}"
        self._sessions[session_id] = cwd
        return acp.NewSessionResponse(session_id=session_id)

    async def prompt(self, session_id: str, prompt: list, **_: object) -> acp.PromptResponse:
        instruction = _instruction(prompt)
        if not instruction.strip():
            raise acp.RequestError.invalid_params("empty prompt")

        kwargs = self._kwargs
        runner.write_inputs(
            instruction,
            system_prompt=kwargs.get("agent_system_prompt"),
            agent_config_id=kwargs.get("agent_config_id"),
            agent_config_values=kwargs.get("agent_config_values"),
            orchestrator_extra_args=kwargs.get("orchestrator_extra_args"),
            agent_name=kwargs.get("agent_name"),
        )
        timeout = (kwargs.get("agent_config_values") or {}).get("timeout")
        native = await runner.run(
            trajectory_id=session_id,
            model_name=kwargs.get("model_name"),
            timeout_sec=int(timeout) if timeout else None,
        )

        atif = convert_trajectory(native, session_id=session_id)
        (runner.LOG_DIR / "trajectory.json").write_text(json.dumps(atif, indent=1))

        streamed = await self._stream_turns(session_id, native)

        summary = ((native.get("output") or {}).get("finish_reason") or "").strip()
        if summary:
            await self._say(session_id, summary)
        # after _say so Harbor attaches it to a step instead of orphaning it.
        # Skipped when the replay already carried cost on its closing turn.
        if not streamed:
            await self._report_usage(session_id, native)
        return acp.PromptResponse(
            stop_reason=_stop_reason(native), usage=_safe_usage(native)
        )

    async def cancel(self, session_id: str, **_: object) -> None:
        self._sessions.pop(session_id, None)

    async def _say(self, session_id: str, text: str) -> None:
        if self._conn is None:
            return
        await self._conn.session_update(session_id, acp.update_agent_message_text(text))

    async def _stream_turns(self, session_id: str, native: dict) -> bool:
        """Replay the finished run as one ACP step per assistant turn.

        Harbor builds its own ATIF from this stream, so a single closing message
        read as 2 steps against roughly 66 real turns. It opens a step on the
        first message or tool call and closes it on a usage_update, so one
        update per turn also moves cost onto the steps instead of the orphan
        bucket. Returns whether the closing turn carried the cost.
        """
        if self._conn is None:
            return False
        try:
            turns = _turns(native)
            if not turns:
                return False
            closing = _usage_update(native)
            for index, turn in enumerate(turns):
                await self._send_turn(session_id, turn)
                last = index == len(turns) - 1
                await self._conn.session_update(
                    session_id, _turn_usage(turn, closing if last else None)
                )
            return closing is not None
        except Exception as error:  # a finished run must not fail over telemetry
            print(f"WARNING: turn replay not sent: {error!r}", file=sys.stderr)
            return False

    async def _send_turn(self, session_id: str, turn: _Turn) -> None:
        """Emit a turn's content before its usage_update, or the step is empty
        and Harbor orphans the update instead of closing a step with it."""
        if turn.reasoning:
            await self._conn.session_update(
                session_id, acp.update_agent_thought_text(turn.reasoning)
            )
        if turn.message:
            await self._conn.session_update(
                session_id, acp.update_agent_message_text(turn.message)
            )
        for call in turn.tool_calls:
            await self._conn.session_update(
                session_id,
                acp.start_tool_call(
                    call.id, call.name, status="pending", raw_input=call.arguments
                ),
            )
            await self._conn.session_update(
                session_id,
                acp.update_tool_call(
                    call.id, status="completed", raw_output=call.output
                ),
            )

    async def _report_usage(self, session_id: str, native: dict) -> None:
        """Harbor fills the Hub's Cost column from this update and nowhere else."""
        if self._conn is None:
            return
        try:
            update = _usage_update(native)
            if update is not None:
                await self._conn.session_update(session_id, update)
        except Exception as error:  # a finished run must not fail over telemetry
            print(f"WARNING: usage update not sent: {error!r}", file=sys.stderr)


class _Call(NamedTuple):
    id: str
    name: str
    arguments: object
    output: object


class _Turn(NamedTuple):
    reasoning: str
    message: str
    tool_calls: list[_Call]
    used: int | None


def _turns(native: dict) -> list[_Turn]:
    """One entry per assistant turn, paired with that turn's context reading.

    call_log is per model call in execution order, so it lines up with the
    assistant messages; a turn past the end of it simply carries no reading.
    """
    messages = native.get("messages") or []
    if not isinstance(messages, list):
        return []
    calls = (native.get("usage") or {}).get("call_log")
    calls = calls if isinstance(calls, list) else []
    outputs = {
        m.get("tool_call_id"): _text(m.get("content"))
        for m in messages
        if isinstance(m, dict) and m.get("role") == "tool" and m.get("tool_call_id")
    }
    turns: list[_Turn] = []
    # call_log has an entry per model call, including a turn we drop below, so
    # it is keyed on the assistant message and not on what we keep
    assistant_index = 0
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in (
            "assistant",
            "agent",
        ):
            continue
        tool_calls = []
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            function = call.get("function") or {}
            call_id = call.get("id") or function.get("name")
            if not call_id:
                continue
            tool_calls.append(
                _Call(
                    id=str(call_id),
                    name=str(function.get("name") or "tool"),
                    arguments=function.get("arguments"),
                    output=outputs.get(call_id),
                )
            )
        entry = calls[assistant_index] if assistant_index < len(calls) else {}
        assistant_index += 1
        message_text = _text(message.get("content"))
        reasoning = _text(message.get("reasoning_content"))
        if not (message_text or reasoning or tool_calls):
            # nothing to open a step with, so a usage_update here would orphan
            continue
        turns.append(
            _Turn(
                reasoning=reasoning,
                message=message_text,
                tool_calls=tool_calls,
                used=_int_or_none((entry or {}).get("prompt_tokens")),
            )
        )
    return turns


def _turn_usage(turn: _Turn, closing: UsageUpdate | None) -> UsageUpdate:
    """The turn's own context reading, carrying cost only where we measured it."""
    used = turn.used if turn.used is not None else 0
    if closing is None:
        return UsageUpdate(sessionUpdate="usage_update", used=used, size=max(used, 1))
    size = max(closing.size, used)
    return UsageUpdate(
        sessionUpdate="usage_update", used=used or closing.used, size=size,
        cost=closing.cost,
    )


def _text(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [
            part.get("text", "")
            for part in value
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        ]
        return "\n".join(p for p in parts if p).strip()
    return ""


def _safe_usage(native: dict) -> Usage | None:
    try:
        return _usage(native)
    except Exception as error:  # a finished run must not fail over telemetry
        print(f"WARNING: token usage not reported: {error!r}", file=sys.stderr)
        return None


def _usage(native: dict) -> Usage | None:
    """Harbor reads this off the PromptResponse to fill the Hub's token columns."""
    usage = native.get("usage") or {}
    prompt = _int_or_none(usage.get("prompt_tokens"))
    completion = _int_or_none(usage.get("completion_tokens"))
    if prompt is None and completion is None:
        return None
    prompt, completion = prompt or 0, completion or 0
    total = _int_or_none(usage.get("total_tokens"))
    return Usage(
        total_tokens=prompt + completion if total is None else total,
        input_tokens=prompt,
        output_tokens=completion,
        thought_tokens=_int_or_none(usage.get("reasoning_tokens")),
        cached_read_tokens=_int_or_none(usage.get("cached_tokens")),
        cached_write_tokens=_int_or_none(usage.get("cache_creation_tokens")),
    )


def _usage_update(native: dict) -> UsageUpdate | None:
    """Skips a run the runner could not price at all, so $0.00 never stands in
    for a figure we never measured."""
    usage = native.get("usage") or {}
    spent = usage.get("cost_usd_spent")
    if isinstance(spent, bool) or not isinstance(spent, int | float) or spent < 0:
        return None
    # a tracker drops the call_log entry when usage is unreadable but still
    # counts the call as unpriced, so the two never line up; $0 with any
    # unpriced call means we measured nothing, not that the run was free
    if not spent and _int_or_none(usage.get("cost_unpriced_calls")):
        return None
    calls = usage.get("call_log")
    calls = calls if isinstance(calls, list) else []
    last = calls[-1] if calls and isinstance(calls[-1], dict) else {}
    peak = _int_or_none(usage.get("max_prompt_tokens")) or 0
    used = _int_or_none(last.get("prompt_tokens"))
    used = peak if used is None else used
    return UsageUpdate(
        sessionUpdate="usage_update",
        used=used,
        # the largest context we occupied; the runner never reports the window
        size=max(peak, used),
        cost=Cost(amount=float(spent), currency="USD"),
    )


def _int_or_none(value: object) -> int | None:
    """bool is an int in Python, and ACP rejects negatives, so both are dropped.
    A float count is still a count."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return int(value) if value >= 0 else None


def _stop_reason(native: dict) -> str:
    status = (native.get("status") or "").lower()
    if status in {"cancelled", "canceled"}:
        return "cancelled"
    if status == "refused":
        return "refusal"
    if (native.get("output") or {}).get("abandoned"):
        return "max_turn_requests"
    return "end_turn"


def prepare_log_dir(path: Path | None = None) -> Path:
    """The task's collect hooks write here too and may run as another user."""
    log_dir = Path(path or runner.LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        log_dir.chmod(0o777)
    except OSError:
        pass  # a dir we do not own is already someone else's to share
    return log_dir


def main() -> None:
    prepare_log_dir()
    asyncio.run(acp.run_agent(StirrupAgent))
