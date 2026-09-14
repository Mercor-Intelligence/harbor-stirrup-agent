"""Stirrup as an ACP agent, so hosted Harbor can run it from this repo."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path

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
AGENT_VERSION = "1.0"


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

        await self._report_usage(session_id, native)
        summary = ((native.get("output") or {}).get("finish_reason") or "").strip()
        if summary:
            await self._say(session_id, summary)
        return acp.PromptResponse(
            stop_reason=_stop_reason(native), usage=_usage(native)
        )

    async def cancel(self, session_id: str, **_: object) -> None:
        self._sessions.pop(session_id, None)

    async def _say(self, session_id: str, text: str) -> None:
        if self._conn is None:
            return
        await self._conn.session_update(session_id, acp.update_agent_message_text(text))

    async def _report_usage(self, session_id: str, native: dict) -> None:
        """Harbor fills the Hub's Cost column from this update, and nowhere else.

        Only fires in the runner's cost_accounting mode, which is what prices
        the calls. `size` is the largest context we actually occupied, not the
        model's window: the runner never reports the window, and a made-up
        number is worse than a measured one that is named honestly here.
        """
        if self._conn is None:
            return
        usage = native.get("usage") or {}
        spent = usage.get("cost_usd_spent")
        if not isinstance(spent, int | float):
            return
        calls = usage.get("call_log") or []
        last = calls[-1] if isinstance(calls, list) and calls else {}
        peak = _int_or_none(usage.get("max_prompt_tokens")) or 0
        used = _int_or_none(last.get("prompt_tokens")) or peak
        await self._conn.session_update(
            session_id,
            UsageUpdate(
                sessionUpdate="usage_update",
                used=used,
                size=max(peak, used),
                cost=Cost(amount=float(spent), currency="USD"),
            ),
        )


def _usage(native: dict) -> Usage | None:
    """Harbor reads this off the PromptResponse to fill the Hub's token columns."""
    usage = native.get("usage") or {}
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if not isinstance(prompt, int) or not isinstance(completion, int):
        return None
    total = usage.get("total_tokens")
    return Usage(
        total_tokens=total if isinstance(total, int) else prompt + completion,
        input_tokens=prompt,
        output_tokens=completion,
        thought_tokens=_int_or_none(usage.get("reasoning_tokens")),
        cached_read_tokens=_int_or_none(usage.get("cached_tokens")),
        cached_write_tokens=_int_or_none(usage.get("cache_creation_tokens")),
    )


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) else None


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
