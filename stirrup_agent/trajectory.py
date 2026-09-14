"""Native archipelago trajectory to ATIF. Copied verbatim from the task adapter."""

from __future__ import annotations

import json
import os
import sys


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "")
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"WARNING: ignoring non-numeric {name}={raw!r}", file=sys.stderr)
        return default


def _trajectory_id(runtime_env: dict[str, str]) -> str:
    return f"harbor-{runtime_env.get('WORLD_TASK_ID') or 'task'}"


def _text(content: object) -> str:
    if content is None:
        return ""
    if isinstance(content, list):
        return " ".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return str(content)


def _arguments(value: object) -> dict:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, dict) else {"_value": parsed}
    except Exception:
        return {"_raw": str(value)}


def _metrics(call: dict | None) -> dict | None:
    if not call:
        return None
    metrics = {
        key: call.get(key)
        for key in ("prompt_tokens", "completion_tokens", "cached_tokens")
        if call.get(key) is not None
    }
    extra = {
        key: call[key]
        for key in ("cache_creation_tokens", "reasoning_tokens", "total_tokens")
        if key in call
    }
    if extra:
        metrics["extra"] = extra
    return metrics or None


def convert_trajectory(trajectory: dict, session_id: str | None = None) -> dict:
    messages = trajectory.get("messages") or []
    call_log = (trajectory.get("usage") or {}).get("call_log") or []
    steps: list[dict] = []
    assistant_index = 0
    index = 0
    while index < len(messages):
        message = messages[index]
        role = message.get("role")
        if role in ("system", "user"):
            steps.append(
                {
                    "step_id": len(steps) + 1,
                    "source": role,
                    "message": _text(message.get("content")),
                }
            )
            index += 1
            continue
        if role == "assistant":
            step: dict = {
                "step_id": len(steps) + 1,
                "source": "agent",
                "message": _text(message.get("content")),
            }
            if "model_name" in message:
                step["model_name"] = message["model_name"]
            calls = []
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                calls.append(
                    {
                        "tool_call_id": call.get("id") or function.get("name"),
                        "function_name": function.get("name"),
                        "arguments": _arguments(function.get("arguments")),
                    }
                )
            if calls:
                step["tool_calls"] = calls
            metrics = _metrics(
                call_log[assistant_index] if assistant_index < len(call_log) else None
            )
            if metrics:
                step["metrics"] = metrics
            assistant_index += 1
            results = []
            next_index = index + 1
            while (
                next_index < len(messages)
                and messages[next_index].get("role") == "tool"
            ):
                tool_message = messages[next_index]
                results.append(
                    {
                        "source_call_id": tool_message.get("tool_call_id"),
                        "content": _text(tool_message.get("content")),
                    }
                )
                next_index += 1
            if results:
                step["observation"] = {"results": results}
            steps.append(step)
            index = next_index
            continue
        step = {
            "step_id": len(steps) + 1,
            "source": "agent",
            "message": _text(message.get("content")),
            "extra": {"native_role": role},
        }
        if "model_name" in message:
            step["model_name"] = message["model_name"]
        steps.append(step)
        index += 1
    usage = trajectory.get("usage") or {}
    final_metrics = {
        key: value
        for key, value in {
            "total_prompt_tokens": usage.get("prompt_tokens"),
            "total_completion_tokens": usage.get("completion_tokens"),
            "total_cached_tokens": usage.get("cached_tokens"),
            "total_steps": len(steps),
        }.items()
        if value is not None
    }
    return {
        "schema_version": "ATIF-v1.7",
        "session_id": trajectory.get("session_id") or session_id or "archipelago",
        "agent": {"name": "archipelago", "version": "1.0"},
        "steps": steps,
        "final_metrics": final_metrics,
        "extra": {
            "converted_from": "archipelago-native",
            "native_status": trajectory.get("status"),
            "time_elapsed_sec": trajectory.get("time_elapsed"),
            "native_output": trajectory.get("output") or {},
        },
    }
