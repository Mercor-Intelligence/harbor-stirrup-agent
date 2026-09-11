"""Invokes the archipelago runner baked into the task image at /agent_runner."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

RUNNER_DIR = Path(os.environ.get("STIRRUP_RUNNER_DIR", "/agent_runner"))
LOG_DIR = Path(os.environ.get("STIRRUP_LOG_DIR", "/logs/agent"))
DEFAULT_GATEWAY_URL = os.environ.get("STIRRUP_GATEWAY_URL", "http://localhost:8000/mcp")

DEFAULT_AGENT_CONFIG_ID = "react_toolbelt_agent"
DEFAULT_TIMEOUT_SEC = 10800


class RunnerMissingError(RuntimeError):
    pass


class RunnerFailedError(RuntimeError):
    def __init__(self, returncode: int, tail: str) -> None:
        super().__init__(f"archipelago runner exited {returncode}")
        self.returncode = returncode
        self.tail = tail


def assert_runner_present() -> None:
    if not (RUNNER_DIR / "runner").is_dir():
        raise RunnerMissingError(f"no baked agents project at {RUNNER_DIR}")


def _use_hosted_inference(env: dict[str, str]) -> bool:
    """Default on only when no provider key is present, so a direct key wins."""
    choice = env.get("STIRRUP_INFERENCE", "").strip().lower()
    if choice in {"hosted", "proxy"}:
        return True
    if choice in {"direct", "provider"}:
        return False
    if choice:
        raise ValueError(f"STIRRUP_INFERENCE must be hosted or direct, got {choice!r}")
    if not {"HOSTED_INFERENCE_URL", "HOSTED_INFERENCE_TOKEN"} <= env.keys():
        return False
    return not any(
        env.get(name)
        for name in (
            "GOOGLE_API_KEY",
            "GEMINI_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "LITELLM_PROXY_API_KEY",
        )
    )


def inference_env() -> tuple[dict[str, str], bool]:
    """Hosted Harbor supplies HOSTED_INFERENCE_*; the runner reads LITELLM_PROXY_*."""
    env = dict(os.environ)
    hosted = _use_hosted_inference(env)
    if hosted:
        env["LITELLM_PROXY_API_BASE"] = env["HOSTED_INFERENCE_URL"]
        env["LITELLM_PROXY_API_KEY"] = env["HOSTED_INFERENCE_TOKEN"]
    return env, hosted


def _model_name(env: dict[str, str], requested: str | None, hosted: bool) -> str:
    model = requested or env.get("STIRRUP_MODEL") or env.get("MODEL_NAME")
    if not model:
        raise RunnerFailedError(2, "no model name supplied")
    # a hosted token only authenticates their proxy, so route through it
    if hosted and not model.startswith("litellm_proxy/"):
        model = f"litellm_proxy/{model}"
    return model


def write_inputs(
    instruction: str,
    *,
    system_prompt: str | None = None,
    agent_config_id: str | None = None,
    agent_config_values: dict | None = None,
    orchestrator_extra_args: dict | None = None,
    agent_name: str | None = None,
) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": instruction})
    (LOG_DIR / "messages.json").write_text(json.dumps(messages))

    config_id = agent_config_id or DEFAULT_AGENT_CONFIG_ID
    (LOG_DIR / "agent_config.json").write_text(
        json.dumps(
            {
                "agent_config_id": config_id,
                "agent_name": agent_name or config_id.replace("_", " ").title(),
                "agent_config_values": dict(agent_config_values or {}),
            }
        )
    )
    (LOG_DIR / "orchestrator_extra_args.json").write_text(
        json.dumps(dict(orchestrator_extra_args or {}))
    )


async def run(
    *,
    trajectory_id: str,
    model_name: str | None = None,
    timeout_sec: int | None = None,
) -> dict:
    assert_runner_present()
    env, hosted = inference_env()
    model = _model_name(env, model_name, hosted)
    output = LOG_DIR / "trajectory.native.json"
    run_log = LOG_DIR / "agent_run.log"
    output.unlink(missing_ok=True)

    # the baked project pins its own interpreter, so --no-sync keeps that venv
    argv = [
        "uv",
        "run",
        "--no-sync",
        "python",
        "-m",
        "runner.main",
        "--trajectory-id",
        trajectory_id,
        "--initial-messages",
        str(LOG_DIR / "messages.json"),
        "--mcp-gateway-url",
        DEFAULT_GATEWAY_URL,
        "--agent-config",
        str(LOG_DIR / "agent_config.json"),
        "--orchestrator-model",
        model,
        "--orchestrator-extra-args",
        str(LOG_DIR / "orchestrator_extra_args.json"),
        "--output",
        str(output),
    ]
    env.setdefault("UV_PYTHON", "python3")

    with run_log.open("wb") as log:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(RUNNER_DIR),
            env=env,
            stdout=log,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            returncode = await asyncio.wait_for(
                proc.wait(), timeout=(timeout_sec or DEFAULT_TIMEOUT_SEC) + 900
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise RunnerFailedError(124, _tail(run_log)) from None

    if not output.exists():
        raise RunnerFailedError(returncode, _tail(run_log))
    return json.loads(output.read_text())


def _tail(path: Path, limit: int = 2000) -> str:
    try:
        return path.read_text(errors="replace")[-limit:]
    except OSError:
        return ""
