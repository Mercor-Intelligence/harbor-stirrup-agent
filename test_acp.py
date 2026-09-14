"""End-to-end ACP check: a real client drives the agent against a stubbed runner.

Run: uv run python test_acp.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

import acp

# mimics the archipelago runner's native output: messages + usage.call_log
STUB_RUNNER = '''#!/usr/bin/env python3
import json, sys

args = sys.argv
out = args[args.index("--output") + 1]
model = args[args.index("--orchestrator-model") + 1]
msgs = json.load(open(args[args.index("--initial-messages") + 1]))

json.dump({
    "session_id": "stub",
    "status": "completed",
    "time_elapsed": 1.5,
    "messages": msgs + [
        {"role": "assistant", "content": "Listing the workspace.", "model_name": model,
         "tool_calls": [{"id": "call_1", "function": {"name": "run_shell",
                         "arguments": json.dumps({"command": "ls"})}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "report.xlsx"},
        {"role": "assistant", "content": "Done, wrote report.xlsx.", "model_name": model},
    ],
    "usage": {"prompt_tokens": 1200, "completion_tokens": 80, "cached_tokens": 900,
              "max_prompt_tokens": 1100, "cost_usd_spent": 0.004212,
              "call_log": [{"prompt_tokens": 1100, "completion_tokens": 80}]},
    "output": {"finish_reason": "wrote report.xlsx with model " + model,
               "finish_paths": ["/filesystem/report.xlsx"], "abandoned": False},
}, open(out, "w"))
'''


def make_stub(root: Path) -> Path:
    runner_dir = root / "agent_runner"
    (runner_dir / "runner").mkdir(parents=True)
    (root / "logs").mkdir()
    bin_dir = root / "bin"
    bin_dir.mkdir()
    shim = bin_dir / "uv"
    shim.write_text(STUB_RUNNER)
    shim.chmod(0o755)
    return runner_dir


class CaptureClient(acp.Client):
    def __init__(self) -> None:
        self.messages: list[str] = []
        self.usage_updates: list[object] = []

    async def session_update(self, session_id, update, **_):
        if getattr(update, "session_update", None) == "usage_update":
            self.usage_updates.append(update)
            return
        text = getattr(getattr(update, "content", None), "text", None)
        if text:
            self.messages.append(text)


async def main() -> int:
    root = Path(tempfile.mkdtemp())
    runner_dir = make_stub(root)

    env = dict(os.environ)
    env["PATH"] = f"{root / 'bin'}:{env['PATH']}"
    env["STIRRUP_RUNNER_DIR"] = str(runner_dir)
    env["STIRRUP_LOG_DIR"] = str(root / "logs")
    env["HOSTED_INFERENCE_URL"] = "https://hosted.example/v1"
    env["HOSTED_INFERENCE_TOKEN"] = "placeholder-not-a-real-secret"
    env["STIRRUP_AGENT_KWARGS"] = json.dumps(
        {
            "agent_config_id": "stirrup_agent",
            "agent_name": "Stirrup Agent",
            "model_name": "gemini/gemini-3.8-flash",
            "agent_config_values": {"timeout": 600, "max_turns": 200},
            "orchestrator_extra_args": {"reasoning_effort": "xhigh"},
        }
    )

    captured = CaptureClient()
    async with acp.spawn_agent_process(
        captured, sys.executable, "-m", "stirrup_agent",
        env=env, cwd=str(Path(__file__).parent),
    ) as (agent, _proc):
        init = await agent.initialize(protocol_version=acp.PROTOCOL_VERSION)
        print("initialize   ->", init.agent_info.name, init.agent_info.version,
              "| proto", init.protocol_version)

        session = await agent.new_session(cwd=str(root))
        print("new_session  ->", session.session_id)

        resp = await agent.prompt(
            session_id=session.session_id,
            prompt=[acp.text_block("Summarize the well data into an Excel file.")],
        )
        print("prompt       -> stop_reason:", resp.stop_reason)
        print("streamed     ->", captured.messages)

    atif = json.loads((root / "logs" / "trajectory.json").read_text())
    steps = atif["steps"]
    print()
    print("ATIF schema  :", atif["schema_version"])
    print("steps        :", len(steps), [s["source"] for s in steps])
    print("tool calls   :", sum(len(s.get("tool_calls") or []) for s in steps))
    print("final_metrics:", atif["final_metrics"])
    print("prompt usage :", resp.usage)
    print("usage update :", captured.usage_updates)
    print("finish_reason:", atif["extra"]["native_output"]["finish_reason"])

    extra = json.loads((root / "logs" / "orchestrator_extra_args.json").read_text())
    cfg = json.loads((root / "logs" / "agent_config.json").read_text())

    assert atif["schema_version"] == "ATIF-v1.7", atif["schema_version"]
    assert resp.stop_reason == "end_turn", resp.stop_reason
    assert [s["source"] for s in steps] == ["user", "agent", "agent"], steps
    assert sum(len(s.get("tool_calls") or []) for s in steps) == 1, "tool call lost"
    assert atif["final_metrics"]["total_prompt_tokens"] == 1200, atif["final_metrics"]
    # Harbor builds its own ATIF from the ACP stream, so usage has to ride the response
    assert resp.usage is not None, "PromptResponse carries no usage"
    assert resp.usage.input_tokens == 1200, resp.usage
    assert resp.usage.output_tokens == 80, resp.usage
    assert resp.usage.cached_read_tokens == 900, resp.usage
    assert resp.usage.total_tokens == 1280, resp.usage
    # cost only reaches the Hub through a usage_update, never the response
    assert len(captured.usage_updates) == 1, captured.usage_updates
    cost = captured.usage_updates[0].cost
    assert cost.currency == "USD" and cost.amount == 0.004212, cost
    assert captured.usage_updates[0].used == 1100, captured.usage_updates[0]
    # HOSTED_INFERENCE_* must route the model through the proxy prefix
    assert "litellm_proxy/gemini/gemini-3.8-flash" in \
        atif["extra"]["native_output"]["finish_reason"], "hosted creds not mapped"
    assert extra["reasoning_effort"] == "xhigh", extra
    assert cfg["agent_config_id"] == "stirrup_agent", cfg
    assert captured.messages, "no progress streamed to the client"
    print()
    print("ALL ASSERTIONS PASSED")
    return 0


raise SystemExit(asyncio.run(main()))
