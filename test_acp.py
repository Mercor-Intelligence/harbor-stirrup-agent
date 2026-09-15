"""End-to-end ACP check: a real client drives the agent against a stubbed runner.

Run: uv run python test_acp.py
"""

from __future__ import annotations

import asyncio
import json
import os
import re
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
    # shaped like UsageTracker.to_dict(): every counter present on the total
    # and on each call_log entry, as cost_accounting mode emits it
    "usage": {"prompt_tokens": 1200, "completion_tokens": 80, "total_tokens": 1280,
              "cached_tokens": 900, "cache_creation_tokens": 64,
              "reasoning_tokens": 40, "final_answer_tokens": 12,
              "max_prompt_tokens": 1100, "compaction_count": 0,
              "accounting_mode": "cost_accounting",
              "cost_usd_spent": 0.004212, "cost_unpriced_calls": 0,
              "call_log": [
                  {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
                   "cached_tokens": 0, "cache_creation_tokens": 64,
                   "reasoning_tokens": 10},
                  {"prompt_tokens": 1100, "completion_tokens": 60, "total_tokens": 1160,
                   "cached_tokens": 900, "cache_creation_tokens": 0,
                   "reasoning_tokens": 30},
              ]},
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
        self.stream: list[str] = []

    async def session_update(self, session_id, update, **_):
        kind = getattr(update, "session_update", None)
        if kind:
            self.stream.append(kind)
        if kind == "usage_update":
            self.usage_updates.append(update)
            return
        text = getattr(getattr(update, "content", None), "text", None)
        if text:
            self.messages.append(text)


def _check_version_agreement() -> None:
    """The Hub's Agent Version column comes from harbor-agent.json while ACP
    reports its own constant; they have drifted before."""
    from stirrup_agent.agent import AGENT_NAME, AGENT_VERSION

    root = Path(__file__).parent
    manifest = json.loads((root / "harbor-agent.json").read_text())
    declared = re.search(
        r'^version = "([^"]+)"', (root / "pyproject.toml").read_text(), re.M
    ).group(1)

    assert manifest["id"] == AGENT_NAME, (manifest["id"], AGENT_NAME)
    assert manifest["version"] == AGENT_VERSION, (manifest["version"], AGENT_VERSION)
    assert declared == AGENT_VERSION, (declared, AGENT_VERSION)
    print(f"version      : {AGENT_VERSION} agrees across manifest, pyproject and ACP")


def _check_call_log_alignment() -> None:
    """A dropped turn must not shift every later context reading.

    call_log has an entry per model call, including the turn we drop for having
    nothing to open a step with, so indexing on kept turns reuses an earlier
    reading for every turn after the gap.
    """
    from stirrup_agent.agent import _turns

    native = {
        "messages": [
            {"role": "assistant", "content": "first"},
            {"role": "assistant", "content": ""},  # dropped: nothing to show
            {"role": "assistant", "content": "third"},
        ],
        "usage": {
            "call_log": [
                {"prompt_tokens": 10},
                {"prompt_tokens": 20},
                {"prompt_tokens": 30},
            ]
        },
    }
    turns = _turns(native)

    assert [t.message for t in turns] == ["first", "third"], turns
    # the kept turns take the 1st and 3rd readings, not the 1st and 2nd
    assert [t.used for t in turns] == [10, 30], [t.used for t in turns]
    print("call log     : a dropped turn does not shift later readings")


def _check_degraded_usage() -> None:
    """Telemetry must never fail a run that already finished, and never invent
    a figure the runner did not measure."""
    from stirrup_agent.agent import _safe_usage, _usage_update

    # no cost_usd_spent at all: the default accounting mode
    assert _usage_update({"usage": {"prompt_tokens": 5}}) is None
    # priced nothing, so $0.00 would be a measurement we never made
    assert _usage_update(
        {"usage": {"cost_usd_spent": 0.0, "cost_unpriced_calls": 2,
                   "call_log": [{}, {}]}}
    ) is None
    # a malformed count is dropped, not raised, and the good ones survive
    degraded = _safe_usage({"usage": {"prompt_tokens": -1, "completion_tokens": 5}})
    assert degraded is not None and degraded.output_tokens == 5, degraded
    # bool is an int in Python; it must not read as a count of 1
    assert _safe_usage(
        {"usage": {"prompt_tokens": 10, "completion_tokens": 2,
                   "reasoning_tokens": True}}
    ).thought_tokens is None
    # a float count is still a count, not a reason to drop every column
    floats = _safe_usage({"usage": {"prompt_tokens": 1200.0, "completion_tokens": 80.0}})
    assert floats is not None and floats.input_tokens == 1200, floats
    # unpriced calls outnumber call_log entries, and can exist with none at all
    assert _usage_update(
        {"usage": {"cost_usd_spent": 0.0, "cost_unpriced_calls": 10,
                   "call_log": [{}] * 9}}
    ) is None
    assert _usage_update(
        {"usage": {"cost_usd_spent": 0.0, "cost_unpriced_calls": 3, "call_log": []}}
    ) is None
    print("degraded     : no-cost, unpriced, negative and bool paths all handled")


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
    # one step per assistant turn: the stub has two, so two usage_updates
    # close two steps rather than one closing update for the whole run
    assert len(captured.usage_updates) == 2, captured.usage_updates
    assert captured.stream.count("tool_call") == 1, captured.stream
    assert captured.stream.count("tool_call_update") == 1, captured.stream
    # content must precede its usage_update or Harbor orphans the update
    first_usage = captured.stream.index("usage_update")
    assert first_usage > 0 and captured.stream[0] != "usage_update", captured.stream
    # only the closing turn carries cost; the rest are context readings
    assert [u.cost is not None for u in captured.usage_updates] == [False, True], (
        [u.cost for u in captured.usage_updates]
    )
    cost = captured.usage_updates[-1].cost
    assert cost.currency == "USD" and cost.amount == 0.004212, cost
    # per-turn readings come from call_log in order: 100 then 1100
    assert [u.used for u in captured.usage_updates] == [100, 1100], captured.usage_updates
    assert resp.usage.thought_tokens == 40, resp.usage
    assert resp.usage.cached_write_tokens == 64, resp.usage
    _check_degraded_usage()
    _check_call_log_alignment()
    _check_version_agreement()
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
