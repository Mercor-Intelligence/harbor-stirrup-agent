# stirrup-agent

Mercor's Stirrup agent, exposed over [ACP](https://agentclientprotocol.com) so it can run on
Harbor's hosted compute.

Hosted Harbor only accepts built-in agents or a custom ACP agent from a GitHub repo. The agent
we ship inside task packages uses Harbor's `import_path` mechanism, which hosted rejects. This
repo is the same agent behind an ACP front door.

## How it works

The orchestrator is not in this repo. It is baked into each published task image at
`/agent_runner`. This package is a thin shim:

```
prompt arrives over ACP
  -> write /logs/agent/{messages,agent_config,orchestrator_extra_args}.json
  -> run: cd /agent_runner && uv run --no-sync python -m runner.main ... --output trajectory.native.json
  -> convert the native trajectory to ATIF-v1.7
  -> stream the finish summary back, then the run's usage and cost
  -> return a stop reason, with token counts on the response
```

Harbor builds its own ATIF from the ACP stream, so token counts ride the
`PromptResponse` and cost rides a `usage_update`. Cost only appears when the
runner priced the calls, which needs `accounting_mode: cost_accounting` below.
Without it the Hub's Cost column stays empty.

`trajectory.py` is copied verbatim from the task adapter so both paths produce identical ATIF.

## Configuration

Agent kwargs arrive as JSON in `STIRRUP_AGENT_KWARGS`, since ACP has no kwargs channel:

```json
{
  "agent_config_id": "stirrup_agent",
  "agent_name": "Stirrup Agent",
  "model_name": "gemini/gemini-3.8-flash",
  "agent_config_values": {"timeout": 10800, "max_turns": 200,
                          "accounting_mode": "cost_accounting"},
  "orchestrator_extra_args": {"reasoning_effort": "xhigh"}
}
```

| Variable | Purpose |
| --- | --- |
| `STIRRUP_AGENT_KWARGS` | agent config as JSON |
| `STIRRUP_INFERENCE` | `hosted` or `direct`, overrides the default choice |
| `HOSTED_INFERENCE_URL` / `_TOKEN` | Harbor's inference broker, mapped onto `LITELLM_PROXY_*` |
| `GEMINI_API_KEY` etc. | a provider key, used directly |
| `STIRRUP_RUNNER_DIR` | defaults to `/agent_runner` |
| `STIRRUP_LOG_DIR` | defaults to `/logs/agent` |

### Which credential gets used

A provider key wins over hosted inference, so supplying `GEMINI_API_KEY` bills that key rather
than Harbor's broker. Set `STIRRUP_INFERENCE` to force either path. When hosted inference is
used the model is prefixed with `litellm_proxy/`, because that token only authenticates the
broker.

## Launching

```json
{
  "config": {
    "job_name": "gdpval-hosted",
    "agents": [{
      "name": "acp",
      "source": {"type": "github", "repo": "<owner>/<repo>", "manifest": "harbor-agent.json"},
      "model_name": "gemini/gemini-3.8-flash",
      "secrets": ["GEMINI_API_KEY"]
    }],
    "datasets": [{"name": "mercor/gdpval", "ref": "latest"}],
    "n_attempts": 1,
    "n_concurrent_trials": 20
  },
  "organization": "mercor",
  "job_secrets": {"GEMINI_API_KEY": "..."},
  "dry_run": true
}
```

`POST /job-submit` with an `Idempotency-Key` header. Keep `dry_run` true to validate the repo,
manifest, and dataset without creating a job.

## Tests

```bash
uv run python test_acp.py          # end to end over ACP against a stubbed runner
uv run python test_credentials.py  # which credential path wins
```

Neither needs a real task image or an API key.
