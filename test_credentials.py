"""Which credential path the runner picks. Run: uv run python test_credentials.py"""

from __future__ import annotations

import os

from stirrup_agent import runner

HOSTED = {"HOSTED_INFERENCE_URL": "https://hosted.example/v1",
          "HOSTED_INFERENCE_TOKEN": "placeholder"}
PROVIDER_KEYS = ("GOOGLE_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY",
                 "ANTHROPIC_API_KEY", "LITELLM_PROXY_API_KEY")


def resolve(env: dict, model: str = "gemini/gemini-3.8-flash") -> tuple[str, bool]:
    saved = dict(os.environ)
    try:
        for name in list(HOSTED) + list(PROVIDER_KEYS) + ["STIRRUP_INFERENCE"]:
            os.environ.pop(name, None)
        os.environ.update(env)
        resolved, hosted = runner.inference_env()
        return runner._model_name(resolved, model, hosted), hosted
    finally:
        os.environ.clear()
        os.environ.update(saved)


cases = [
    # Harbor's two modes, per the launcher UI:
    #   gateway = HOSTED_INFERENCE_URL + HOSTED_INFERENCE_TOKEN
    #   direct  = provider key + HOSTED_INFERENCE_TOKEN, no URL
    ("gateway mode", HOSTED, "litellm_proxy/gemini/gemini-3.8-flash", True),
    ("direct mode", {"HOSTED_INFERENCE_TOKEN": "placeholder", "GEMINI_API_KEY": "k"},
     "gemini/gemini-3.8-flash", False),
    ("direct mode, token only", {"HOSTED_INFERENCE_TOKEN": "placeholder"},
     "gemini/gemini-3.8-flash", False),
    ("hosted only", HOSTED, "litellm_proxy/gemini/gemini-3.8-flash", True),
    ("provider key only", {"GEMINI_API_KEY": "k"}, "gemini/gemini-3.8-flash", False),
    ("both, direct wins", {**HOSTED, "GEMINI_API_KEY": "k"},
     "gemini/gemini-3.8-flash", False),
    ("forced hosted", {**HOSTED, "GEMINI_API_KEY": "k", "STIRRUP_INFERENCE": "hosted"},
     "litellm_proxy/gemini/gemini-3.8-flash", True),
    ("forced direct", {**HOSTED, "STIRRUP_INFERENCE": "direct"},
     "gemini/gemini-3.8-flash", False),
    ("neither", {}, "gemini/gemini-3.8-flash", False),
]

for name, env, want_model, want_hosted in cases:
    model, hosted = resolve(env, "gemini/gemini-3.8-flash")
    status = "ok " if (model, hosted) == (want_model, want_hosted) else "FAIL"
    print(f"{status} {name:22s} -> hosted={hosted!s:5s} model={model}")
    assert (model, hosted) == (want_model, want_hosted), (name, model, hosted)

try:
    resolve({**HOSTED, "STIRRUP_INFERENCE": "nonsense"})
except ValueError as error:
    print(f"ok  bad STIRRUP_INFERENCE  -> {error}")
else:
    raise AssertionError("a bad STIRRUP_INFERENCE should be rejected")

print("\nALL CREDENTIAL CASES PASSED")
