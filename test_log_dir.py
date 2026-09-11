"""The agent log dir must stay writable by the task's collect hooks.

Harbor runs `capture_agent_artifacts.py` as a collect hook, which writes
/logs/agent/agent_artifacts.tar.gz.partial. Hosted runs that hook as a
different user than the agent, so a default-mode directory locks it out and
the verifier then has no artifacts to grade.

Run: uv run python test_log_dir.py
"""

from __future__ import annotations

import stat
import tempfile
from pathlib import Path

from stirrup_agent.agent import prepare_log_dir

root = Path(tempfile.mkdtemp())

# fresh directory
fresh = prepare_log_dir(root / "logs" / "agent")
mode = stat.S_IMODE(fresh.stat().st_mode)
print(f"fresh dir     : {fresh}  mode={oct(mode)}")
assert fresh.is_dir(), "log dir was not created"
assert mode & 0o222 == 0o222, f"not writable by all: {oct(mode)}"

# already exists with a restrictive mode, as when another user made it first
existing = root / "logs2" / "agent"
existing.mkdir(parents=True)
existing.chmod(0o700)
again = prepare_log_dir(existing)
mode = stat.S_IMODE(again.stat().st_mode)
print(f"existing dir  : {again}  mode={oct(mode)}")
assert mode & 0o222 == 0o222, f"restrictive dir not relaxed: {oct(mode)}"

# a collect hook running as another user must be able to create its file here
partial = again / "agent_artifacts.tar.gz.partial"
partial.write_bytes(b"x")
print(f"hook file     : wrote {partial.name}")
assert partial.exists()

# idempotent
prepare_log_dir(again)
print("idempotent    : ok")

print("\nALL LOG DIR CASES PASSED")
