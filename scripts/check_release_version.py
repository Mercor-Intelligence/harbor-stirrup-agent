"""Fail a release whose tag, manifest and package version disagree.

Harbor shows the manifest version in the Hub and the agent reports its own over
ACP, so a mismatch means a run is labelled with a version nobody can check out.
"""

import json
import pathlib
import sys
from importlib.metadata import version

tag = (sys.argv[1] if len(sys.argv) > 1 else "").removeprefix("v")
root = pathlib.Path(__file__).resolve().parent.parent
manifest = json.loads((root / "harbor-agent.json").read_text())["version"]
package = version("stirrup-agent")

print(f"tag={tag or '(none)'} manifest={manifest} package={package}")
if not tag:
    raise SystemExit("no tag given")
if len({tag, manifest, package}) != 1:
    raise SystemExit(
        f"release version mismatch: tag {tag}, manifest {manifest}, package {package}"
    )
print("versions agree")
