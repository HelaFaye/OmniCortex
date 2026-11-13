"""Is someone using this computer? (The code lives with the remote agent, which hosts carry on their own.)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "remote"))
from omnicortex_remote.presence import *  # noqa: E402,F401,F403
from omnicortex_remote.presence import in_use, snapshot  # noqa: E402,F401

if __name__ == "__main__":
    import json
    s = snapshot()
    print(json.dumps(s, indent=1))
    print("in use:", in_use(s))
