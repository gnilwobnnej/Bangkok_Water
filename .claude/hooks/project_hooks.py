"""Claude Code hooks for this project (configured in .claude/settings.json).

    python .claude/hooks/project_hooks.py protect   PreToolUse Edit|Write: refuse writes to data/raw/, secrets, key files
    python .claude/hooks/project_hooks.py tests     PostToolUse Edit|Write: run the tests for an edited src/ or scripts/ file
    python .claude/hooks/project_hooks.py push      PreToolUse Bash(git push*): refuse the push unless pytest passes

Reads the hook's JSON from stdin. Exit code 2 blocks the action and shows stderr to Claude.
Python 3.9 compatible, standard library only.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KEY_FILE = re.compile(r"(key|credential|service[-_]?account|secret)[^/\\]*\.json$", re.I)


def payload() -> dict:
    try:
        return json.loads(sys.stdin.buffer.read().decode("utf-8") or "{}")
    except ValueError:
        return {}


def rel_path(data: dict) -> str | None:
    p = (data.get("tool_input") or {}).get("file_path")
    if not p:
        return None
    path = Path(p)
    try:
        path = path.resolve().relative_to(ROOT)
    except ValueError:
        return None  # outside the project: not ours to guard
    return path.as_posix()


def protect(data: dict) -> int:
    p = rel_path(data)
    if p is None:
        return 0
    reason = None
    if p.startswith("data/raw/"):
        reason = "data/raw/ holds downloads that scripts recreate; change the prepare/fetch script instead."
    elif p.endswith(".streamlit/secrets.toml"):
        reason = ".streamlit/secrets.toml holds secrets and must not be written by Claude."
    elif KEY_FILE.search(p):
        reason = (f"{p} looks like a key/credential file. Keys belong in GitHub secrets "
                  "(GEE_SERVICE_ACCOUNT_KEY), never in this folder.")
    if reason:
        print(f"Blocked by project hook: {reason}", file=sys.stderr)
        return 2
    return 0


def tests_for(p: str) -> list:
    """Test files that import the edited module (src/<m>.py or scripts/<m>.py)."""
    stem = Path(p).stem
    pat = re.compile(rf"(from src\.{stem} import|from src import [^\n]*\b{stem}\b|import {stem}\b)")
    found = []
    for t in sorted((ROOT / "tests").glob("test_*.py")):
        if t.stem == f"test_{stem}" or pat.search(t.read_text(encoding="utf-8")):
            found.append(t)
    return found


def tests(data: dict) -> int:
    p = rel_path(data)
    if not p or not p.endswith(".py") or not (p.startswith("src/") or p.startswith("scripts/")):
        return 0
    files = tests_for(p)
    if not files:
        print(json.dumps({"hookSpecificOutput": {"hookEventName": "PostToolUse",
                          "additionalContext": f"No tests import {p}; consider adding one in tests/."}}))
        return 0
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", *map(str, files)], cwd=ROOT,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    names = ", ".join(f.name for f in files)
    if r.returncode == 0:
        print(json.dumps({"hookSpecificOutput": {"hookEventName": "PostToolUse",
                          "additionalContext": f"Tests passed after editing {p}: {names}"}}))
        return 0
    tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-25:])
    print(json.dumps({"decision": "block",
                      "reason": f"Tests failed after editing {p} ({names}):\n{tail}"}))
    return 0


def push(data: dict) -> int:
    cmd = (data.get("tool_input") or {}).get("command", "")
    if not re.search(r"\bgit\s+push\b", cmd):
        return 0
    r = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode == 0:
        return 0
    tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-25:])
    print(f"Blocked by project hook: pytest fails, so the push was stopped. Fix the tests first.\n{tail}",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    handler = {"protect": protect, "tests": tests, "push": push}.get(mode)
    sys.exit(handler(payload()) if handler else 0)
