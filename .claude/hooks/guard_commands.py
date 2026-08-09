#!/usr/bin/env python3
"""PreToolUse guard for the Bash tool. Cross-platform (Windows/macOS/Linux).

Exit 2 = block; stderr goes back to Claude. Parses shell syntax properly, so:
  BLOCKED: git push / git -C . push / cd x && git push / bash -c "git push"
           terraform apply / terraform -chdir=infra apply
           git merge, git rebase, git reset --hard
  ALLOWED: git commit -m "docs: explain git push policy", echo "git push",
           git status, terraform plan
This is a seatbelt, not a wall — real enforcement for push/merge lives in
GitHub branch protection (see GUIDE.md).
"""
import json
import re
import shlex
import sys

BLOCK_GIT = {"push", "merge", "rebase"}
BLOCK_TF = {"apply", "destroy", "import"}
GIT_OPTS_WITH_ARG = {"-C", "-c", "--exec-path", "--git-dir", "--work-tree", "--namespace"}
SHELL_WRAPPERS = {"bash", "sh", "zsh", "cmd", "cmd.exe", "powershell", "powershell.exe", "pwsh"}
# Coarse net for command substitution: $(git push), `terraform apply`, etc.
SUBST_RE = re.compile(
    r"[$`]\(?\s*(?:git\s+(?:push|merge|rebase)|terraform\s+(?:apply|destroy|import))",
    re.IGNORECASE,
)


def split_segments(cmd: str):
    return [s for s in re.split(r"(?:&&|\|\||;|\||\r?\n)", cmd) if s.strip()]


def toks(segment: str):
    try:
        return shlex.split(segment, posix=True)
    except ValueError:
        return segment.split()


def prog_name(token: str) -> str:
    return token.replace("\\", "/").rsplit("/", 1)[-1].lower()


def git_subcommand(tokens):
    i = 1
    while i < len(tokens):
        t = tokens[i]
        if t in GIT_OPTS_WITH_ARG:
            i += 2
            continue
        if t.startswith("-"):
            i += 1
            continue
        return t, tokens[i + 1 :]
    return None, []


def check(cmd: str, depth: int = 0):
    if depth > 3:
        return None
    if SUBST_RE.search(cmd):
        return "command substitution containing a reserved git/terraform action"
    for seg in split_segments(cmd):
        tokens = toks(seg)
        if not tokens:
            continue
        prog = prog_name(tokens[0])
        if prog in ("git", "git.exe"):
            sub, rest = git_subcommand(tokens)
            if sub in BLOCK_GIT:
                return f"git {sub} — the user does all pushing/merging/rebasing"
            if sub == "reset" and "--hard" in rest:
                return "git reset --hard — destructive; ask first"
        elif prog in ("terraform", "terraform.exe"):
            sub = next((t for t in tokens[1:] if not t.startswith("-")), None)
            if sub in BLOCK_TF:
                return f"terraform {sub} — needs explicit per-instance permission"
        elif prog in SHELL_WRAPPERS:
            for t in tokens[1:]:
                if " " in t:  # quoted command string handed to a subshell
                    reason = check(t, depth + 1)
                    if reason:
                        return reason
    return None


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except Exception:
        return 0
    cmd = (data.get("tool_input") or {}).get("command") or ""
    reason = check(cmd)
    if reason:
        print(f"BLOCKED: {reason}.", file=sys.stderr)
        print("This action is reserved for the user. Ask them to run it, or get explicit permission.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
