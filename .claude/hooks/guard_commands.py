#!/usr/bin/env python3
"""PreToolUse guard for Bash and PowerShell. Stdlib only, cross-platform.

Reads the repo's git policy from .claude/TIER.md (`Git policy: main|pr`, `Deploys on push:
yes|no`) and answers on stdout with a permissionDecision: "deny" blocks, "ask" puts the command
to the user, no output allows. It parses shell syntax, so chaining (`a && git push`), `git -C`,
`cd x && ...`, subshells (`bash -c "..."`) and `$(...)` are judged like plain commands, while
text inside single quotes, heredocs and PowerShell here-strings (a commit message) is data, not
a command, unless it is fed to a shell or expands a substitution.

Both policies   deny  force-push, push --all/--mirror, filter-branch/filter-repo, gh pr merge,
                      terraform apply/destroy/import/taint and state edits, docker push,
                      npm publish, kubectl/az mutations, gh release/workflow/run/cache writes
                ask   git reset --hard, git clean -f, git branch -D, git worktree remove --force,
                      deleting a remote branch, git remote add/set-url/remove, gh repo create,
                      git stash clear, tawa deploy/promote/approve
main policy     allow push main, other branches, merge
                deny  push main when Deploys on push: yes (a live site: the user pushes it)
pr policy       deny  push to main/master, commit on main, merge into main (--ff-only is fine)
                allow push other branches, gh pr create
No readable policy counts as pr, the strict one. The nested project-management/ repo of private
PM mode may commit and push its own main under either policy. In a folder holding that nested
repo, it asks before switching to a branch that still tracks project-management/.

It fails safe: on an internal error it asks before anything that looks like a push or merge and
allows the rest, so a bug can never brick the shell. settings.json carries a deny list as backup.
"""
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

PROTECTED = {"main", "master"}
WRAPPERS = {"env", "sudo", "command", "nice", "time", "exec"}
SHELL_WRAPPERS = {"bash", "sh", "zsh", "cmd", "powershell", "pwsh"}
GIT_OPTS_WITH_ARG = {"-C", "-c", "--exec-path", "--git-dir", "--work-tree", "--namespace"}
# Never-allowed actions hidden in command substitution. Plain `git push` is left to the full
# segment parse, because whether a push is allowed depends on its target.
SUBST_RE = re.compile(
    r"[$`]\(?\s*(?:git\s+(?:filter-branch|filter-repo)|terraform\s+(?:apply|destroy|import))\b",
    re.IGNORECASE,
)
TF_MUTATE = {"apply", "destroy", "import", "taint", "untaint"}
AZ_MUTATE = {
    "create", "delete", "update", "set", "deploy", "restart", "start", "stop", "scale", "add",
    "remove", "purge", "regenerate", "renew", "import", "push", "up", "enable", "disable",
    "assign", "reset", "replace", "restore",
}
DENY, ASK = "deny", "ask"


class Ctx:
    def __init__(self, policy: str, deploys_on_push: bool):
        self.policy, self.deploys_on_push = policy, deploys_on_push


def read_policy(project_dir: str) -> Ctx:
    try:
        text = (Path(project_dir) / ".claude" / "TIER.md").read_text(encoding="utf-8")
    except OSError:
        return Ctx("pr", False)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    policy = re.search(r"^\W*Git policy:\**\s*`?(main|pr)\b", text, re.M | re.I)
    deploys = re.search(r"^\W*Deploys on push:\**\s*`?(yes|no)\b", text, re.M | re.I)
    return Ctx(policy.group(1).lower() if policy else "pr",
               bool(deploys and deploys.group(1).lower() == "yes"))


def git_out(cwd: str, *args: str) -> str | None:
    try:
        p = subprocess.run(["git", *args], cwd=cwd or None, capture_output=True, text=True,
                           timeout=5)
    except Exception:
        return None
    return p.stdout.strip() if p.returncode == 0 else None


def current_branch(cwd: str) -> str | None:
    return git_out(cwd, "symbolic-ref", "--short", "-q", "HEAD")


def is_nested_pm(cwd: str) -> bool:
    """The private-mode PM repo: its own git root, named project-management, inside a repo."""
    top = git_out(cwd, "rev-parse", "--show-toplevel")
    return bool(top) and Path(top).name == "project-management" and (
        Path(top).parent / ".claude").is_dir()


def holds_nested_pm(cwd: str) -> bool:
    """This folder's repo keeps a private-mode PM repo (the main folder, not a worktree)."""
    top = git_out(cwd, "rev-parse", "--show-toplevel")
    return bool(top) and (Path(top) / "project-management" / ".git").exists()


# --- shell parsing (from lca-pave-app's field-tested guard) -----------------------------------

def _without_single_quoted(command: str) -> str:
    return re.sub(r"'[^']*'", "  ", command)


# --- heredocs and here-strings: their bodies are data unless they feed a shell -------------------

HEREDOC_RE = re.compile(r"<<(-?)[ \t]*(?:(['\"])([^'\"\s]+)\2|(\\?)([A-Za-z0-9_]+))")
PS_HERESTRING_RE = re.compile(r"@(['\"])[ \t]*\r?\n(.*?)\r?\n\1@", re.S)
RUNS_TEXT = {"bash", "sh", "zsh", "dash", "ksh", "cmd", "powershell", "pwsh", "eval", "source",
             ".", "xargs", "ssh", "iex", "invoke-expression"}


def feeds_shell(line: str) -> bool:
    """Whether a command line hands text to something that runs it as commands."""
    for segment in split_segments(line):
        try:
            tokens = shlex.split(segment, posix=True)
        except ValueError:
            tokens = segment.split()
        if any(prog_name(t) in RUNS_TEXT for t in tokens):
            return True
    return False


def substitutions(text: str, backticks: bool = True) -> list[str]:
    """The commands inside $( ) (balanced) and, for bash, `...` in text the shell expands."""
    found, i = [], 0
    while (start := text.find("$(", i)) != -1:
        depth, j = 1, start + 2
        while j < len(text) and depth:
            depth += {"(": 1, ")": -1}.get(text[j], 0)
            j += 1
        found.append(text[start + 2:j - 1])
        i = j
    return found + (re.findall(r"`([^`]*)`", text) if backticks else [])


def lift_heredocs(command: str) -> tuple[str, list[str]]:
    """The command with heredoc and PowerShell here-string bodies taken out, plus whatever in
    those bodies still runs: the whole body when it feeds a shell, else the substitutions an
    unquoted body expands. So a commit message in `git commit -F - <<'EOF'` or `-m @'...'@` is
    data. A heredoc with no closing line lifts nothing: the command is judged as before."""
    extras: list[str] = []

    def here_string(m: re.Match) -> str:
        line_start = command.rfind("\n", 0, m.start()) + 1
        line_end = command.find("\n", m.end())
        around = command[line_start:m.start()] + command[m.end():None if line_end < 0 else line_end]
        if feeds_shell(around):
            extras.append(m.group(2))
        elif m.group(1) == '"':
            extras.extend(substitutions(m.group(2), backticks=False))
        return "''"

    command = PS_HERESTRING_RE.sub(here_string, command)
    out, stack, pending, i, n = [], ["code"], [], 0, len(command)
    while i < n:
        ch, ctx = command[i], stack[-1]
        if ctx == "single":
            if ch == "'":
                stack.pop()
        elif ch == "\\":
            out.append(command[i:i + 2])
            i += 2
            continue
        elif ctx == "double" and ch == '"':
            stack.pop()
        elif command.startswith("$(", i):
            stack.append("code")
            out.append("$(")
            i += 2
            continue
        elif ctx == "double":
            pass
        elif ch == "'":
            stack.append("single")
        elif ch == '"':
            stack.append("double")
        elif ch == "(":
            stack.append("code")
        elif ch == ")" and len(stack) > 1:
            stack.pop()
        elif command.startswith("<<<", i):
            out.append("<<<")
            i += 3
            continue
        elif command.startswith("<<", i) and (m := HEREDOC_RE.match(command, i)):
            pending.append((m.group(1) == "-", bool(m.group(2) or m.group(4)),
                            m.group(3) or m.group(5)))
            out.append(m.group(0))
            i = m.end()
            continue
        elif ch == "\n" and pending:  # the bodies start on the next line, in order
            line = "".join(out)
            shell = feeds_shell(line[line.rfind("\n") + 1:])
            j = i + 1
            for dash, quoted, word in pending:
                body = []
                while True:
                    if j >= n:
                        return command, extras  # never closed: lift nothing more
                    k = command.find("\n", j)
                    k = n if k < 0 else k
                    text = command[j:k].rstrip("\r")
                    j = k + 1
                    if (text.lstrip("\t") if dash else text) == word:
                        break
                    body.append(text)
                if shell:
                    extras.append("\n".join(body))
                elif not quoted:
                    extras.extend(substitutions("\n".join(body)))
            pending = []
            out.append("\n")
            i = j
            continue
        out.append(ch)
        i += 1
    return "".join(out), extras


def split_segments(command: str) -> list[str]:
    """Split on ; && || | newline and command-substitution starts, treating single-quoted text
    as literal: bash does no expansion inside '...', so a commit message is data."""
    segments, buf, in_single, i, n = [], [], False, 0, len(command)
    while i < n:
        ch = command[i]
        if in_single:
            buf.append(ch)
            in_single = ch != "'"
            i += 1
        elif ch == "'":
            in_single = True
            buf.append(ch)
            i += 1
        elif command[i:i + 2] in ("&&", "||", "$("):
            segments.append("".join(buf))
            buf = []
            i += 2
        elif ch in (";", "|", "`", "\n"):
            segments.append("".join(buf))
            buf = []
            i += 1
        else:
            buf.append(ch)
            i += 1
    segments.append("".join(buf))
    return [s.strip() for s in segments if s.strip()]


def program_tokens(segment: str) -> list[str]:
    try:
        tokens = shlex.split(segment, posix=True)
    except ValueError:
        tokens = segment.split()
    while tokens and (re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", tokens[0]) or tokens[0] in WRAPPERS):
        tokens = tokens[1:]
    return tokens


def prog_name(tok: str) -> str:
    name = tok.replace("\\", "/").split("/")[-1].lower()
    return name[:-4] if name.endswith(".exe") else name


def positional(tokens: list[str]) -> list[str]:
    return [t for t in tokens if not t.startswith("-")]


# --- checks -----------------------------------------------------------------------------------

def git_parts(tokens: list[str], cwd: str) -> tuple[str | None, list[str], str]:
    """(subcommand, its args, the directory git will run in)."""
    i = 1
    while i < len(tokens):
        t = tokens[i]
        if t == "-C" and i + 1 < len(tokens):
            cwd = os.path.join(cwd, tokens[i + 1])
            i += 2
        elif t in GIT_OPTS_WITH_ARG:
            i += 2
        elif t.startswith("-"):
            i += 1
        else:
            return t, tokens[i + 1:], cwd
    return None, [], cwd


def push_target(refspec: str, cwd: str) -> str:
    spec = refspec[1:] if refspec.startswith("+") else refspec
    dst = spec.split(":", 1)[1] if ":" in spec else spec
    dst = re.sub(r"^refs/heads/", "", dst).strip("\"'`() \t")
    if dst == "HEAD":
        dst = current_branch(cwd) or "HEAD"
    return dst


def check_push(args: list[str], cwd: str, ctx: Ctx):
    refspecs = positional(args)[1:]  # the first positional is the remote
    if any(a in ("--force", "-f", "--force-if-includes") or a.startswith("--force-with-lease")
           for a in args) or any(r.startswith("+") for r in refspecs):
        return DENY, "force-push rewrites pushed history; never allowed"
    if any(a in ("--all", "--mirror", "--branches") for a in args):
        return DENY, "push one branch at a time, not --all/--mirror"
    if any(a in ("--delete", "-d") for a in args) or any(r.startswith(":") for r in refspecs):
        return ASK, "this deletes a remote branch"
    if is_nested_pm(cwd):
        return None
    targets = [push_target(r, cwd) for r in refspecs] or [current_branch(cwd) or "HEAD"]
    if not any(t in PROTECTED or t == "HEAD" for t in targets):
        return None
    if ctx.policy != "main":
        if "HEAD" in targets:
            return ASK, "can't tell which branch this push targets (pr policy: never main)"
        return DENY, ("pr policy: main moves only by a merged PR. Push the branch and open a PR "
                      "(gh pr create); the user merges")
    if ctx.deploys_on_push:
        return DENY, ("pushing main deploys this site (Deploys on push: yes), so the user pushes "
                      "it: commit, then hand over `! git push origin main`")
    return None


def check_checkout(args: list[str], cwd: str):
    """Ask before checking out a commit that still tracks project-management/ in a folder that
    holds the private PM repo: git overwrites ignored files silently, and switching back
    deletes them."""
    if "--" in args or "--no-overwrite-ignore" in args or not holds_nested_pm(cwd):
        return None
    refs, skip_next = [], False
    for a in args:
        if skip_next:
            skip_next = False
        elif a in ("-c", "-C", "-b", "-B", "--create", "--force-create", "--orphan"):
            skip_next = True  # the new branch's name, not a commit
        elif not a.startswith("-"):
            refs.append(a)
    if refs and git_out(cwd, "ls-tree", "-d", "--name-only", refs[0], "project-management"):
        return ASK, (f"'{refs[0]}' still tracks project-management/: checking it out here "
                     "overwrites the private PM files, and switching back deletes them. Open it "
                     "in a worktree and merge main into it there (.claude/rules/git.md)")
    return None


def check_git(tokens: list[str], cwd: str, ctx: Ctx):
    sub, args, cwd = git_parts(tokens, cwd)
    if sub is None:
        return None
    if sub == "push":
        return check_push(args, cwd, ctx)
    if sub in ("switch", "checkout"):
        return check_checkout(args, cwd)
    if sub in ("filter-branch", "filter-repo"):
        return DENY, "rewriting history (filter-branch/filter-repo) is never allowed"
    if sub == "reset" and "--hard" in args:
        return ASK, "git reset --hard discards uncommitted changes"
    if sub == "clean" and any(re.match(r"^-[a-zA-Z]*f", a) or a == "--force" for a in args) \
            and not any(a in ("-n", "--dry-run") for a in args):
        return ASK, "git clean -f deletes untracked files"
    if sub == "branch" and ("-D" in args or ("--force" in args and "--delete" in args)):
        return ASK, "force-deleting a branch can lose unmerged work"
    if sub == "worktree" and args[:1] == ["remove"] and any(a in ("-f", "--force") for a in args):
        return ASK, "force-removing a worktree discards its uncommitted changes"
    if sub == "remote" and args[:1] and args[0] in ("add", "set-url", "remove", "rm", "rename"):
        return ASK, "this changes where the repo pushes; remotes are the user's call"
    if sub == "stash" and args[:1] == ["clear"]:
        return ASK, "the stash stack is shared by every worktree of this repo"
    if ctx.policy != "main" and sub in ("commit", "merge"):
        if sub == "merge" and any(a in ("--ff-only", "--abort", "--continue", "--quit")
                                  for a in args):
            return None
        if current_branch(cwd) in PROTECTED and not is_nested_pm(cwd):
            return DENY, (f"pr policy: no {sub} on main. Work on a branch "
                          "(feat/F-n-slug); main moves only by a merged PR")
    return None


def check_gh(tokens: list[str]):
    rest = positional(tokens[1:])
    a, b = (rest + [None, None])[:2]
    if a == "pr" and b == "merge":
        return DENY, "Claude never merges a PR; the user does"
    if a == "repo" and b == "create":
        return ASK, "creating a remote repo is the user's call"
    if (a, b) in {("workflow", "run"), ("run", "rerun"), ("run", "cancel"), ("cache", "delete")} \
            or (a == "release" and b in ("create", "delete", "upload", "edit")):
        return DENY, f"gh {a} {b} triggers or alters CI/releases; the user runs it"
    return None


def check_other(prog: str, tokens: list[str]):
    rest = positional(tokens[1:])
    a, b = (rest + [None, None])[:2]
    if prog == "terraform":
        if a in TF_MUTATE or (a == "state" and b in ("rm", "mv", "push", "replace-provider")):
            return DENY, f"terraform {a} changes infrastructure; prepare it, the user runs it"
    elif prog == "az" and any(t in AZ_MUTATE for t in rest):
        return DENY, "this az command changes a cloud resource; read-only az is fine"
    elif prog == "kubectl" and a in ("apply", "delete", "edit", "scale", "patch", "rollout"):
        return DENY, "changing a Kubernetes cluster is the user's call"
    elif (prog, a) in {("docker", "push"), ("npm", "publish"), ("swa", "deploy")}:
        return DENY, f"{prog} {a} publishes or deploys; the user runs it"
    elif prog == "tawa" and a in ("deploy", "promote", "approve"):
        return ASK, f"tawa {a} ships to an environment; the user confirms"
    return None


def evaluate(command: str, cwd: str, ctx: Ctx, depth: int = 0):
    """Return (decision, reason) for the strictest finding, or None to allow."""
    if depth > 3:
        return None
    command, extras = lift_heredocs(command)
    if SUBST_RE.search(_without_single_quoted(command)):
        return DENY, "a reserved git/terraform action inside command substitution"
    found = [evaluate(e, cwd, ctx, depth + 1) for e in extras]
    for segment in split_segments(command):
        tokens = program_tokens(segment)
        if not tokens:
            continue
        prog = prog_name(tokens[0])
        if prog in ("cd", "set-location", "pushd") and len(tokens) > 1:
            cwd = os.path.join(cwd, tokens[1])
            continue
        if prog == "git":
            found.append(check_git(tokens, cwd, ctx))
        elif prog == "gh":
            found.append(check_gh(tokens))
        elif prog in SHELL_WRAPPERS:
            found += [evaluate(t, cwd, ctx, depth + 1) for t in tokens[1:] if " " in t]
        else:
            found.append(check_other(prog, tokens))
    found = [f for f in found if f]
    return next((f for f in found if f[0] == DENY), found[0] if found else None)


def main() -> int:
    try:
        event = json.load(sys.stdin)
    except Exception:
        return 0
    if event.get("tool_name", "Bash") not in ("Bash", "PowerShell"):
        return 0
    command = (event.get("tool_input") or {}).get("command") or ""
    cwd = event.get("cwd") or os.getcwd()
    try:
        ctx = read_policy(os.environ.get("CLAUDE_PROJECT_DIR") or cwd)
        result = evaluate(command, cwd, ctx)
    except Exception as e:  # fail safe, never brick the shell
        result = (ASK, f"guard error ({e}); confirm by hand") if re.search(
            r"\b(push|merge)\b", command) else None
    if result:
        decision, reason = result
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": decision,
            "permissionDecisionReason": f"{'Blocked' if decision == DENY else 'Check'}: {reason}.",
        }}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
