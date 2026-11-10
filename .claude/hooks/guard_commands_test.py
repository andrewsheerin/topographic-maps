"""Behaviour matrix for guard_commands.py under both git policies.  Run: python -m pytest -q

Each case runs in a throwaway git repo whose .claude/TIER.md sets the policy, on branch main
unless the case says otherwise, and expects allow / ask / deny.
"""
import importlib.util
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("guard_commands", HERE / "guard_commands.py")
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)

GIT_ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}


def git(cwd, *args):
    subprocess.run(["git", "-c", "init.defaultBranch=main", "-c", "commit.gpgsign=false", *args],
                   cwd=cwd, check=True, capture_output=True, env=GIT_ENV)


def make_repo(path, tier_text=None):
    path.mkdir(parents=True)
    if tier_text is not None:
        (path / ".claude").mkdir()
        (path / ".claude/TIER.md").write_text(tier_text, encoding="utf-8")
    git(path, "init", "-q")
    git(path, "commit", "-q", "--allow-empty", "-m", "init")
    return path


@pytest.fixture(scope="module")
def repos(tmp_path_factory):
    base = tmp_path_factory.mktemp("guard")
    tier = "**TIER: L2**\n\nGit policy: {}\nPM mode: committed\nDeploys on push: {}\n"
    out = {
        "main": make_repo(base / "main", tier.format("main", "no")),
        "main-deploys": make_repo(base / "deploys", tier.format("main", "yes")),
        "pr": make_repo(base / "pr", tier.format("pr", "no")),
        "none": make_repo(base / "none"),  # no TIER.md: treated as pr
    }
    feat = make_repo(base / "pr-feat", tier.format("pr", "no"))
    git(feat, "switch", "-q", "-c", "feat/F-1-x")
    out["pr-feat"] = feat
    # Private PM mode: project-management/ is its own repo inside a pr host.
    make_repo(out["pr"] / "project-management")
    return out


def decide(repos, which, command):
    root = str(repos[which])
    result = guard.evaluate(command, root, guard.read_policy(root))
    return result[0] if result else "allow"


BOTH = [
    ("git status && git log --oneline -5", "allow"),
    ("git push -u origin feat/F-1-x", "allow"),
    ("gh pr create --fill", "allow"),
    ("echo 'git push --force origin main && gh pr merge 8'", "allow"),
    ("git push --force origin feat/x", "deny"),
    ("git push -f", "deny"),
    ("git push origin +feat/x", "deny"),
    ("git push --force-with-lease origin feat/x", "deny"),
    ("git push --mirror origin", "deny"),
    ("git push --all origin", "deny"),
    ("git filter-branch --tree-filter x HEAD", "deny"),
    ("gh pr merge 8 --squash", "deny"),
    ("gh release create v1.0.0", "deny"),
    ("terraform apply", "deny"),
    ("terraform -chdir=infra plan", "allow"),
    ("bash -c \"git push --force origin x\"", "deny"),
    ("echo $(git filter-branch)", "deny"),
    ("git push origin --delete feat/x", "ask"),
    ("git reset --hard HEAD~1", "ask"),
    ("git reset --soft HEAD~1", "allow"),
    ("git clean -fd", "ask"),
    ("git clean -n", "allow"),
    ("git branch -D feat/x", "ask"),
    ("git branch -d feat/x", "allow"),
    ("git worktree remove --force .claude/worktrees/x", "ask"),
    ("git worktree add .claude/worktrees/x -b feat/F-2-y", "allow"),
    ("git remote add origin https://github.com/x/y.git", "ask"),
    ("gh repo create x --private", "ask"),
    ("tawa deploy --prod", "ask"),
    ("docker push registry/x", "deny"),
]


@pytest.mark.parametrize("policy", ["main", "pr"])
@pytest.mark.parametrize("command,expected", BOTH)
def test_both_policies(repos, policy, command, expected):
    assert decide(repos, policy, command) == expected


@pytest.mark.parametrize("which,command,expected", [
    # main policy: Claude commits on main, merges locally and pushes main after verify.
    ("main", "git push origin main", "allow"),
    ("main", "git push", "allow"),
    ("main", "git push origin HEAD", "allow"),
    ("main", "git commit -m x", "allow"),
    ("main", "git merge feat/F-1-x", "allow"),
    ("main-deploys", "git push origin main", "ask"),
    ("main-deploys", "git push", "ask"),
    ("main-deploys", "git push origin feat/F-1-x", "allow"),
    # pr policy: never push to main, never commit or merge on main, never merge a PR.
    ("pr", "git push origin main", "deny"),
    ("pr", "git push origin master", "deny"),
    ("pr", "git push origin HEAD:main", "deny"),
    ("pr", "git push origin feat/x:refs/heads/main", "deny"),
    ("pr", "git push", "deny"),
    ("pr", "git push origin HEAD", "deny"),
    ("pr", "git -C . push origin main", "deny"),
    ("pr", "git add -A && git push origin main", "deny"),
    ("pr", "git commit -m x", "deny"),
    ("pr", "git merge feat/F-1-x", "deny"),
    ("pr", "git merge --ff-only origin/main", "allow"),
    ("pr", "git pull --ff-only", "allow"),
    ("pr-feat", "git commit -m x", "allow"),
    ("pr-feat", "git commit -m 'docs: explain git push origin main and gh pr merge'", "allow"),
    ("pr-feat", "git push", "allow"),
    ("pr-feat", "git merge main", "allow"),
    ("none", "git push origin main", "deny"),
    # private PM mode: the nested project-management/ repo pushes its own main.
    ("pr", "git -C project-management commit -m 'docs(pm): wrap'", "allow"),
    ("pr", "git -C project-management push origin main", "allow"),
    ("pr", "cd project-management && git push origin main", "allow"),
])
def test_policy_specific(repos, which, command, expected):
    assert decide(repos, which, command) == expected


def run_main(repos, command, which="pr", tool="Bash"):
    event = json.dumps({"tool_name": tool, "tool_input": {"command": command},
                        "cwd": str(repos[which])})
    env = {**os.environ, "CLAUDE_PROJECT_DIR": str(repos[which])}
    out = subprocess.run([sys.executable, str(HERE / "guard_commands.py")], input=event,
                         capture_output=True, text=True, env=env).stdout.strip()
    return json.loads(out)["hookSpecificOutput"]["permissionDecision"] if out else "allow"


def test_hook_protocol_end_to_end(repos):
    assert run_main(repos, "git push origin main") == "deny"
    assert run_main(repos, "git push origin main", tool="PowerShell") == "deny"
    assert run_main(repos, "git push origin main", which="main") == "allow"
    assert run_main(repos, "git status") == "allow"
    assert run_main(repos, "git push origin main", tool="Read") == "allow"


def test_fails_safe_on_internal_error(repos, monkeypatch, capsys):
    def boom(*a, **k):
        raise RuntimeError("bug")
    monkeypatch.setattr(guard, "evaluate", boom)
    for command, expected in (("git push origin feat/x", "ask"), ("ls -la", "allow")):
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(
            {"tool_name": "Bash", "tool_input": {"command": command}, "cwd": str(repos["main"])})))
        assert guard.main() == 0
        out = capsys.readouterr().out.strip()
        got = json.loads(out)["hookSpecificOutput"]["permissionDecision"] if out else "allow"
        assert got == expected
