#!/usr/bin/env python3
"""pm_check: is this repo's project-management/ on the app-factory v3 standard?

Usage:  python pm_check.py <repo> [--json]
Exit:   0 = no errors (warnings allowed) · 1 = errors · 2 = bad usage

In a linked worktree of a private-mode repo, project-management/ lives only in the main folder.
find_pm() finds it there; the SessionStart hook imports it from this file and verify.py runs
this file, so the rule exists once.

Standard library only, Python 3.10+. Reads, never writes. The rules it checks are written out
in app-factory's STANDARD.md. Canonical copy: app-factory/tools/pm_check.py. The copy at
template/.claude/skills/cleanup/pm_check.py must stay byte-identical (a test enforces it).

Suppress a finding for a file that code depends on with a line in .claude/TIER.md:
    - pm_check ignore: project-management/reports/*_data-quality.md (written by cli.py)
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

PROFILES = {"app", "paper", "engagement", "partner", "site"}
GIT_POLICIES = {"main", "pr"}
PM_MODES = {"committed", "private"}
FRONTMATTER_KEYS = ("pm", "profile", "git_policy", "pm_mode", "updated", "branch", "phase",
                    "wip", "blocked_on")
CURRENT_MAX_LINES = 20
PLAN_MAX_LINES = 500
PM_FILES = ("CURRENT.md", "FEATURES.md", "DECISIONS.md", "CHORES.md", "PLAN.md")
RENAMED = {"FEATURE_REGISTER.md": "FEATURES.md", "plans": "PLAN.md"}  # in v3.2
REGISTER_STATUSES = {"IDEA", "PLANNED", "WIP", "BLOCKED", "DONE", "DROPPED"}
NEEDS_YOU_STATUSES = {"OPEN", "PROPOSED"}
RESOLVED_STATUSES = {"RESOLVED", "SUPERSEDED"}
CHORE_STATUSES = {"TODO", "DONE", "DROPPED"}
NONE_VALUES = {"", "-", "—", "none"}
ID_RES = {
    "F": re.compile(r"^F-[1-9]\d*[a-z]?$"),
    "D": re.compile(r"^D-[1-9]\d*$"),
    "C": re.compile(r"^C-[1-9]\d*$"),
}
REPORT_RES = {
    "sessions": re.compile(r"^\d{4}-\d{2}-\d{2}-\d{4}(-[a-z0-9]+)*\.md$"),
    "phases": re.compile(r"^phase-[A-Za-z0-9]+(-[a-z0-9]+)*\.md$"),
    "topics": re.compile(r"^\d{4}-\d{2}-\d{2}(-[a-z0-9]+)+\.md$"),
}
NEXT_PROMPT_RE = re.compile(r"^##\s+Next session prompt", re.IGNORECASE | re.MULTILINE)
V1_MARKERS = ("tasks/tasks.md", "tasks.md", "chores/decisions.md", "chores/manual-tasks.md")
SEP_RE = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class Report:
    def __init__(self, repo: Path, ignores: list[str], pm: Path | None = None):
        self.repo, self.ignores = repo, ignores
        self.pm = pm or repo / "project-management"
        self.errors: list[dict] = []
        self.warnings: list[dict] = []
        self.skipped: list[str] = []
        self.notes: list[str] = []
        self.ignored = 0

    def rel(self, path) -> str | None:
        if not isinstance(path, Path):
            return path
        for base in (self.repo, self.pm.parent):  # PM files may sit in the main folder
            try:
                return path.relative_to(base).as_posix()
            except ValueError:
                pass
        return path.as_posix()

    def _add(self, bucket, code, path, line, message):
        rel = self.rel(path)
        if rel and any(fnmatch.fnmatch(rel, g) for g in self.ignores):
            self.ignored += 1
            return
        bucket.append({"code": code, "path": rel, "line": line, "message": message})

    def error(self, code, path, line, message):
        self._add(self.errors, code, path, line, message)

    def warn(self, code, path, line, message):
        self._add(self.warnings, code, path, line, message)


# --- markdown ---------------------------------------------------------------------------------

def read_lines(path: Path) -> list[str]:
    """File lines with HTML comments and fenced code blanked, so line numbers still match."""
    text = path.read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"<!--.*?-->", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S)
    lines, in_fence = text.splitlines(), False
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
            lines[i] = ""
        elif in_fence:
            lines[i] = ""
    return lines


def split_cells(row: str) -> list[str]:
    s = row.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith("\\|"):
        s = s[:-1]
    return [c.strip() for c in re.split(r"(?<!\\)\|", s)]


class Table:
    def __init__(self, header: list[str], line: int, section: str):
        self.header = [h.strip("* ").lower() for h in header]
        self.line, self.section = line, section
        self.rows: list[tuple[int, list[str]]] = []

    def col(self, *names: str) -> int | None:
        for n in names:
            if n in self.header:
                return self.header.index(n)
        return None

    def cell(self, cells: list[str], *names: str) -> str:
        i = self.col(*names)
        return cells[i].strip("* ") if i is not None and i < len(cells) else ""


def parse_tables(lines: list[str], path: Path, rep: Report) -> list[Table]:
    tables, section, i, n = [], "", 0, len(lines)
    while i < n:
        ln = lines[i]
        if ln.startswith("## "):
            section = ln[3:].strip().lower()
        is_row = ln.lstrip().startswith("|")
        if is_row and i + 1 < n and SEP_RE.match(lines[i + 1].strip()):
            table = Table(split_cells(ln), i + 1, section)
            width = len(table.header)
            j = i + 2
            while j < n and lines[j].lstrip().startswith("|"):
                row = lines[j].rstrip()
                cells = split_cells(row)
                if not row.endswith("|") or row.endswith("\\|"):
                    rep.error("table", path, j + 1,
                              "row does not end with '|' (a line break inside a cell?)")
                elif len(cells) != width:
                    rep.error("table", path, j + 1,
                              f"{len(cells)} cells but the header has {width} "
                              "(an unescaped '|' in a cell? write it as \\|)")
                table.rows.append((j + 1, cells))
                j += 1
            if j < n and lines[j].strip() and "|" in lines[j]:
                rep.error("table", path, j + 1,
                          "text containing '|' right after a table (a row broken across lines?)")
            tables.append(table)
            i = j
            continue
        if is_row:
            rep.error("table", path, i + 1,
                      "table row outside a table (missing header, or a broken table above)")
        i += 1
    return tables


def sections(lines: list[str]) -> dict[str, tuple[int, int]]:
    """'## ' heading (lowercased) -> (first line index, end index exclusive)."""
    heads = [(i, ln[3:].strip().lower()) for i, ln in enumerate(lines) if ln.startswith("## ")]
    out = {}
    for k, (i, name) in enumerate(heads):
        end = heads[k + 1][0] if k + 1 < len(heads) else len(lines)
        out[name] = (i, end)
    return out


def frontmatter(path: Path) -> tuple[dict | None, list[str]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines or lines[0].strip() != "---":
        return None, lines
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            fm = {}
            for ln in lines[1:i]:
                if ":" in ln:
                    k, v = ln.split(":", 1)
                    fm[k.strip()] = v.strip().strip("\"'")
            return fm, lines[i + 1:]
    return None, lines


def is_none(value: str) -> bool:
    return value.strip().lower() in NONE_VALUES


def check_id(rep, path, line, kind, value, ids_seen):
    if not ID_RES[kind].match(value):
        rep.error("id-style", path, line,
                  f"'{value}' is not a v3 ID (expected {kind}-n, unpadded); "
                  "/pm-upgrade maps old IDs and keeps the old one in Notes")
    elif value in ids_seen:
        rep.error("duplicate-id", path, line, f"{value} appears more than once")
    ids_seen.add(value)


# --- the PM files -----------------------------------------------------------------------------

def read_tier(repo: Path) -> tuple[dict, list[str]]:
    p = repo / ".claude" / "TIER.md"
    if not p.is_file():
        return {}, []
    text = "\n".join(read_lines(p))
    found = {}
    for key, label in (("git_policy", "Git policy"), ("pm_mode", "PM mode")):
        m = re.search(rf"^\W*{label}:\**\s*`?([A-Za-z]+)", text, re.MULTILINE)
        if m:
            found[key] = m.group(1).lower()
    ignores = re.findall(r"^\s*[-*]?\s*pm_check ignore:\s*`?([^\s`]+)", text, re.MULTILINE)
    return found, ignores


def check_current(pm: Path, rep: Report, tier: dict) -> dict:
    path = pm / "CURRENT.md"
    fm, body = frontmatter(path)
    if fm is None:
        rep.error("frontmatter", path, 1, "no frontmatter (--- block with pm, profile, git_policy, "
                  "pm_mode, updated, branch, phase, wip, blocked_on)")
        fm = {}
    else:
        for key in FRONTMATTER_KEYS:
            if key not in fm:
                rep.error("frontmatter", path, 1, f"frontmatter is missing '{key}'")
        checks = (("pm", {"v3"}), ("profile", PROFILES), ("git_policy", GIT_POLICIES),
                  ("pm_mode", PM_MODES))
        for key, allowed in checks:
            if key in fm and fm[key] not in allowed:
                rep.error("frontmatter", path, 1,
                          f"{key}: '{fm[key]}' is not one of {', '.join(sorted(allowed))}")
        if "updated" in fm and not DATE_RE.match(fm["updated"]):
            rep.error("frontmatter", path, 1, f"updated: '{fm['updated']}' is not YYYY-MM-DD")
        for key in ("git_policy", "pm_mode"):
            if key in fm and key in tier and fm[key] != tier[key]:
                rep.error("policy-mismatch", path, 1,
                          f"{key} is '{fm[key]}' here but '{tier[key]}' in .claude/TIER.md")
    while body and not body[-1].strip():
        body.pop()
    if len(body) > CURRENT_MAX_LINES:
        rep.error("current-length", path, None,
                  f"{len(body)} lines below the frontmatter (limit {CURRENT_MAX_LINES}); "
                  "history belongs in a session report")
    return fm


def check_register(pm: Path, rep: Report) -> dict[str, str]:
    path = pm / "FEATURES.md"
    statuses, seen = {}, set()
    for t in parse_tables(read_lines(path), path, rep):
        if t.col("id") is None or t.col("status") is None:
            continue
        if t.col("ref") is None:
            old = "Commit(s)" if t.col("commit(s)", "commit", "commits") is not None else None
            rep.error("header", path, t.line, "register table has no Ref column"
                      + (f" (rename {old} to Ref)" if old else ""))
        for line, cells in t.rows:
            fid, status = t.cell(cells, "id"), t.cell(cells, "status").upper()
            check_id(rep, path, line, "F", fid, seen)
            if status not in REGISTER_STATUSES:
                rep.error("status", path, line, f"{fid}: status '{status}' is not one of "
                          + ", ".join(sorted(REGISTER_STATUSES)))
            if status == "DONE" and t.col("ref") is not None and is_none(t.cell(cells, "ref")):
                rep.error("ref-missing", path, line,
                          f"{fid} is DONE with an empty Ref (commit hash, PR number or file)")
            statuses[fid] = status
    wip = [f for f, s in statuses.items() if s == "WIP"]
    if len(wip) > 1:
        rep.error("wip-count", path, None, f"{len(wip)} items are WIP ({', '.join(wip)}); "
                  "one at a time, the rest go to PLANNED or BLOCKED")
    return statuses


def check_decisions(pm: Path, rep: Report) -> dict[str, str]:
    path = pm / "DECISIONS.md"
    lines = read_lines(path)
    secs = sections(lines)
    found: dict[str, str] = {}
    seen: set[str] = set()
    needs = next((v for k, v in secs.items() if k.startswith("needs you")), None)
    if needs is None:
        rep.error("missing-section", path, None, "no '## Needs you' section")
    else:
        start, end = needs
        for i in range(start, end):
            m = re.match(r"^###\s+(\S+?):?\s", lines[i] + " ")
            if not m:
                continue
            did, status = m.group(1), None
            for k in range(i + 1, end):
                if lines[k].startswith("#"):
                    break
                sm = re.search(r"\*\*Status:\*\*\s*\**([A-Za-z]+)", lines[k])
                if sm:
                    status = sm.group(1).upper()
                    break
            check_id(rep, path, i + 1, "D", did, seen)
            if status is None:
                rep.error("status", path, i + 1, f"{did} has no **Status:** line")
            elif status not in NEEDS_YOU_STATUSES:
                rep.error("needs-you", path, i + 1, f"{did} is {status} but Needs you holds only "
                          "OPEN and PROPOSED; move it to Resolved and its text to "
                          "archive/DECISIONS-answered.md")
            found[did] = status or ""
    for t in parse_tables(lines, path, rep):
        if t.col("id") is None:
            continue
        in_needs = t.section.startswith("needs you")
        allowed = NEEDS_YOU_STATUSES if in_needs else RESOLVED_STATUSES
        for line, cells in t.rows:
            did = t.cell(cells, "id")
            check_id(rep, path, line, "D", did, seen)
            status = t.cell(cells, "status").upper().split(" ")[0] if t.col("status") is not None else ""
            if status not in allowed:
                code = "needs-you" if in_needs else "status"
                rep.error(code, path, line, f"{did}: status '{status}' here must be one of "
                          + ", ".join(sorted(allowed)))
            found[did] = status
    return found


def check_chores(pm: Path, rep: Report) -> dict[str, str]:
    path = pm / "CHORES.md"
    found, seen = {}, set()
    for t in parse_tables(read_lines(path), path, rep):
        if t.col("id") is None or t.col("status") is None:
            continue
        if t.col("due") is None:
            rep.error("header", path, t.line, "chores table has no Due column")
        for line, cells in t.rows:
            cid, status = t.cell(cells, "id"), t.cell(cells, "status").upper()
            check_id(rep, path, line, "C", cid, seen)
            if status not in CHORE_STATUSES:
                rep.error("status", path, line, f"{cid}: status '{status}' is not one of "
                          + ", ".join(sorted(CHORE_STATUSES)))
            due = t.cell(cells, "due")
            if t.col("due") is not None and not is_none(due) and not DATE_RE.match(due):
                rep.error("due", path, line, f"{cid}: Due '{due}' is not YYYY-MM-DD")
            found[cid] = status
    return found


def check_plan(pm: Path, rep: Report) -> None:
    path = pm / "PLAN.md"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    if len(lines) > PLAN_MAX_LINES:  # a soft target: bigger projects may run longer
        rep.warn("plan-length", path, None,
                 f"{len(lines)} lines (target {PLAN_MAX_LINES}); check that detail which "
                 "could live with the code, or a one-feature plan, hasn't crept in")


def check_reports(pm: Path, rep: Report) -> None:
    reports = pm / "reports"
    if not reports.is_dir():
        return
    for p in sorted(reports.rglob("*")):
        if p.is_dir() or p.name == ".gitkeep":
            continue
        rel = p.relative_to(reports).parts
        rx = REPORT_RES.get(rel[0]) if len(rel) == 2 else None
        if rx is None:
            rep.error("report-name", p, None, "reports live in reports/sessions/, "
                      "reports/phases/ or reports/topics/ (one level deep)")
        elif not rx.match(p.name):
            rep.error("report-name", p, None, f"name doesn't match the {rel[0]} pattern")
    sessions = sorted(q for q in (reports / "sessions").glob("*.md")
                      if REPORT_RES["sessions"].match(q.name))
    if sessions and not NEXT_PROMPT_RE.search(sessions[-1].read_text(encoding="utf-8",
                                                                        errors="replace")):
        rep.warn("report-prompt", sessions[-1], None,
                 "newest session report has no '## Next session prompt' section")


def check_cross(pm: Path, rep: Report, fm: dict, reg: dict, dec: dict, chores: dict) -> None:
    path = pm / "CURRENT.md"
    wip = fm.get("wip", "")
    reg_wip = [f for f, s in reg.items() if s == "WIP"]
    if not is_none(wip) and reg.get(wip) != "WIP":
        rep.warn("wip-mismatch", path, 1, f"wip is {wip} but the register doesn't show it as WIP")
    elif is_none(wip) and reg_wip:
        rep.warn("wip-mismatch", path, 1, f"wip is empty but the register has {reg_wip[0]} as WIP")
    for ref in re.split(r"[,\s]+", fm.get("blocked_on", "")):
        if is_none(ref):
            continue
        status = {"D": dec, "C": chores, "F": reg}.get(ref[:1], {}).get(ref)
        if status is None:
            rep.warn("blocked-on", path, 1, f"blocked_on names {ref}, which doesn't exist")
        elif status in RESOLVED_STATUSES | {"DONE", "DROPPED"}:
            rep.warn("blocked-on", path, 1, f"blocked_on names {ref}, which is already {status}")


# --- git --------------------------------------------------------------------------------------

def git(repo: Path, *args: str) -> tuple[int, str]:
    try:
        p = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                           timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return 1, ""
    return p.returncode, p.stdout.strip()


def same_path(a: str | Path, b: str | Path) -> bool:
    return os.path.normcase(str(Path(a).resolve())) == os.path.normcase(str(Path(b).resolve()))


def find_pm(repo: Path) -> tuple[Path, bool]:
    """(the repo's project-management/ folder, whether it is the main folder's).

    A linked worktree of a private-mode repo has no project-management/: the nested PM repo
    lives only in the main folder, the parent of git's common dir."""
    pm = repo / "project-management"
    if pm.is_dir():
        return pm, False
    rc, common = git(repo, "rev-parse", "--path-format=absolute", "--git-common-dir")
    if rc == 0 and common:
        main_pm = Path(common).parent / "project-management"
        if main_pm.is_dir() and not same_path(main_pm, pm):
            return main_pm, True
    return pm, False


def check_pm_remote(pm: Path, rep: Report) -> None:
    """The nested PM repo's off-disk copy: a remote, and nothing unpushed (as of the last
    fetch; pm_check never fetches)."""
    _, remotes = git(pm, "remote")
    if not remotes:
        rep.warn("pm-remote", pm, None, "the nested PM repo has no remote, so the PM files have "
                 "no off-disk copy (/pm-upgrade step 6 sets one up)")
        return
    rc, branch = git(pm, "symbolic-ref", "--short", "-q", "HEAD")
    if rc != 0:
        return
    rc, upstream = git(pm, "rev-parse", "--abbrev-ref", "-q", f"{branch}@{{upstream}}")
    if rc != 0 or not upstream:
        remote = remotes.splitlines()[0]
        upstream = f"{remote}/{branch}"
        if git(pm, "rev-parse", "--verify", "-q", f"refs/remotes/{upstream}")[0] != 0:
            rep.warn("pm-remote", pm, None, f"the nested PM repo's {branch} has never been "
                     f"pushed to {remote} (git -C project-management push -u {remote} {branch})")
            return
    _, ahead = git(pm, "rev-list", "--count", f"{upstream}..{branch}")
    if ahead.isdigit() and int(ahead) > 0:
        rep.warn("pm-ahead", pm, None, f"the nested PM repo is {ahead} commit(s) ahead of "
                 f"{upstream}; push it (git -C project-management push)")


def check_git(repo: Path, pm: Path, rep: Report, fm: dict, policy: str, mode: str) -> None:
    rc, top = git(repo, "rev-parse", "--show-toplevel")
    if rc != 0:
        rep.skipped.append("git checks: not a git repository")
        return
    if not same_path(top, repo):
        rep.skipped.append(f"git checks: {repo.name} is not a repository root")
        return
    _, tracked = git(repo, "ls-files", "--", "project-management")
    if mode == "private":
        if not (pm / ".git").exists():
            rep.error("private-mode", pm, None, "pm_mode is private but project-management/ "
                      "is not its own git repo")
        else:
            check_pm_remote(pm, rep)
        if tracked:
            rep.error("private-mode", pm, None, "pm_mode is private but the host repo tracks "
                      "project-management/ files")
        if git(pm.parent, "check-ignore", "-q", "project-management")[0] != 0:
            rep.error("private-mode", pm, None, "project-management/ is not excluded from the "
                      "host repo (add it to .git/info/exclude)")
    elif mode == "committed" and (pm / ".git").exists():
        rep.error("private-mode", pm, None, "pm_mode is committed but project-management/ "
                  "is a nested git repo")

    rc, last = git(repo, "log", "-1", "--format=%cs")
    updated = fm.get("updated", "")
    if rc == 0 and DATE_RE.match(last) and DATE_RE.match(updated):
        if date.fromisoformat(last) > date.fromisoformat(updated):
            rep.warn("current-stale", pm / "CURRENT.md", 1, f"updated {updated}, but the last "
                     f"commit is {last}; /wrap-session refreshes it")

    main = next((b for b in ("main", "master")
                 if git(repo, "rev-parse", "--verify", "-q", f"refs/heads/{b}")[0] == 0), None)
    if main is None:
        rep.skipped.append("branch checks: no main or master branch")
        return
    _, gitdir = git(repo, "rev-parse", "--path-format=absolute", "--git-dir")
    _, common = git(repo, "rev-parse", "--path-format=absolute", "--git-common-dir")
    if same_path(gitdir, common):  # the main folder, not a linked worktree
        rc, branch = git(repo, "symbolic-ref", "--short", "-q", "HEAD")
        if rc != 0:
            rep.warn("git-branch", None, None, "the main folder has a detached HEAD")
        elif branch != main:
            rep.warn("git-branch", None, None, f"the main folder is on '{branch}'; between tasks "
                     f"it belongs on {main} (finish, merge or park the branch)")

    if git(repo, "rev-parse", "--verify", "-q", f"refs/remotes/origin/{main}")[0] == 0:
        _, ahead = git(repo, "rev-list", "--count", f"origin/{main}..{main}")
        if ahead.isdigit() and int(ahead) > 0:
            if policy == "pr":
                rep.error("git-ahead", None, None, f"{main} has {ahead} commit(s) not on origin; "
                          "under the pr policy main only moves by merged PRs")
            else:
                rep.warn("git-ahead", None, None, f"{main} is {ahead} commit(s) ahead of origin; "
                         "push after verify passes")

    def merged(branch: str) -> bool:
        if git(repo, "merge-base", "--is-ancestor", branch, main)[0] != 0:
            return False
        rc, log = git(repo, "reflog", "show", "--format=%H", f"refs/heads/{branch}")
        return rc != 0 or len(log.splitlines()) > 1  # one entry = created, never committed to

    _, porcelain = git(repo, "worktree", "list", "--porcelain")
    worktrees = [dict(ln.split(" ", 1) if " " in ln else (ln, "") for ln in block.splitlines())
                 for block in porcelain.split("\n\n") if block.strip()]
    in_worktree = set()
    for wt in worktrees[1:]:
        branch = wt.get("branch", "").removeprefix("refs/heads/")
        in_worktree.add(branch)
        if "prunable" in wt:
            rep.warn("git-worktree", None, None, f"worktree record for {wt['worktree']} points at "
                     "a missing folder (git worktree prune)")
        elif branch and merged(branch):
            rep.warn("git-worktree", None, None, f"worktree {wt['worktree']} is on '{branch}', "
                     f"which is merged into {main}; remove the worktree and the branch")
    _, heads = git(repo, "for-each-ref", "--format=%(refname:short)", "refs/heads")
    for branch in heads.splitlines():
        if branch != main and branch not in in_worktree and merged(branch):
            rep.warn("git-merged-branch", None, None,
                     f"branch '{branch}' is merged into {main}; delete it")
        if mode == "private" and branch != main and git(
                repo, "ls-tree", "-d", "--name-only", branch, "project-management")[1]:
            rep.warn("pm-old-branch", None, None,
                     f"branch '{branch}' still tracks project-management/: checking it out in "
                     "the main folder overwrites the private PM files. Open it in a worktree "
                     f"and merge {main} into it there (.claude/rules/git.md)")
    wt_dir = repo / ".claude" / "worktrees"
    if wt_dir.is_dir():
        known = [wt.get("worktree", "") for wt in worktrees]
        for d in sorted(wt_dir.iterdir()):
            if d.is_dir() and not any(same_path(d, k) for k in known if k):
                rep.warn("git-worktree", d, None, "folder under .claude/worktrees/ is not a "
                         "registered worktree (an orphaned copy?)")


# --- main -------------------------------------------------------------------------------------

def run(repo: Path) -> Report:
    tier, ignores = read_tier(repo)
    pm, from_main = find_pm(repo)
    rep = Report(repo, ignores, pm)
    if from_main:
        rep.notes.append(f"PM files: {pm.as_posix()} (the main folder's; this is a worktree)")
    if not pm.is_dir() and tier.get("pm_mode") == "private" and os.environ.get("CI"):
        rep.skipped.append("PM checks: private PM mode, and a CI checkout has no "
                           "project-management/")
        return rep
    if not pm.is_dir():
        rep.error("missing-file", pm, None, "no project-management/ folder" + (
            "; if the PR that untracks it just merged, finish /pm-upgrade step 6 in the main "
            "folder (restore it from the content commit)" if tier.get("pm_mode") == "private"
            else ""))
        return rep
    if not (repo / ".claude" / "TIER.md").is_file():
        rep.error("missing-file", repo / ".claude" / "TIER.md", None, "no .claude/TIER.md")
    else:
        for key, label in (("git_policy", "Git policy"), ("pm_mode", "PM mode")):
            if key not in tier:
                rep.error("tier", repo / ".claude" / "TIER.md", None, f"no '{label}:' line")
    for marker in V1_MARKERS:
        parent = (pm / marker).parent
        if parent.is_dir() and Path(marker).name in os.listdir(parent):
            rep.error("layout", pm / marker, None, "pre-v3 PM layout; run /pm-upgrade")
    for old, new in RENAMED.items():
        if (pm / old).exists():
            rep.error("layout", pm / old, None,
                      f"{old} became {new} in v3.2; /pm-upgrade moves it")
    missing = [n for n in PM_FILES if not (pm / n).is_file()]
    for n in missing:
        rep.error("missing-file", pm / n, None, f"no {n}")
    if not (pm / "reports" / "sessions").is_dir():
        rep.error("missing-file", pm / "reports" / "sessions", None, "no reports/sessions/ folder")
    fm = check_current(pm, rep, tier) if "CURRENT.md" not in missing else {}
    reg = check_register(pm, rep) if "FEATURES.md" not in missing else {}
    if "PLAN.md" not in missing:
        check_plan(pm, rep)
    dec = check_decisions(pm, rep) if "DECISIONS.md" not in missing else {}
    chores = check_chores(pm, rep) if "CHORES.md" not in missing else {}
    check_reports(pm, rep)
    if fm:
        check_cross(pm, rep, fm, reg, dec, chores)
    policy = tier.get("git_policy") or fm.get("git_policy", "pr")
    mode = tier.get("pm_mode") or fm.get("pm_mode", "committed")
    check_git(repo, pm, rep, fm, policy, mode)
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Check a repo's project-management/ against v3.")
    ap.add_argument("repo", help="repo root (the folder holding project-management/)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)
    repo = Path(args.repo).resolve()
    if not repo.is_dir():
        print(f"pm_check: {args.repo} is not a folder", file=sys.stderr)
        return 2
    rep = run(repo)
    if args.json:
        print(json.dumps({"repo": str(repo), "pm": str(rep.pm), "ok": not rep.errors,
                          "errors": rep.errors, "warnings": rep.warnings,
                          "ignored": rep.ignored, "skipped": rep.skipped,
                          "notes": rep.notes}, indent=2))
    else:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        for label, items in (("ERROR", rep.errors), ("WARN ", rep.warnings)):
            for f in items:
                where = (f["path"] or "(git)") + (f":{f['line']}" if f["line"] else "")
                print(f"{label} {f['code']:<18} {where}  {f['message']}")
        for s in rep.notes:
            print(f"note  {s}")
        for s in rep.skipped:
            print(f"skip  {s}")
        print(f"pm_check: {len(rep.errors)} error(s), {len(rep.warnings)} warning(s)"
              + (f", {rep.ignored} ignored" if rep.ignored else ""))
    return 1 if rep.errors else 0


if __name__ == "__main__":
    sys.exit(main())
