#!/usr/bin/env python3
"""Read-only repo hygiene auditor.

checklist.yaml is the generic standard; this file only executes it, so a
check's pass/fail rule lives here but its existence, tier and thresholds
never do - they are always read from settings/profiles at run time. See
SKILL.md for the rules of engagement (never writes, fixing is a separate
approved step).

checklist.yaml carries no machine-specific data (which repo gets which
profile, local filesystem paths). That optionally lives in a private
overlay.yaml next to the checklist, deep-merged in at run time - see
load_overlay/merge_overlay and the --overlay/--no-overlay flags. This is what
lets one public checklist.yaml be shared across repos/CI while each machine
keeps its own private assignments.
"""

import argparse
import base64
import fnmatch
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

CHECKLIST_PATH = Path(__file__).resolve().parent.parent / "checklist.yaml"
DEFAULT_OVERLAY_PATH = CHECKLIST_PATH.parent / "overlay.yaml"


class ChecklistError(Exception):
    """checklist.yaml itself is missing, unreadable, or structurally invalid."""


class GhError(Exception):
    """The gh CLI (or git, for the local skills-repo facts) could not answer -
    no auth, unknown repo, network hiccup. Distinct from a 404 on an optional
    endpoint (README, tree), which is a legitimate "absent" signal handled
    inline rather than raised."""


# --------------------------------------------------------------------------
# Facts: the only thing check functions are allowed to see. Keeping this a
# plain dataclass with no methods is what lets --self-test build fixtures by
# hand instead of faking a whole gh CLI.
# --------------------------------------------------------------------------


@dataclass
class RepoFacts:
    repo: str
    profile: str
    description: str | None
    topics: list[str]
    license_declared: bool
    default_branch: str | None
    visibility: str
    archived: bool
    tree_paths: list[str]
    readme_text: str | None
    # Populated only for the skills-repo profile, which audits the local
    # working copy instead of (or in addition to) the GitHub API.
    local_status_porcelain: str | None = None
    local_ahead: int | None = None
    local_behind: int | None = None
    local_dir_names: list[str] | None = None
    local_index_text: str | None = None
    # Set instead of the fields above when the local working copy this check
    # needs isn't on this host at all (a sweep box that only has GitHub
    # access, not the laptop's filesystem) - the reason a local-only check
    # must SKIP, not FAIL: a missing path is not evidence of drift.
    local_skills_unavailable: str | None = None
    # Set when skills_local_path exists but isn't a usable git working copy
    # against an `origin` remote - only level_with_origin cares about this;
    # index_covers_skills only needs a directory listing, not git-ness.
    local_git_unavailable: str | None = None


@dataclass
class CheckResult:
    id: str
    tier: str
    status: str  # "PASS" | "FAIL" | "SKIP"
    detail: str


# --------------------------------------------------------------------------
# Small pure helpers shared by several checks.
# --------------------------------------------------------------------------


def path_matches_glob(path: str, pattern: str) -> bool:
    """fnmatch anchors a bare pattern like "id_rsa*" to the start of the
    whole string, so a nested tracked file (dir/id_rsa) would evade a secret
    glob that is clearly meant to catch it anywhere. Also trying the basename
    covers that "anywhere in the tree" intent without a heavier glob library.
    """
    return fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(Path(path).name, pattern)


def heading_lines(readme_text: str) -> list[str]:
    """Section-presence checks must look only at actual markdown headings -
    prose that happens to mention "quickstart" in passing must not satisfy
    readme_has_quickstart, or the check stops meaning anything."""
    out: list[str] = []
    for line in readme_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            out.append(stripped.lstrip("#").strip())
    return out


_BADGE_TOKEN = r"(\[!\[[^\]]*\]\([^)]*\)\]\([^)]*\)|!\[[^\]]*\]\([^)]*\))"
_BADGE_LINE_RE = re.compile(rf"^{_BADGE_TOKEN}(\s+{_BADGE_TOKEN})*$")


def _is_badge_line(line: str) -> bool:
    return bool(_BADGE_LINE_RE.match(line))


def extract_readme_intro(readme_text: str) -> str | None:
    """The intro is the first real paragraph after the H1 - badge rows and a
    repeated title are the common things sitting right after it, and reading
    either of those as "the pitch" would make readme_has_intro meaningless.
    """
    lines = readme_text.splitlines()
    h1_idx = next((i for i, l in enumerate(lines) if l.strip().startswith("# ")), None)
    if h1_idx is None:
        return None
    paragraph: list[str] = []
    for line in lines[h1_idx + 1 :]:
        stripped = line.strip()
        if not stripped:
            if paragraph:
                break
            continue
        if stripped.startswith("#") or _is_badge_line(stripped):
            continue
        paragraph.append(stripped)
    return " ".join(paragraph) if paragraph else None


_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def extract_relative_link_targets(readme_text: str) -> list[str]:
    """Pull every markdown link target that readme_links_resolve is actually
    responsible for: local paths. External links, mailto and pure same-page
    anchors are out of scope by definition (nothing in the tree could ever
    make those "resolve"), so excluding them here is what keeps the check
    from producing false positives on the very things it must ignore."""
    targets: list[str] = []
    for raw in _LINK_RE.findall(readme_text):
        target = raw.strip()
        if " " in target:  # optional markdown link title: [t](url "title")
            target = target.split(" ", 1)[0]
        if not target or target.startswith("#"):
            continue
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        target = target.split("#", 1)[0].split("?", 1)[0]
        if target.startswith("./"):
            target = target[2:]
        if target:
            targets.append(target)
    return targets


def target_exists_in_tree(target: str, tree_paths: list[str]) -> bool:
    """A directory link is satisfied by any tracked path having it as a
    prefix - the tree listing has no standalone "directory" entries to check
    against directly for every case, so prefix matching is the only way a
    link to a folder can ever resolve."""
    if target in tree_paths:
        return True
    prefix = target.rstrip("/") + "/"
    return any(p.startswith(prefix) for p in tree_paths)


def root_md_paths(tree_paths: list[str]) -> list[str]:
    return sorted(p for p in tree_paths if "/" not in p and p.lower().endswith(".md"))


# --------------------------------------------------------------------------
# Checks. Each takes (facts, settings) and returns (status, detail). Every
# threshold, name list and pattern comes from settings - nothing here is a
# hardcoded number, so checklist.yaml stays the single source of truth.
# --------------------------------------------------------------------------


def check_has_description(facts: RepoFacts, settings: dict[str, Any]) -> tuple[str, str]:
    desc = (facts.description or "").strip()
    return ("PASS", desc) if desc else ("FAIL", "no description set")


def check_has_topics(facts: RepoFacts, settings: dict[str, Any]) -> tuple[str, str]:
    n = len(facts.topics)
    status = "PASS" if n >= settings["min_topics"] else "FAIL"
    return status, f"{n} topic(s): {', '.join(facts.topics) or '(none)'}"


def check_license_declared(facts: RepoFacts, settings: dict[str, Any]) -> tuple[str, str]:
    return ("PASS", "declared") if facts.license_declared else ("FAIL", "no license in repo metadata")


def check_license_file(facts: RepoFacts, settings: dict[str, Any]) -> tuple[str, str]:
    hits = [p for p in facts.tree_paths if "/" not in p and fnmatch.fnmatch(p, "LICENSE*")]
    return ("PASS", hits[0]) if hits else ("FAIL", "no LICENSE* file at root")


def check_default_branch_main(facts: RepoFacts, settings: dict[str, Any]) -> tuple[str, str]:
    want = settings["default_branch"]
    got = facts.default_branch or "(none)"
    return ("PASS", got) if got == want else ("FAIL", f"default branch is {got!r}, expected {want!r}")


def check_no_tracked_secrets(facts: RepoFacts, settings: dict[str, Any]) -> tuple[str, str]:
    globs = settings["secret_globs"]
    exceptions = settings["secret_glob_exceptions"]
    hits = [
        p
        for p in facts.tree_paths
        if any(path_matches_glob(p, g) for g in globs)
        and not any(path_matches_glob(p, e) for e in exceptions)
    ]
    return ("FAIL", "; ".join(hits)) if hits else ("PASS", "clean")


def check_gitignore_present(facts: RepoFacts, settings: dict[str, Any]) -> tuple[str, str]:
    return ("PASS", ".gitignore") if ".gitignore" in facts.tree_paths else ("FAIL", "no .gitignore at root")


def check_readme_present(facts: RepoFacts, settings: dict[str, Any]) -> tuple[str, str]:
    return ("PASS", "present") if facts.readme_text is not None else ("FAIL", "no README")


def check_readme_has_intro(facts: RepoFacts, settings: dict[str, Any]) -> tuple[str, str]:
    if facts.readme_text is None:
        return "FAIL", "no README to check"
    intro = extract_readme_intro(facts.readme_text)
    if intro is None:
        return "FAIL", "no paragraph found after the H1"
    n = len(intro.split())
    status = "PASS" if n >= settings["readme_intro_min_words"] else "FAIL"
    return status, f"{n} words"


def _section_status(facts: RepoFacts, settings: dict[str, Any], pattern_key: str) -> tuple[str, str]:
    if facts.readme_text is None:
        return "FAIL", "no README to check"
    pattern = settings["section_patterns"][pattern_key]
    for heading in heading_lines(facts.readme_text):
        if re.search(pattern, heading, re.IGNORECASE):
            return "PASS", heading
    return "FAIL", f"no heading matches /{pattern}/i"


def check_readme_has_quickstart(facts: RepoFacts, settings: dict[str, Any]) -> tuple[str, str]:
    return _section_status(facts, settings, "quickstart")


def check_readme_has_stack(facts: RepoFacts, settings: dict[str, Any]) -> tuple[str, str]:
    return _section_status(facts, settings, "stack")


def check_readme_has_running_for_real(facts: RepoFacts, settings: dict[str, Any]) -> tuple[str, str]:
    return _section_status(facts, settings, "running_for_real")


def check_readme_has_how_built(facts: RepoFacts, settings: dict[str, Any]) -> tuple[str, str]:
    return _section_status(facts, settings, "how_built")


def check_readme_links_resolve(facts: RepoFacts, settings: dict[str, Any]) -> tuple[str, str]:
    if facts.readme_text is None:
        return "PASS", "no README, nothing to check (see readme_present)"
    broken = [
        t
        for t in extract_relative_link_targets(facts.readme_text)
        if not target_exists_in_tree(t, facts.tree_paths)
    ]
    return ("FAIL", "; ".join(broken)) if broken else ("PASS", "all relative links resolve")


_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")


def check_readme_has_screenshots(facts: RepoFacts, settings: dict[str, Any]) -> tuple[str, str]:
    markers = settings["screenshot_markers"]
    path_hit = next((p for p in facts.tree_paths if any(p.startswith(m) for m in markers)), None)
    if path_hit:
        return "PASS", f"tree path {path_hit}"
    if facts.readme_text and _IMAGE_RE.search(facts.readme_text):
        return "PASS", "README image embed"
    return "FAIL", "no screenshot marker path and no image embed in README"


def check_quickstart_artifact(facts: RepoFacts, settings: dict[str, Any]) -> tuple[str, str]:
    artifacts = settings["quickstart_artifacts"]
    hit = next((a for a in artifacts if a in facts.tree_paths), None)
    return ("PASS", hit) if hit else ("FAIL", f"none of {artifacts} present")


def _tree_indicates_env_usage(tree_paths: list[str]) -> bool:
    """Whether env_example_present even applies. Deliberately cheap: this
    infers env-var usage from compose/Dockerfile/.env.example presence in the
    already-fetched tree, rather than downloading source files to grep for
    os.getenv - that would cost one extra call per file across 11 repos for a
    check that is SHOULD/conditional to begin with."""
    compose_names = {"docker-compose.yml", "compose.yaml"}
    for p in tree_paths:
        name = Path(p).name
        if name in compose_names or name == "Dockerfile" or name.startswith("Dockerfile."):
            return True
        if ".env.example" in name or ".env.sample" in name:
            return True
    return False


def check_env_example_present(facts: RepoFacts, settings: dict[str, Any]) -> tuple[str, str]:
    if not _tree_indicates_env_usage(facts.tree_paths):
        return "SKIP", "no compose/Dockerfile/.env.example marker; env usage not indicated"
    names = settings["env_example_names"]
    hit = next((n for n in names if n in facts.tree_paths), None)
    return ("PASS", hit) if hit else ("FAIL", f"env usage indicated but none of {names} present")


def check_demo_path(facts: RepoFacts, settings: dict[str, Any]) -> tuple[str, str]:
    markers = [m.lower() for m in settings["demo_markers"]]
    for p in facts.tree_paths:
        low = p.lower()
        hit = next((m for m in markers if m in low), None)
        if hit:
            return "PASS", p
    return "FAIL", f"no tree path matches any of {settings['demo_markers']}"


def check_ci_workflow_present(facts: RepoFacts, settings: dict[str, Any]) -> tuple[str, str]:
    globs = settings["ci_globs"]
    hit = next((p for p in facts.tree_paths if any(path_matches_glob(p, g) for g in globs)), None)
    return ("PASS", hit) if hit else ("FAIL", f"no path matches {globs}")


def check_root_md_budget(facts: RepoFacts, settings: dict[str, Any]) -> tuple[str, str]:
    mds = root_md_paths(facts.tree_paths)
    budget = settings["root_md_budget"]
    status = "PASS" if len(mds) <= budget else "FAIL"
    return status, f"{len(mds)}/{budget}: {', '.join(mds)}"


def check_root_md_linked(facts: RepoFacts, settings: dict[str, Any]) -> tuple[str, str]:
    allowed = set(settings["root_md_always_allowed"])
    readme = facts.readme_text or ""
    orphans = [p for p in root_md_paths(facts.tree_paths) if p not in allowed and p not in readme]
    return ("FAIL", "; ".join(orphans)) if orphans else ("PASS", "all root .md referenced or allowed")


def check_status_doc(facts: RepoFacts, settings: dict[str, Any]) -> tuple[str, str]:
    names = settings["status_doc_names"]
    hit = next((n for n in names if n in facts.tree_paths), None)
    return ("PASS", hit) if hit else ("FAIL", f"none of {names} present")


def check_index_covers_skills(facts: RepoFacts, settings: dict[str, Any]) -> tuple[str, str]:
    if facts.local_skills_unavailable is not None:
        # Cannot be evaluated on this host, not "evaluated and clean" - a
        # missing working copy must never read as a silent PASS either.
        return "SKIP", facts.local_skills_unavailable
    if facts.local_dir_names is None or facts.local_index_text is None:
        # Facts were never even attempted (this check reached without the
        # skills-repo gathering step running at all) - a real bug, not an
        # absent-host situation, so it stays a loud FAIL.
        return "FAIL", "local skills directory facts were not gathered"
    exempt = settings.get("skills_index_exempt", [])
    uncovered = [
        d
        for d in facts.local_dir_names
        if d not in facts.local_index_text and not any(fnmatch.fnmatch(d, pat) for pat in exempt)
    ]
    return ("FAIL", "; ".join(uncovered)) if uncovered else ("PASS", "all skill dirs covered")


def check_no_tracked_agent_instructions(facts: RepoFacts, settings: dict[str, Any]) -> tuple[str, str]:
    # .get(..., []) rather than a direct index: an older checklist.yaml (or a
    # profile assigned this check before the settings keys existed) must
    # degrade to "nothing to flag", not crash the whole run.
    globs = settings.get("agent_instruction_globs", [])
    exempt = settings.get("agent_instruction_exempt", [])
    hits = [
        p
        for p in facts.tree_paths
        if any(path_matches_glob(p, g) for g in globs) and not any(path_matches_glob(p, e) for e in exempt)
    ]
    return ("FAIL", "; ".join(hits)) if hits else ("PASS", "clean")


def check_level_with_origin(facts: RepoFacts, settings: dict[str, Any]) -> tuple[str, str]:
    if facts.local_skills_unavailable is not None:
        return "SKIP", facts.local_skills_unavailable
    if facts.local_git_unavailable is not None:
        return "SKIP", facts.local_git_unavailable
    if facts.local_status_porcelain is None or facts.local_ahead is None or facts.local_behind is None:
        return "FAIL", "local git facts were not gathered"
    dirty = bool(facts.local_status_porcelain.strip())
    if dirty or facts.local_ahead or facts.local_behind:
        return "FAIL", f"dirty={dirty} ahead={facts.local_ahead} behind={facts.local_behind}"
    return "PASS", "clean, level with origin"


CHECKS: dict[str, Callable[[RepoFacts, dict[str, Any]], tuple[str, str]]] = {
    "has_description": check_has_description,
    "has_topics": check_has_topics,
    "license_declared": check_license_declared,
    "license_file": check_license_file,
    "default_branch_main": check_default_branch_main,
    "no_tracked_secrets": check_no_tracked_secrets,
    "no_tracked_agent_instructions": check_no_tracked_agent_instructions,
    "gitignore_present": check_gitignore_present,
    "readme_present": check_readme_present,
    "readme_has_intro": check_readme_has_intro,
    "readme_has_quickstart": check_readme_has_quickstart,
    "readme_has_stack": check_readme_has_stack,
    "readme_links_resolve": check_readme_links_resolve,
    "readme_has_screenshots": check_readme_has_screenshots,
    "readme_has_running_for_real": check_readme_has_running_for_real,
    "readme_has_how_built": check_readme_has_how_built,
    "quickstart_artifact": check_quickstart_artifact,
    "env_example_present": check_env_example_present,
    "demo_path": check_demo_path,
    "ci_workflow_present": check_ci_workflow_present,
    "root_md_budget": check_root_md_budget,
    "root_md_linked": check_root_md_linked,
    "status_doc": check_status_doc,
    "index_covers_skills": check_index_covers_skills,
    "level_with_origin": check_level_with_origin,
}


# --------------------------------------------------------------------------
# --suggest: for each FAIL, the exact remedy - a command or a concrete edit.
# Lives in code, not in checklist.yaml, on purpose: the YAML says what the
# bar is, the code says how to meet it. This still keeps the read-only
# guarantee, since a suggestion is a string to print, never something run.
#
# One template per check id, keyed the same as CHECKS. --self-test asserts
# the two dicts have identical key sets, so a new check cannot ship without
# a remedy - the assertion fails loudly instead of quietly leaving a FAIL
# with nothing actionable next to it.
# --------------------------------------------------------------------------

SuggestFn = Callable[[RepoFacts, CheckResult, dict[str, Any]], str]


def _split_detail(detail: str) -> list[str]:
    """Checks that list multiple offending paths join them with '; ' (see
    no_tracked_secrets, root_md_linked, etc) - this undoes that join so a
    suggestion can address each one by name instead of echoing the blob."""
    return [p.strip() for p in detail.split(";") if p.strip()]


def _section_suggestion(pattern_key: str, human_heading: str) -> SuggestFn:
    def fn(facts: RepoFacts, result: CheckResult, settings: dict[str, Any]) -> str:
        pattern = settings["section_patterns"][pattern_key]
        return f'Add a "## {human_heading}" heading to the README (matches the required pattern /{pattern}/i).'

    return fn


def suggest_has_description(facts: RepoFacts, result: CheckResult, settings: dict[str, Any]) -> str:
    return f'gh repo edit {facts.repo} --description "one sentence describing what this repo is"'


def suggest_has_topics(facts: RepoFacts, result: CheckResult, settings: dict[str, Any]) -> str:
    n = settings["min_topics"]
    return f"gh repo edit {facts.repo} --add-topic topic1,topic2,topic3  # at least {n} topics required"


def suggest_license_declared(facts: RepoFacts, result: CheckResult, settings: dict[str, Any]) -> str:
    return (
        "Add a LICENSE file at the repo root; GitHub detects it and populates this field on its own. "
        "Pick one at https://choosealicense.com/ if unsure."
    )


def suggest_license_file(facts: RepoFacts, result: CheckResult, settings: dict[str, Any]) -> str:
    return "Add a LICENSE file at the repo root. Pick one at https://choosealicense.com/ if unsure."


def suggest_default_branch_main(facts: RepoFacts, result: CheckResult, settings: dict[str, Any]) -> str:
    want = settings["default_branch"]
    return f"gh api -X PATCH repos/{facts.repo} -f default_branch={want}"


def suggest_no_tracked_secrets(facts: RepoFacts, result: CheckResult, settings: dict[str, Any]) -> str:
    paths = _split_detail(result.detail)
    lines = [f'git rm --cached "{p}"  # then add a matching line to .gitignore' for p in paths]
    lines.append(
        "If any of these were ever pushed, treat the secret as compromised and rotate it: removing it "
        "from HEAD does not remove it from git history."
    )
    return "\n".join(lines)


def suggest_no_tracked_agent_instructions(facts: RepoFacts, result: CheckResult, settings: dict[str, Any]) -> str:
    paths = _split_detail(result.detail)
    lines = [f'git rm --cached "{p}"' for p in paths]
    lines.append("Add the matching filename(s) (e.g. CLAUDE.md, AGENTS.md, .cursorrules) to .gitignore.")
    return "\n".join(lines)


def suggest_gitignore_present(facts: RepoFacts, result: CheckResult, settings: dict[str, Any]) -> str:
    return "Create a .gitignore at the repo root (language-appropriate basics: caches, build output, local env files)."


def suggest_readme_present(facts: RepoFacts, result: CheckResult, settings: dict[str, Any]) -> str:
    return 'Add a README.md at the repo root: start with "# <name>", then a plain paragraph on what it does.'


def suggest_readme_has_intro(facts: RepoFacts, result: CheckResult, settings: dict[str, Any]) -> str:
    n = settings["readme_intro_min_words"]
    return (
        f"Right after the H1, add a plain paragraph (at least {n} words) saying what this is and what "
        "problem it solves for someone landing on it cold. Badges or a repeated title before it don't count."
    )


suggest_readme_has_quickstart: SuggestFn = _section_suggestion("quickstart", "Quick start")
suggest_readme_has_stack: SuggestFn = _section_suggestion("stack", "How it works")
suggest_readme_has_running_for_real: SuggestFn = _section_suggestion("running_for_real", "Running it for real")
suggest_readme_has_how_built: SuggestFn = _section_suggestion("how_built", "How this was built")


def suggest_readme_links_resolve(facts: RepoFacts, result: CheckResult, settings: dict[str, Any]) -> str:
    targets = _split_detail(result.detail)
    if not targets:
        return "Fix the broken relative link(s), or restore the missing file(s) they point to."
    return "\n".join(f'Broken link target "{t}": fix the link, or restore the file at that path.' for t in targets)


def suggest_readme_has_screenshots(facts: RepoFacts, result: CheckResult, settings: dict[str, Any]) -> str:
    marker = settings["screenshot_markers"][0]
    return f"Add a screenshot under {marker}/ (or embed one directly in the README with an ![...](...) image)."


def suggest_quickstart_artifact(facts: RepoFacts, result: CheckResult, settings: dict[str, Any]) -> str:
    artifacts = settings["quickstart_artifacts"]
    return f"Add one of {artifacts} at the repo root so a stranger has one command to run this."


def suggest_env_example_present(facts: RepoFacts, result: CheckResult, settings: dict[str, Any]) -> str:
    name = settings["env_example_names"][0]
    return f"Add {name} listing every environment variable name the app reads, with placeholder or no values."


def suggest_demo_path(facts: RepoFacts, result: CheckResult, settings: dict[str, Any]) -> str:
    markers = settings["demo_markers"]
    return f"Add a demo path or seed data so it's obvious how to see this with real-looking data, e.g. {markers[0]}/."


def suggest_ci_workflow_present(facts: RepoFacts, result: CheckResult, settings: dict[str, Any]) -> str:
    return "Copy WORKFLOW_TEMPLATE.yml from willianpripp/repo-hygiene to .github/workflows/hygiene.yml in this repo."


def suggest_root_md_budget(facts: RepoFacts, result: CheckResult, settings: dict[str, Any]) -> str:
    return f"Move some of these under docs/ or docs/archive/ to get back under budget: {result.detail}"


def suggest_root_md_linked(facts: RepoFacts, result: CheckResult, settings: dict[str, Any]) -> str:
    orphans = _split_detail(result.detail)
    if not orphans:
        return "Link each orphaned root .md from the README, or delete it if it's stale."
    return "\n".join(f'Link "{p}" from the README, or delete it if it is stale.' for p in orphans)


def suggest_status_doc(facts: RepoFacts, result: CheckResult, settings: dict[str, Any]) -> str:
    name = settings["status_doc_names"][0]
    return f"Add {name}: current state, what's done, what's next, so a cold session knows where to resume."


def suggest_index_covers_skills(facts: RepoFacts, result: CheckResult, settings: dict[str, Any]) -> str:
    dirs = _split_detail(result.detail)
    if not dirs:
        return "Add a row to the skills index for each uncovered skill directory."
    return "\n".join(f"Add a row to the skills index covering `{d}`." for d in dirs)


def suggest_level_with_origin(facts: RepoFacts, result: CheckResult, settings: dict[str, Any]) -> str:
    return "Commit and push (git add, git commit, git push), or if intentionally diverged, note why in STATUS.md."


# A local-only check's SKIP means "cannot be evaluated on this host," not
# "nothing to check" (that's what env_example_present's SKIP means, and it
# gets no suggestion at all - there's genuinely nothing to say). These two
# get a suggestion naming what would have to be true to run them: never a
# fix for a failure, since none was established.
def suggest_skip_index_covers_skills(facts: RepoFacts, result: CheckResult, settings: dict[str, Any]) -> str:
    return (
        f"{result.detail}. Run this on the machine that holds the skills working copy, or point "
        "`skills_local_path` (and `skills_index`) in the overlay at one that exists on this host."
    )


def suggest_skip_level_with_origin(facts: RepoFacts, result: CheckResult, settings: dict[str, Any]) -> str:
    return (
        f"{result.detail}. Run this on the machine that holds the working copy, or point "
        "`skills_local_path` in the overlay at a usable git working copy on this host."
    )


LOCAL_SKIP_SUGGESTIONS: dict[str, SuggestFn] = {
    "index_covers_skills": suggest_skip_index_covers_skills,
    "level_with_origin": suggest_skip_level_with_origin,
}


SUGGESTIONS: dict[str, SuggestFn] = {
    "has_description": suggest_has_description,
    "has_topics": suggest_has_topics,
    "license_declared": suggest_license_declared,
    "license_file": suggest_license_file,
    "default_branch_main": suggest_default_branch_main,
    "no_tracked_secrets": suggest_no_tracked_secrets,
    "no_tracked_agent_instructions": suggest_no_tracked_agent_instructions,
    "gitignore_present": suggest_gitignore_present,
    "readme_present": suggest_readme_present,
    "readme_has_intro": suggest_readme_has_intro,
    "readme_has_quickstart": suggest_readme_has_quickstart,
    "readme_has_stack": suggest_readme_has_stack,
    "readme_links_resolve": suggest_readme_links_resolve,
    "readme_has_screenshots": suggest_readme_has_screenshots,
    "readme_has_running_for_real": suggest_readme_has_running_for_real,
    "readme_has_how_built": suggest_readme_has_how_built,
    "quickstart_artifact": suggest_quickstart_artifact,
    "env_example_present": suggest_env_example_present,
    "demo_path": suggest_demo_path,
    "ci_workflow_present": suggest_ci_workflow_present,
    "root_md_budget": suggest_root_md_budget,
    "root_md_linked": suggest_root_md_linked,
    "status_doc": suggest_status_doc,
    "index_covers_skills": suggest_index_covers_skills,
    "level_with_origin": suggest_level_with_origin,
}


def suggestion_for(facts: RepoFacts, result: CheckResult, settings: dict[str, Any]) -> str | None:
    """None for most things that aren't a FAIL (PASS, ordinary SKIP): a
    suggestion only makes sense next to a real defect. The local-only checks
    are the one carve-out - see LOCAL_SKIP_SUGGESTIONS. Missing a template
    for a known FAIL is a checklist/audit.py mismatch the self-test assertion
    is meant to catch before this ever runs against a real repo."""
    if result.status == "SKIP":
        fn = LOCAL_SKIP_SUGGESTIONS.get(result.id)
        return fn(facts, result, settings) if fn else None
    if result.status != "FAIL":
        return None
    fn = SUGGESTIONS.get(result.id)
    if fn is None:
        return "no suggestion template registered for this check id (see --self-test)"
    return fn(facts, result, settings)


# --------------------------------------------------------------------------
# Enumeration reconciliation: is every repo the owner actually has on GitHub
# accounted for by the overlay, one way or another?
#
# A sweep with no --repo/--here used to just iterate `assignments`, which
# makes it structurally blind to a repo that was created and never added -
# the standard silently does not apply, and nothing ever says so. Enumerating
# the owner's real repos and reconciling against assignments/excluded closes
# that hole: a forgotten repo becomes a loud finding instead of a silent gap.
# --------------------------------------------------------------------------


def owners_from_assignments(assignments: dict[str, str]) -> list[str]:
    """Assignment keys are always owner/name, so the owner(s) to enumerate
    are derived from the overlay itself rather than hardcoded - a second
    owner in a future overlay is picked up for free."""
    return sorted({repo.split("/", 1)[0] for repo in assignments if "/" in repo})


def reconcile_enumeration(
    enumerated: list[dict[str, Any]], assignments: dict[str, str], excluded: dict[str, str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """`enumerated` is what the API actually returns right now for the
    owner(s) - real repos, current names, post-rename (a repo that was
    renamed shows up only under its current name, never as a phantom old one,
    precisely because this reconciles against live API data instead of any
    name seen elsewhere, such as a stale local clone's remote).

    Returns (unassigned, informational).

    `excluded` is what closes the loop for any repo, archived or not: being
    in `excluded` (whatever the reason) always means "accounted for."

    `assignments` closes the loop for a normal, non-archived repo. It does
    NOT close the loop for an archived one: a profile implies ongoing work or
    a live publication, neither of which an archived repo still has, so an
    archived repo needs a deliberate retirement note in `excluded` instead of
    just sitting in `assignments` from before it was archived. That is the
    exact situation a repo is in when it gets archived and nobody updates
    the overlay it was already listed in.

    So: excluded -> accounted for (archived ones also get an informational
    note, just for visibility, since a reader may want to know a repo was
    consciously retired rather than simply absent from the report).
    Not excluded + archived -> unassigned, regardless of assignments.
    Not excluded + not archived + in assignments -> accounted for.
    Not excluded + not archived + not in assignments -> unassigned.
    """
    unassigned: list[dict[str, Any]] = []
    informational: list[dict[str, Any]] = []
    for entry in enumerated:
        repo = entry["repo"]
        archived = bool(entry.get("archived", False))
        if repo in excluded:
            if archived:
                informational.append({"repo": repo, "archived": True, "reason": excluded[repo]})
            continue
        if archived:
            unassigned.append({"repo": repo, "archived": True, "reason": "archived but not in excluded"})
        elif repo not in assignments:
            unassigned.append({"repo": repo, "archived": False, "reason": "not in assignments or excluded"})
    return unassigned, informational


def suggestion_for_unassigned(finding: dict[str, Any], settings: dict[str, Any]) -> str:
    repo = finding["repo"]
    if finding.get("archived"):
        return (
            f"Add {repo} to the overlay's `excluded` map with a reason. Archived repos are usually best "
            "excluded rather than left in `assignments`: a profile implies ongoing work or a live "
            "publication, and an archived repo no longer has either."
        )
    return f"Add {repo} to the overlay's `assignments` with a profile, or to `excluded` with a reason."


def run_checks(facts: RepoFacts, profile_checks: dict[str, str], settings: dict[str, Any]) -> list[CheckResult]:
    results: list[CheckResult] = []
    for check_id, raw_tier in profile_checks.items():
        # YAML 1.1 folds a bare, unquoted OFF/off/Off into the boolean False
        # (same family as on/yes/no), so a checklist author writing the
        # natural `has_topics: OFF` gets a bool back from yaml.safe_load, not
        # the string "OFF". Coercing it back here is what keeps the tier ==
        # "OFF" skip below actually firing, instead of silently running (and
        # then silently dropping from the report, since a bool tier also
        # fails to match "MUST"/"SHOULD" downstream).
        tier = "OFF" if raw_tier is False else raw_tier
        if tier == "OFF":
            continue
        fn = CHECKS.get(check_id)
        if fn is None:
            # A checklist typo or a genuinely new check id the script hasn't
            # caught up with yet - surfacing it as a loud failure is safer
            # than silently dropping a check nobody asked to disable.
            results.append(CheckResult(check_id, tier, "FAIL", "check id not implemented in audit.py"))
            continue
        status, detail = fn(facts, settings)
        results.append(CheckResult(check_id, tier, status, detail))
    return results


# --------------------------------------------------------------------------
# Gathering layer: the only place that touches the network or local git.
# Everything above this line is pure and covered by --self-test; everything
# below is I/O and deliberately kept out of the check functions' reach.
# --------------------------------------------------------------------------


def load_checklist(path: Path) -> dict[str, Any]:
    try:
        import yaml  # deferred: only needed for the real run, not --self-test's import path

        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise ChecklistError(f"cannot read checklist at {path}: {e}") from e
    try:
        data = yaml.safe_load(text)
    except Exception as e:  # yaml.YAMLError plus anything a malformed file can raise
        raise ChecklistError(f"checklist.yaml is not valid YAML: {e}") from e
    if not isinstance(data, dict) or "profiles" not in data or "settings" not in data:
        raise ChecklistError("checklist.yaml is missing required top-level keys (profiles/settings)")
    return data


def load_overlay(path: Path) -> dict[str, Any]:
    """A private overlay is optional and much smaller than the checklist: no
    profiles, just assignments/excluded/settings. Malformed still raises -
    an overlay that fails to parse should not silently vanish and produce a
    sweep of zero repos when the user expected eleven."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise ChecklistError(f"cannot read overlay at {path}: {e}") from e
    try:
        import yaml  # deferred, same reason as load_checklist

        data = yaml.safe_load(text)
    except Exception as e:
        raise ChecklistError(f"overlay at {path} is not valid YAML: {e}") from e
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ChecklistError(f"overlay at {path} must be a YAML mapping")
    return data


def merge_overlay(checklist: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """assignments/excluded are unioned (overlay wins on key collisions);
    overlay settings keys override or extend the generic ones. profiles are
    never touched by an overlay - the bar itself is not machine-specific."""
    merged = dict(checklist)
    merged["assignments"] = {**checklist.get("assignments", {}), **overlay.get("assignments", {})}
    merged["excluded"] = {**checklist.get("excluded", {}), **overlay.get("excluded", {})}
    merged["settings"] = {**checklist.get("settings", {}), **overlay.get("settings", {})}
    return merged


def run_gh(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Every call into gh funnels through here, which is what makes 'grep for
    a write path and find nothing' a meaningful check on this file: there is
    exactly one place gh gets invoked, and it is always a GET-shaped read."""
    return subprocess.run(["gh", *args], capture_output=True, text=True, timeout=30)


def check_gh_auth() -> None:
    result = run_gh(["auth", "status"])
    if result.returncode != 0:
        raise GhError(f"gh is not authenticated: {(result.stderr or result.stdout).strip()}")


def gh_api(path: str, jq: str | None = None) -> Any:
    cmd = ["api", path]
    if jq is not None:
        cmd += ["--jq", jq]
    result = run_gh(cmd)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        if "404" in stderr or "Not Found" in stderr:
            return None  # absent, not broken - the repo/README/tree just doesn't have this
        raise GhError(f"gh api {path} failed: {stderr or result.stdout.strip()}")
    out = result.stdout.strip()
    if not out:
        return None
    if jq is not None:
        return out  # --jq on a scalar field returns raw text, not a JSON literal
    try:
        return json.loads(out)
    except json.JSONDecodeError as e:
        raise GhError(f"gh api {path} returned unparsable output: {e}") from e


def list_owner_repos(owner: str) -> list[dict[str, Any]]:
    """All of an owner's repos, public and private, via one `gh repo list`
    invocation - gh paginates the underlying API internally, so this stays a
    single call from our side. Deliberately not `users/{owner}/repos`: that
    REST endpoint is public-only regardless of auth, which would make every
    private repo structurally invisible to the exact check meant to catch a
    forgotten one."""
    result = run_gh(["repo", "list", owner, "--limit", "1000", "--json", "nameWithOwner,isArchived"])
    if result.returncode != 0:
        raise GhError(f"gh repo list {owner} failed: {(result.stderr or result.stdout).strip()}")
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise GhError(f"gh repo list {owner} returned unparsable output: {e}") from e
    return [{"repo": r["nameWithOwner"], "archived": bool(r.get("isArchived"))} for r in data]


def gather_repo_facts(repo: str, profile: str, settings: dict[str, Any]) -> RepoFacts:
    meta = gh_api(f"repos/{repo}")
    if meta is None:
        raise GhError(f"repo {repo} not found or inaccessible")
    default_branch = meta.get("default_branch")

    tree_paths: list[str] = []
    if default_branch:  # a truly empty repo has no default branch to list a tree for
        tree = gh_api(f"repos/{repo}/git/trees/{default_branch}?recursive=1")
        if isinstance(tree, dict):
            tree_paths = [entry["path"] for entry in tree.get("tree", []) if "path" in entry]

    readme_text: str | None = None
    readme_b64 = gh_api(f"repos/{repo}/readme", jq=".content")
    if readme_b64:
        try:
            readme_text = base64.b64decode(readme_b64.replace("\n", "")).decode("utf-8", errors="replace")
        except (ValueError, UnicodeDecodeError):
            readme_text = None

    facts = RepoFacts(
        repo=repo,
        profile=profile,
        description=meta.get("description"),
        topics=meta.get("topics") or [],
        license_declared=bool(meta.get("license")),
        default_branch=default_branch,
        visibility=meta.get("visibility") or ("private" if meta.get("private") else "public"),
        archived=bool(meta.get("archived")),
        tree_paths=tree_paths,
        readme_text=readme_text,
    )

    if profile == "skills-repo":
        _attach_local_skills_facts(facts, settings)

    return facts


def _run_git(args: list[str], cwd: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=30)


def _attach_local_skills_facts(facts: RepoFacts, settings: dict[str, Any]) -> None:
    """skills-repo's whole point is catching drift between disk, the human
    index and origin, so (unlike every other profile) it needs the local
    working copy, not just the GitHub API. On a host that does not have that
    working copy - a headless sweep box with GitHub access but not the
    laptop's filesystem - there is nothing to gather. Setting
    local_skills_unavailable (instead of leaving the fields below at their
    None default and letting a git subprocess crash on a nonexistent cwd) is
    what lets the two checks that need this SKIP cleanly instead of FAILing
    on a condition nobody here can evaluate."""
    local_path = settings.get("skills_local_path")
    if not local_path or not Path(local_path).is_dir():
        facts.local_skills_unavailable = f"no local working copy at {local_path!r} on this host"
        return

    status = _run_git(["status", "--porcelain"], cwd=local_path)
    facts.local_status_porcelain = status.stdout if status.returncode == 0 else status.stderr

    # Read the current branch instead of hardcoding "main" - a renamed or
    # detached local branch would otherwise silently diff against the wrong
    # upstream ref and report a false level_with_origin failure.
    branch_result = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=local_path)
    branch = branch_result.stdout.strip() if branch_result.returncode == 0 else "HEAD"
    rev_list = _run_git(["rev-list", "--left-right", "--count", f"origin/{branch}...HEAD"], cwd=local_path)
    if rev_list.returncode == 0 and len(rev_list.stdout.split()) == 2:
        behind_str, ahead_str = rev_list.stdout.split()
        facts.local_behind, facts.local_ahead = int(behind_str), int(ahead_str)
    else:
        # Path exists but isn't a usable git working copy against an
        # `origin` remote (no .git, no origin, detached with nothing to
        # diff against, ...) - level_with_origin cannot be evaluated either,
        # for the same "absent, not dirty" reason as the path-missing case.
        detail = (rev_list.stderr or rev_list.stdout).strip() or "git rev-list failed"
        facts.local_git_unavailable = f"{local_path} is not a usable git working copy against origin/{branch}: {detail}"

    facts.local_dir_names = sorted(
        p.name for p in Path(local_path).iterdir() if p.is_dir() and not p.name.startswith(".")
    )

    index_path_str = settings.get("skills_index")
    facts.local_index_text = (
        Path(index_path_str).read_text(encoding="utf-8", errors="replace")
        if index_path_str and Path(index_path_str).exists()
        else ""
    )


def resolve_repo_from_cwd() -> str:
    result = subprocess.run(["git", "remote", "get-url", "origin"], capture_output=True, text=True, timeout=15)
    if result.returncode != 0:
        raise GhError("--here requires a git remote named origin in the current directory")
    url = result.stdout.strip()
    match = re.search(r"github\.com[:/]+([^/]+)/([^/.]+?)(?:\.git)?/?$", url)
    if not match:
        raise GhError(f"could not parse an owner/repo from origin url: {url}")
    return f"{match.group(1)}/{match.group(2)}"


# --------------------------------------------------------------------------
# Output rendering.
# --------------------------------------------------------------------------


@dataclass
class RepoRun:
    """One repo's outcome. `error` set means gather_repo_facts blew up for
    just this repo (transient API failure, repo vanished mid-sweep, etc) -
    facts/checks stay empty and the repo is reported as ERROR rather than
    silently dropped or aborting the other repos in the sweep."""

    repo: str
    profile: str
    facts: RepoFacts | None
    checks: list[CheckResult]
    error: str | None = None


@dataclass
class EnumerationResult:
    """`ran` distinguishes "checked, found zero" from "did not check" - the
    latter (no overlay, --no-enumerate, or a targeted --repo/--here run)
    means the unassigned/informational lists say nothing about coverage and
    must not be rendered as if they did."""

    ran: bool
    unassigned: list[dict[str, Any]]
    informational: list[dict[str, Any]]


def _indented_suggestion(text: str) -> str:
    return "\n".join(f"        > {line}" for line in text.splitlines())


def render_table(
    results: list[RepoRun],
    settings: dict[str, Any] | None = None,
    suggest: bool = False,
    enumeration: EnumerationResult | None = None,
) -> str:
    lines: list[str] = []
    summary_rows: list[tuple[str, str, str]] = []
    for run in results:
        if run.error is not None:
            lines.append(f"{run.repo} [{run.profile}]")
            lines.append(f"  ERROR {run.error}")
            lines.append("")
            summary_rows.append((run.repo, "ERROR", "-"))
            continue
        facts = run.facts
        assert facts is not None  # error is None, so gather_repo_facts succeeded
        vis = facts.visibility + (" archived" if facts.archived else "")
        lines.append(f"{run.repo} [{run.profile}] {vis}")
        musts = [c for c in run.checks if c.tier == "MUST" and c.status == "FAIL"]
        shoulds = [c for c in run.checks if c.tier == "SHOULD" and c.status == "FAIL"]
        skips = [c for c in run.checks if c.status == "SKIP"]
        passes = [c for c in run.checks if c.status == "PASS"]
        for c in musts:
            lines.append(f"  FAIL  {c.id:<28} {c.detail}")
            if suggest:
                assert settings is not None
                lines.append(_indented_suggestion(suggestion_for(facts, c, settings)))
        for c in shoulds:
            lines.append(f"  warn  {c.id:<28} {c.detail}")
            if suggest:
                assert settings is not None
                lines.append(_indented_suggestion(suggestion_for(facts, c, settings)))
        for c in skips:
            lines.append(f"  skip  {c.id:<28} {c.detail}")
            if suggest:
                assert settings is not None
                # Unlike musts/shoulds, most SKIPs (env_example_present's
                # "doesn't apply") get no suggestion at all - only the
                # local-only checks' "cannot be evaluated here" SKIP does.
                # suggestion_for returns None for the rest, so only print
                # when there is actually something to say.
                skip_suggestion = suggestion_for(facts, c, settings)
                if skip_suggestion:
                    lines.append(_indented_suggestion(skip_suggestion))
        if passes:
            lines.append(f"  ok    {len(passes)} passed: {', '.join(c.id for c in passes)}")
        lines.append("")
        summary_rows.append((run.repo, str(len(musts)), str(len(shoulds))))

    if enumeration is not None and enumeration.ran:
        lines.append("UNASSIGNED")
        if enumeration.unassigned:
            for u in enumeration.unassigned:
                lines.append(f"  FAIL  {u['repo']:<28} {u['reason']}")
                if suggest:
                    assert settings is not None
                    lines.append(_indented_suggestion(suggestion_for_unassigned(u, settings)))
        else:
            lines.append("  none: every enumerated repo is in assignments or excluded")
        lines.append("")

        if enumeration.informational:
            lines.append("ARCHIVED, EXCLUDED (informational, not a failure)")
            for info in enumeration.informational:
                lines.append(f"  info  {info['repo']:<28} {info['reason']}")
            lines.append("")

    lines.append("SUMMARY")
    lines.append(f"{'repo':<42} {'MUST fail':>10} {'SHOULD fail':>12}")
    for repo, m, s in summary_rows:
        lines.append(f"{repo:<42} {m:>10} {s:>12}")
    return "\n".join(lines)


def render_json(
    results: list[RepoRun],
    exit_code: int,
    settings: dict[str, Any] | None = None,
    suggest: bool = False,
    enumeration: EnumerationResult | None = None,
) -> str:
    def check_payload(run: RepoRun, c: CheckResult) -> dict[str, Any]:
        entry = {"id": c.id, "tier": c.tier, "status": c.status, "detail": c.detail}
        if suggest:
            assert settings is not None and run.facts is not None
            entry["suggestion"] = suggestion_for(run.facts, c, settings)
        return entry

    def unassigned_payload(u: dict[str, Any]) -> dict[str, Any]:
        entry = dict(u)
        if suggest:
            assert settings is not None
            entry["suggestion"] = suggestion_for_unassigned(u, settings)
        return entry

    payload = {
        "repos": [
            {
                "repo": run.repo,
                "profile": run.profile,
                "error": run.error,
                "checks": [check_payload(run, c) for c in run.checks],
                "must_failed": sum(1 for c in run.checks if c.tier == "MUST" and c.status == "FAIL"),
                "should_failed": sum(1 for c in run.checks if c.tier == "SHOULD" and c.status == "FAIL"),
            }
            for run in results
        ],
        # null (not []) when enumeration did not run at all, so a consumer can
        # tell "confirmed zero" apart from "coverage was never checked."
        "unassigned": [unassigned_payload(u) for u in enumeration.unassigned] if enumeration and enumeration.ran else None,
        "archived_excluded": enumeration.informational if enumeration and enumeration.ran else None,
        "exit_code": exit_code,
    }
    return json.dumps(payload, indent=2)


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only repo hygiene auditor (see SKILL.md).")
    parser.add_argument("--repo", help="owner/name to audit a single repo")
    parser.add_argument("--here", action="store_true", help="resolve owner/name from cwd's git remote")
    parser.add_argument("--json", action="store_true", help="machine-readable output instead of the table")
    parser.add_argument("--profile", help="override the assigned profile (mandatory when no overlay applies)")
    parser.add_argument(
        "--overlay",
        help="path to a private overlay to deep-merge onto the checklist "
        "(default: overlay.yaml next to checklist.yaml, if it exists)",
    )
    parser.add_argument("--no-overlay", action="store_true", help="ignore any overlay file, even the default one")
    parser.add_argument(
        "--suggest",
        action="store_true",
        help="print the exact command/edit that would fix each FAIL - print only, never executed",
    )
    parser.add_argument(
        "--no-enumerate",
        action="store_true",
        help="skip enumerating the owner's repos to find ones missing from the overlay (faster, or offline)",
    )
    parser.add_argument("--self-test", action="store_true", help="run offline fixture assertions and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.self_test:
        return run_self_test()

    try:
        checklist = load_checklist(CHECKLIST_PATH)
    except ChecklistError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    overlay_loaded = False
    if not args.no_overlay:
        explicit_overlay = args.overlay is not None
        overlay_path = Path(args.overlay) if explicit_overlay else DEFAULT_OVERLAY_PATH
        if overlay_path.exists():
            try:
                overlay = load_overlay(overlay_path)
            except ChecklistError as e:
                print(f"error: {e}", file=sys.stderr)
                return 2
            checklist = merge_overlay(checklist, overlay)
            overlay_loaded = True
        elif explicit_overlay:
            # An explicit --overlay is a deliberate ask; a missing default
            # path just means "no overlay today" and is not an error.
            print(f"error: overlay not found at {overlay_path}", file=sys.stderr)
            return 2

    settings = checklist["settings"]
    profiles = checklist["profiles"]
    assignments = checklist.get("assignments") or {}
    excluded = checklist.get("excluded") or {}

    try:
        check_gh_auth()
    except GhError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    is_full_sweep = not args.repo and not args.here
    if args.repo:
        targets = [args.repo]
    elif args.here:
        try:
            targets = [resolve_repo_from_cwd()]
        except GhError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
    else:
        targets = [r for r in assignments if r not in excluded]

    # Enumeration reconciliation only makes sense on a full sweep (targeting
    # one repo already names it explicitly, so there's nothing to reconcile),
    # only when an overlay actually supplied assignments to reconcile against,
    # and only when not explicitly skipped.
    enumeration: EnumerationResult | None = None
    if is_full_sweep and overlay_loaded and not args.no_enumerate:
        try:
            enumerated: list[dict[str, Any]] = []
            for owner in owners_from_assignments(assignments):
                enumerated.extend(list_owner_repos(owner))
        except GhError as e:
            print(f"error: repo enumeration failed: {e}", file=sys.stderr)
            return 2
        unassigned, informational = reconcile_enumeration(enumerated, assignments, excluded)
        enumeration = EnumerationResult(ran=True, unassigned=unassigned, informational=informational)

    results: list[RepoRun] = []
    for repo in targets:
        # Config-shaped problems (bad checklist reference, bad CLI input) are
        # genuinely global: they mean this invocation is malformed, not that
        # one repo had a bad day, so they still abort the whole run.
        profile_name = args.profile or assignments.get(repo)
        if profile_name is None:
            print(
                f"error: {repo} has no assigned profile (no overlay applied, or {repo} is not in its "
                "assignments); remedy either by passing --profile explicitly, or by adding "
                f"{repo} to an overlay's assignments",
                file=sys.stderr,
            )
            return 2
        profile_def = profiles.get(profile_name)
        if profile_def is None:
            print(f"error: unknown profile {profile_name!r}", file=sys.stderr)
            return 2
        if profile_def.get("not_implemented"):
            print(f"error: profile {profile_name!r} is not implemented ({repo})", file=sys.stderr)
            return 2
        try:
            facts = gather_repo_facts(repo, profile_name, settings)
        except GhError as e:
            # Unlike the config problems above, this is specific to this one
            # repo (404, transient API failure, ...). Recording it and moving
            # on is what keeps a bad repo #3 of 11 from erasing the other 10.
            results.append(RepoRun(repo, profile_name, None, [], error=str(e)))
            continue
        checks = run_checks(facts, profile_def["checks"], settings)
        results.append(RepoRun(repo, profile_name, facts, checks))

    any_must_failed = any(c.tier == "MUST" and c.status == "FAIL" for r in results for c in r.checks)
    any_errored = any(r.error is not None for r in results)
    # An unassigned repo is treated like a MUST failure for exit-code purposes
    # on purpose: the whole point of enumerating is that it cannot be ignored.
    any_unassigned = bool(enumeration and enumeration.unassigned)
    exit_code = 2 if any_errored else (1 if (any_must_failed or any_unassigned) else 0)

    if args.json:
        print(render_json(results, exit_code, settings, args.suggest, enumeration))
    else:
        print(render_table(results, settings, args.suggest, enumeration))
    return exit_code


# --------------------------------------------------------------------------
# --self-test: offline, no network. Builds fixtures by hand and asserts every
# check fires in both directions, plus the specific edge cases called out in
# the acceptance bar (nested secret globs, link resolution nuances, section
# patterns vs body prose, the env_example_present SKIP path).
# --------------------------------------------------------------------------


def _base_facts(**overrides: Any) -> RepoFacts:
    defaults: dict[str, Any] = dict(
        repo="acme/widget",
        profile="public-portfolio",
        description="A widget",
        topics=["a", "b", "c"],
        license_declared=True,
        default_branch="main",
        visibility="public",
        archived=False,
        tree_paths=["README.md", "LICENSE", ".gitignore"],
        readme_text=(
            "# Widget\n\n"
            "This project is a small self hosted widget manager built for a household "
            "that wanted something simpler than a spreadsheet and quicker than a shared notes app.\n"
        ),
    )
    defaults.update(overrides)
    return RepoFacts(**defaults)


def run_self_test() -> int:
    try:
        checklist = load_checklist(CHECKLIST_PATH)
    except ChecklistError as e:
        print(f"self-test cannot load checklist.yaml: {e}", file=sys.stderr)
        return 1
    settings = checklist["settings"]

    cases: list[tuple[str, bool, str]] = []

    def expect(label: str, actual: str, expected: str) -> None:
        cases.append((label, actual == expected, f"got {actual!r}, expected {expected!r}"))

    # -- direct helper assertions -----------------------------------------
    expect("glob: **/.env matches nested path", str(path_matches_glob("backend/.env", "**/.env")), "True")
    expect(
        "glob: **/.env does not match .env.example",
        str(path_matches_glob("backend/.env.example", "**/.env")),
        "False",
    )
    expect("glob: id_rsa* matches nested basename", str(path_matches_glob("keys/id_rsa", "id_rsa*")), "True")

    # -- has_description ---------------------------------------------------
    expect("has_description pass", check_has_description(_base_facts(description="hi"), settings)[0], "PASS")
    expect("has_description fail", check_has_description(_base_facts(description="  "), settings)[0], "FAIL")

    # -- has_topics ----------------------------------------------------------
    min_topics = settings["min_topics"]
    expect(
        "has_topics pass",
        check_has_topics(_base_facts(topics=[f"t{i}" for i in range(min_topics)]), settings)[0],
        "PASS",
    )
    expect("has_topics fail", check_has_topics(_base_facts(topics=[]), settings)[0], "FAIL")

    # -- license_declared / license_file --------------------------------
    expect("license_declared pass", check_license_declared(_base_facts(license_declared=True), settings)[0], "PASS")
    expect("license_declared fail", check_license_declared(_base_facts(license_declared=False), settings)[0], "FAIL")
    expect(
        "license_file pass",
        check_license_file(_base_facts(tree_paths=["LICENSE", "README.md"]), settings)[0],
        "PASS",
    )
    expect(
        "license_file fail",
        check_license_file(_base_facts(tree_paths=["README.md"]), settings)[0],
        "FAIL",
    )

    # -- default_branch_main ----------------------------------------------
    expect(
        "default_branch_main pass",
        check_default_branch_main(_base_facts(default_branch=settings["default_branch"]), settings)[0],
        "PASS",
    )
    expect(
        "default_branch_main fail",
        check_default_branch_main(_base_facts(default_branch="master"), settings)[0],
        "FAIL",
    )

    # -- no_tracked_secrets: the nested-.env case called out explicitly -----
    expect(
        "no_tracked_secrets pass (clean tree + .env.example)",
        check_no_tracked_secrets(_base_facts(tree_paths=["README.md", ".env.example"]), settings)[0],
        "PASS",
    )
    expect(
        "no_tracked_secrets fail (nested .env)",
        check_no_tracked_secrets(_base_facts(tree_paths=["backend/.env", "README.md"]), settings)[0],
        "FAIL",
    )

    # -- no_tracked_agent_instructions: root CLAUDE.md fails, docs/CLAUDE.md
    #    is exempt, and no such file at all passes. -------------------------
    expect(
        "no_tracked_agent_instructions fail (root CLAUDE.md)",
        check_no_tracked_agent_instructions(_base_facts(tree_paths=["CLAUDE.md", "README.md"]), settings)[0],
        "FAIL",
    )
    expect(
        "no_tracked_agent_instructions pass (docs/CLAUDE.md exempt)",
        check_no_tracked_agent_instructions(_base_facts(tree_paths=["docs/CLAUDE.md", "README.md"]), settings)[0],
        "PASS",
    )
    expect(
        "no_tracked_agent_instructions pass (no such file)",
        check_no_tracked_agent_instructions(_base_facts(tree_paths=["README.md"]), settings)[0],
        "PASS",
    )

    # -- gitignore_present ---------------------------------------------------
    expect(
        "gitignore_present pass",
        check_gitignore_present(_base_facts(tree_paths=[".gitignore"]), settings)[0],
        "PASS",
    )
    expect(
        "gitignore_present fail",
        check_gitignore_present(_base_facts(tree_paths=["README.md"]), settings)[0],
        "FAIL",
    )

    # -- readme_present --------------------------------------------------
    expect("readme_present pass", check_readme_present(_base_facts(readme_text="# X\n"), settings)[0], "PASS")
    expect("readme_present fail", check_readme_present(_base_facts(readme_text=None), settings)[0], "FAIL")

    # -- readme_has_intro: long paragraph, badge-skip, and too-short --------
    long_intro = _base_facts()  # default fixture already has a >=20 word intro
    expect("readme_has_intro pass", check_readme_has_intro(long_intro, settings)[0], "PASS")
    badge_readme = (
        "# Widget\n\n"
        "[![Build](https://img.shields.io/badge.svg)](https://ci.example.com)\n\n"
        "This project is a small self hosted widget manager built for a household "
        "that wanted something simpler than a spreadsheet and quicker than notes.\n"
    )
    expect(
        "readme_has_intro pass (badge line skipped)",
        check_readme_has_intro(_base_facts(readme_text=badge_readme), settings)[0],
        "PASS",
    )
    expect(
        "readme_has_intro fail (too short)",
        check_readme_has_intro(_base_facts(readme_text="# Widget\n\nToo short.\n"), settings)[0],
        "FAIL",
    )

    # -- section_patterns: heading matches, body prose must not -------------
    heading_readme = "# Widget\n\n## Quickstart\n\nRun `make up`.\n"
    prose_readme = "# Widget\n\nA paragraph that mentions quickstart informally.\n\n## Something Else\n\ntext\n"
    expect(
        "readme_has_quickstart pass (real heading)",
        check_readme_has_quickstart(_base_facts(readme_text=heading_readme), settings)[0],
        "PASS",
    )
    expect(
        "readme_has_quickstart fail (word only in body prose)",
        check_readme_has_quickstart(_base_facts(readme_text=prose_readme), settings)[0],
        "FAIL",
    )

    # -- readme_links_resolve: directory, anchor, external all fine; a
    #    deleted-file target must fail. --------------------------------------
    link_tree = ["docs/setup.md", "guide.md"]
    good_links_readme = (
        "# Widget\n\n"
        "See the [docs](docs) folder, the [guide](guide.md#install) and "
        "[upstream](https://example.com) for more.\n"
    )
    expect(
        "readme_links_resolve pass (dir + anchor + external)",
        check_readme_links_resolve(_base_facts(tree_paths=link_tree, readme_text=good_links_readme), settings)[0],
        "PASS",
    )
    broken_links_readme = good_links_readme + "\nAlso see [gone](removed.md).\n"
    expect(
        "readme_links_resolve fail (deleted file)",
        check_readme_links_resolve(_base_facts(tree_paths=link_tree, readme_text=broken_links_readme), settings)[0],
        "FAIL",
    )

    # -- readme_has_screenshots ----------------------------------------------
    expect(
        "readme_has_screenshots pass (marker path)",
        check_readme_has_screenshots(_base_facts(tree_paths=["docs/screenshots/a.png"]), settings)[0],
        "PASS",
    )
    expect(
        "readme_has_screenshots pass (image embed)",
        check_readme_has_screenshots(_base_facts(readme_text="# W\n\n![shot](img.png)\n"), settings)[0],
        "PASS",
    )
    expect(
        "readme_has_screenshots fail",
        check_readme_has_screenshots(_base_facts(tree_paths=["README.md"]), settings)[0],
        "FAIL",
    )

    # -- quickstart_artifact --------------------------------------------------
    expect(
        "quickstart_artifact pass",
        check_quickstart_artifact(_base_facts(tree_paths=["Makefile"]), settings)[0],
        "PASS",
    )
    expect(
        "quickstart_artifact fail",
        check_quickstart_artifact(_base_facts(tree_paths=["README.md"]), settings)[0],
        "FAIL",
    )

    # -- env_example_present: SKIP / PASS / FAIL three-way ------------------
    expect(
        "env_example_present SKIP (no env indicator)",
        check_env_example_present(_base_facts(tree_paths=["README.md", "src/main.py"]), settings)[0],
        "SKIP",
    )
    expect(
        "env_example_present pass",
        check_env_example_present(_base_facts(tree_paths=["docker-compose.yml", ".env.example"]), settings)[0],
        "PASS",
    )
    expect(
        "env_example_present fail",
        check_env_example_present(_base_facts(tree_paths=["Dockerfile"]), settings)[0],
        "FAIL",
    )

    # -- demo_path --------------------------------------------------------
    expect("demo_path pass", check_demo_path(_base_facts(tree_paths=["demo/seed.py"]), settings)[0], "PASS")
    expect("demo_path fail", check_demo_path(_base_facts(tree_paths=["README.md"]), settings)[0], "FAIL")

    # -- ci_workflow_present --------------------------------------------------
    expect(
        "ci_workflow_present pass",
        check_ci_workflow_present(_base_facts(tree_paths=[".github/workflows/ci.yml"]), settings)[0],
        "PASS",
    )
    expect(
        "ci_workflow_present fail",
        check_ci_workflow_present(_base_facts(tree_paths=["README.md"]), settings)[0],
        "FAIL",
    )

    # -- root_md_budget --------------------------------------------------
    budget = settings["root_md_budget"]
    under_budget = [f"DOC{i}.md" for i in range(budget)]
    over_budget = [f"DOC{i}.md" for i in range(budget + 1)]
    expect("root_md_budget pass", check_root_md_budget(_base_facts(tree_paths=under_budget), settings)[0], "PASS")
    expect("root_md_budget fail", check_root_md_budget(_base_facts(tree_paths=over_budget), settings)[0], "FAIL")

    # -- root_md_linked --------------------------------------------------
    expect(
        "root_md_linked pass (allowed + referenced)",
        check_root_md_linked(
            _base_facts(
                tree_paths=["README.md", "CHANGELOG.md", "NOTES.md"],
                readme_text="# Widget\n\nSee NOTES.md for background.\n",
            ),
            settings,
        )[0],
        "PASS",
    )
    expect(
        "root_md_linked fail (orphan)",
        check_root_md_linked(
            _base_facts(tree_paths=["README.md", "ORPHAN.md"], readme_text="# Widget\n\nintro text here.\n"),
            settings,
        )[0],
        "FAIL",
    )

    # -- status_doc --------------------------------------------------
    expect(
        "status_doc pass",
        check_status_doc(_base_facts(tree_paths=["STATUS.md"]), settings)[0],
        "PASS",
    )
    expect("status_doc fail", check_status_doc(_base_facts(tree_paths=["README.md"]), settings)[0], "FAIL")

    # -- index_covers_skills --------------------------------------------------
    # The exempt globs live in the private overlay, not the public bar, so this
    # case supplies them explicitly rather than relying on the shipped settings.
    expect(
        "index_covers_skills pass (named + exempt)",
        check_index_covers_skills(
            _base_facts(local_dir_names=["alpha", "beta-pack-one"], local_index_text="- alpha: does things"),
            {**settings, "skills_index_exempt": ["beta-*"]},
        )[0],
        "PASS",
    )
    # With no overlay merged in there are no exemptions at all, which is the
    # shape a public clone and CI both run: absence must mean "exempt nothing",
    # never a crash.
    expect(
        "index_covers_skills fail (no exempt key at all)",
        check_index_covers_skills(
            _base_facts(local_dir_names=["alpha", "beta-pack-one"], local_index_text="- alpha: does things"),
            {k: v for k, v in settings.items() if k != "skills_index_exempt"},
        )[0],
        "FAIL",
    )
    expect(
        "index_covers_skills fail (uncovered dir)",
        check_index_covers_skills(
            _base_facts(local_dir_names=["alpha", "mystery-block"], local_index_text="- alpha: does things"),
            settings,
        )[0],
        "FAIL",
    )

    # -- level_with_origin --------------------------------------------------
    expect(
        "level_with_origin pass",
        check_level_with_origin(
            _base_facts(local_status_porcelain="", local_ahead=0, local_behind=0), settings
        )[0],
        "PASS",
    )
    expect(
        "level_with_origin fail (dirty + ahead/behind)",
        check_level_with_origin(
            _base_facts(local_status_porcelain=" M x.py\n", local_ahead=2, local_behind=1), settings
        )[0],
        "FAIL",
    )

    # -- local-only checks must SKIP, not FAIL or crash, when the working
    #    copy they need is absent from this host (e.g. a sweep box that only
    #    has GitHub access, not the laptop's filesystem). Both directions:
    #    present+evaluable is unchanged (the four cases just above), absent
    #    yields SKIP with a detail naming the missing path. --------------------
    absent_facts = _base_facts(local_skills_unavailable="no local working copy at '/nonexistent' on this host")
    expect(
        "index_covers_skills SKIP when local working copy is absent",
        check_index_covers_skills(absent_facts, settings)[0],
        "SKIP",
    )
    expect(
        "index_covers_skills SKIP detail names the missing path",
        str("/nonexistent" in check_index_covers_skills(absent_facts, settings)[1]),
        "True",
    )
    expect(
        "level_with_origin SKIP when local working copy is absent",
        check_level_with_origin(absent_facts, settings)[0],
        "SKIP",
    )

    # Path present but not a usable git working copy against origin: only
    # level_with_origin cares about git-ness, so index_covers_skills (which
    # only needs a directory listing) must keep evaluating normally - proves
    # the two unavailable-reason fields are independent, not one flag.
    not_a_git_repo_facts = _base_facts(
        local_dir_names=["alpha"],
        local_index_text="- alpha: does things",
        local_git_unavailable="/some/path is not a usable git working copy against origin/main: fatal: no origin",
    )
    expect(
        "level_with_origin SKIP when path exists but isn't a usable git working copy",
        check_level_with_origin(not_a_git_repo_facts, settings)[0],
        "SKIP",
    )
    expect(
        "index_covers_skills still evaluates normally when only git-ness is unavailable",
        check_index_covers_skills(not_a_git_repo_facts, settings)[0],
        "PASS",
    )

    # -- SKIP must not count as MUST/SHOULD failure, even at MUST tier, and
    #    must not silently read as a pass either - exercised through the
    #    actual run_checks path, not just the bare check function. ----------
    skills_repo_checks = {"index_covers_skills": "MUST", "level_with_origin": "SHOULD"}
    skip_run_results = run_checks(absent_facts, skills_repo_checks, settings)
    expect(
        "run_checks: absent working copy yields SKIP status for both, not PASS/FAIL",
        str(sorted(r.status for r in skip_run_results)),
        "['SKIP', 'SKIP']",
    )
    expect(
        "run_checks: a SKIPped MUST does not count as a MUST failure for exit-code purposes",
        str(any(r.tier == "MUST" and r.status == "FAIL" for r in skip_run_results)),
        "False",
    )

    # -- --suggest for a SKIPped local check: names what would have to be
    #    true to evaluate it, not a fix for a failure that was never
    #    established. A different check's SKIP (env_example_present, which
    #    means "doesn't apply" rather than "cannot be evaluated") still gets
    #    no suggestion at all - the carve-out is narrow, not blanket. --------
    index_skip_result = CheckResult("index_covers_skills", "MUST", "SKIP", absent_facts.local_skills_unavailable)
    index_skip_suggestion = suggestion_for(absent_facts, index_skip_result, settings) or ""
    expect(
        "suggestion_for: SKIPped index_covers_skills names the remedy, not a fix",
        str("Run this on the machine" in index_skip_suggestion and "skills_local_path" in index_skip_suggestion),
        "True",
    )
    origin_skip_result = CheckResult("level_with_origin", "SHOULD", "SKIP", not_a_git_repo_facts.local_git_unavailable)
    origin_skip_suggestion = suggestion_for(not_a_git_repo_facts, origin_skip_result, settings) or ""
    expect(
        "suggestion_for: SKIPped level_with_origin names the remedy, not a fix",
        str("Run this on the machine" in origin_skip_suggestion),
        "True",
    )
    env_skip_result = CheckResult("env_example_present", "MUST", "SKIP", "not indicated")
    expect(
        "suggestion_for: an ordinary SKIP (env_example_present) still gets no suggestion",
        str(suggestion_for(_base_facts(), env_skip_result, settings)),
        "None",
    )

    # -- --suggest: every implemented check id must have a suggestion
    #    template. This is the assertion that matters most in this block: it
    #    means a future check cannot ship without a remedy, and this fails
    #    loudly (not a silent gap in the report) if someone forgets one. -----
    missing_suggestions = sorted(set(CHECKS) - set(SUGGESTIONS))
    expect("suggestion coverage: every check id has a SUGGESTIONS template", str(missing_suggestions), "[]")
    orphaned_suggestions = sorted(set(SUGGESTIONS) - set(CHECKS))
    expect("suggestion coverage: no orphaned SUGGESTIONS entries", str(orphaned_suggestions), "[]")

    # -- suggestion_for: None on PASS/SKIP, a real string on FAIL ------------
    pass_result = CheckResult("has_description", "MUST", "PASS", "hi")
    fail_result = CheckResult("has_description", "MUST", "FAIL", "no description set")
    skip_result = CheckResult("env_example_present", "MUST", "SKIP", "not indicated")
    expect(
        "suggestion_for: None on PASS",
        str(suggestion_for(_base_facts(), pass_result, settings)),
        "None",
    )
    expect(
        "suggestion_for: None on SKIP",
        str(suggestion_for(_base_facts(), skip_result, settings)),
        "None",
    )
    fail_suggestion = suggestion_for(_base_facts(), fail_result, settings)
    expect(
        "suggestion_for: non-empty actionable string on FAIL",
        str(bool(fail_suggestion) and "gh repo edit" in fail_suggestion),
        "True",
    )

    # -- suggestion templates that render per-item detail lists: sanity-check
    #    a couple of the multi-path ones actually mention each path. --------
    secrets_fail = CheckResult("no_tracked_secrets", "MUST", "FAIL", "backend/.env; keys/id_rsa")
    secrets_suggestion = suggestion_for(_base_facts(), secrets_fail, settings) or ""
    expect(
        "suggestion: no_tracked_secrets mentions every offending path",
        str(all(p in secrets_suggestion for p in ["backend/.env", "keys/id_rsa"])),
        "True",
    )
    expect(
        "suggestion: no_tracked_secrets warns about git history",
        str("history" in secrets_suggestion),
        "True",
    )

    # -- owners_from_assignments: unique owners, sorted -----------------------
    expect(
        "owners_from_assignments: derives unique, sorted owners",
        str(owners_from_assignments({"acme/x": "p", "acme/y": "p", "beta/z": "p"})),
        "['acme', 'beta']",
    )

    # -- reconcile_enumeration: the enumeration-vs-overlay reconciliation that
    #    makes an unassigned repo a loud finding instead of a silent gap. Each
    #    case below is one of the required fixtures: missing from both maps
    #    (fails), present in assignments (passes), present in excluded
    #    (passes), and archived-and-excluded (informational, does not fail).
    #    A fifth covers the rule that forces an archived repo into
    #    excluded: archived-but-still-in-assignments still fails. ------------
    enum_assignments = {"acme/widget": "public-portfolio", "acme/tool": "private-work"}
    enum_excluded = {"acme/scratch": "work scratch, not a project"}

    # missing from both maps -> unassigned, and a failure
    missing_case = [{"repo": "acme/forgotten", "archived": False}]
    unassigned, informational = reconcile_enumeration(missing_case, enum_assignments, enum_excluded)
    expect("reconcile: repo in neither map is unassigned", str([u["repo"] for u in unassigned]), "['acme/forgotten']")
    expect("reconcile: repo in neither map has no informational entry", str(informational), "[]")

    # present in assignments (non-archived) -> accounted for, no finding
    assigned_case = [{"repo": "acme/widget", "archived": False}]
    unassigned, informational = reconcile_enumeration(assigned_case, enum_assignments, enum_excluded)
    expect("reconcile: assigned repo is not unassigned", str(unassigned), "[]")

    # present in excluded (non-archived) -> accounted for, no finding at all
    excluded_case = [{"repo": "acme/scratch", "archived": False}]
    unassigned, informational = reconcile_enumeration(excluded_case, enum_assignments, enum_excluded)
    expect("reconcile: excluded repo is not unassigned", str(unassigned), "[]")
    expect("reconcile: non-archived excluded repo gets no informational note", str(informational), "[]")

    # archived AND excluded -> informational, explicitly not a failure
    archived_excluded_case = [{"repo": "acme/scratch", "archived": True}]
    unassigned, informational = reconcile_enumeration(archived_excluded_case, enum_assignments, enum_excluded)
    expect("reconcile: archived+excluded is not a failure", str(unassigned), "[]")
    expect("reconcile: archived+excluded produces an informational entry", str(informational[0]["repo"]), "acme/scratch")

    # archived but only in assignments, NOT in excluded -> still unassigned,
    # still fails - the scenario of a repo archived while its overlay entry
    # still sits in assignments rather than excluded.
    archived_assigned_case = [{"repo": "acme/tool", "archived": True}]
    unassigned, informational = reconcile_enumeration(archived_assigned_case, enum_assignments, enum_excluded)
    expect(
        "reconcile: archived-but-only-assigned still fails",
        str([u["repo"] for u in unassigned]),
        "['acme/tool']",
    )

    # -- suggestion_for_unassigned: both remedies named for the plain case,
    #    excluded specifically recommended for the archived case. ------------
    plain_suggestion = suggestion_for_unassigned({"repo": "acme/forgotten", "archived": False}, settings)
    expect(
        "suggestion_for_unassigned: plain case names both remedies",
        str(all(s in plain_suggestion for s in ["assignments", "excluded"])),
        "True",
    )
    archived_suggestion = suggestion_for_unassigned({"repo": "acme/tool", "archived": True}, settings)
    expect(
        "suggestion_for_unassigned: archived case recommends excluded",
        str("excluded" in archived_suggestion),
        "True",
    )

    failures = [c for c in cases if not c[1]]
    if failures:
        for label, _, detail in failures:
            print(f"FAIL: {label} ({detail})")
        print(f"{len(cases) - len(failures)}/{len(cases)} passed")
        return 1

    print(f"PASS {len(cases)}/{len(cases)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
