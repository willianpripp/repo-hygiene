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
import posixpath
import re
import subprocess
import sys
from dataclasses import dataclass, field
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
    # Only the FILE entries of tree_paths. Defaulted to None so every existing
    # fixture keeps working; when None, content-fetching checks fall back to
    # tree_paths, which is what produced the directory-as-file bug, so the
    # gatherers always set it explicitly.
    blob_paths: list[str] | None = None
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
    # Populated only when a profile actually enables docs_links_resolve (see
    # gather_repo_facts): path -> text for tracked docs/**/*.md files, up to
    # DOCS_LINKS_MAX_FILES. None means the check never ran at all; {} means it
    # ran and found no docs/ markdown to check (the SKIP case).
    docs_file_texts: dict[str, str] | None = None
    docs_files_skipped: list[str] = field(default_factory=list)
    # Which backend populated this RepoFacts. Always "github" in this
    # GitHub-only build (kept as a field, not a bare constant, since
    # render_table/render_json display it per repo).
    platform: str = "github"
    # Populated only when a profile enables no_work_identifiers AND the
    # private overlay actually configured work_identifier_patterns (no point
    # fetching content that will just SKIP). path -> text for every eligible
    # tracked file (not under vendor/, not a known-binary extension), up to
    # WORK_IDENTIFIER_MAX_FILES. None means never gathered; {} means gathered
    # and the tree had nothing eligible to scan.
    work_scan_file_texts: dict[str, str] | None = None
    work_scan_files_skipped: list[str] = field(default_factory=list)


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


def resolve_link_target(raw_target: str, base_dir: str) -> str:
    """Resolve a markdown link target relative to the directory of the file
    that contains it (base_dir), not the repo root. Shared by
    readme_links_resolve (base_dir "" - a root README has no directory to be
    relative to) and docs_links_resolve (base_dir = the containing docs file's
    directory), so a link written as ../../README.md inside docs/history/x.md
    resolves to README.md at the root instead of being checked against the
    literal, nonexistent string "docs/history/../../README.md"."""
    joined = posixpath.join(base_dir, raw_target) if base_dir else raw_target
    return posixpath.normpath(joined).lstrip("/")


def find_broken_links(text: str, base_dir: str, tree_paths: list[str]) -> list[tuple[str, str]]:
    """Every relative link in `text` that does not resolve, as (raw target as
    written, resolved path) pairs. The one implementation both
    readme_links_resolve and docs_links_resolve call, so the extraction and
    resolution rules (skip external/mailto/bare-anchor, strip fragment/query,
    resolve relative to the containing file) can never drift between the two
    checks."""
    broken: list[tuple[str, str]] = []
    for raw in extract_relative_link_targets(text):
        resolved = resolve_link_target(raw, base_dir)
        if not target_exists_in_tree(resolved, tree_paths):
            broken.append((raw, resolved))
    return broken


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
    section_patterns = settings.get("section_patterns", {})
    if pattern_key not in section_patterns:
        # Defensive, not expected in normal operation: every section check
        # this shares (quickstart/stack/running_for_real/how_built/structure)
        # always has a matching entry in checklist.yaml's section_patterns
        # today. This branch exists for an older checklist.yaml that predates
        # a given pattern_key (e.g. `structure`, added for
        # readme_has_structure) - SKIP with a clear reason rather than a
        # KeyError crash or a silent, unearned PASS.
        return "SKIP", f"no {pattern_key!r} entry in settings.section_patterns - cannot evaluate this section check"
    if facts.readme_text is None:
        return "FAIL", "no README to check"
    pattern = section_patterns[pattern_key]
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


def section_body(readme_text: str, pattern: str) -> str | None:
    """The lines under the first heading matching `pattern`, up to the next
    heading of any level. None when no heading matches. Pure, so the
    extraction is covered by --self-test."""
    lines = readme_text.split("\n")
    start = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith("#") and re.search(pattern, line, re.IGNORECASE):
            start = i + 1
            break
    if start is None:
        return None
    body: list[str] = []
    for line in lines[start:]:
        if line.lstrip().startswith("#"):
            break
        body.append(line)
    return "\n".join(body)


def looks_like_a_tree(body: str) -> bool:
    """Is there an actual directory tree in here, rather than prose about one.

    Willian's rule (2026-08-16), and the reason it is enforced rather than
    asked for: three repos had a perfectly good file-by-file section written
    as a flat list, which reads fine but cannot be scanned. A tree shows the
    SHAPE of the repo in one glance; a list makes you infer it from paths.

    Deliberately generous about how the tree is drawn - box-drawing
    characters are the common form, but an indented listing of paths is a
    tree too. What it will not accept is a fenced block with no paths in it,
    a section with no fenced block at all, or a FLAT list of paths.

    That last clause was added 2026-08-16 after this check passed exactly the
    thing it exists to reject. A new repo shipped a Layout section of six
    unindented `dir/file.py  description` lines; it matched "three or more
    lines starting with a path" and scored PASS, while showing no shape at all.
    Willian caught it by eye, which is the failure: a check that only agrees
    with a human who already spotted the problem is not doing any work.

    So a flat list no longer counts. A tree has to show HIERARCHY - either
    box-drawing characters, or genuine indentation with more than one level."""
    for block in re.findall(r"```[^\n]*\n(.*?)```", body, re.DOTALL):
        if any(ch in block for ch in ("├", "└", "│")):
            return True
        lines = [ln for ln in block.split("\n") if ln.strip()]
        path_lines = [ln for ln in lines if re.match(r"^\s*\S+/", ln)]
        if len(path_lines) < 3:
            continue
        # Indentation is what separates a tree from a list, and it is measured
        # across EVERY line in the block rather than only the ones ending in a
        # slash. In `app/` / `  main.py` / `docs/` the hierarchy is carried
        # entirely by the child lines, which are not paths at all; looking only
        # at path lines saw three depth-zero entries and called a real tree flat.
        depths = {len(ln) - len(ln.lstrip()) for ln in lines}
        if len(depths) > 1:
            return True
    return False


def check_readme_has_structure(facts: RepoFacts, settings: dict[str, Any]) -> tuple[str, str]:
    """A stranger should be able to see where the logic lives before reading
    any code, and see it as a SHAPE rather than a list. Heading-only matching
    like every other section check, plus one extra requirement the others do
    not have: the section must actually contain a tree."""
    status, detail = _section_status(facts, settings, "structure")
    if status != "PASS":
        return status, detail
    pattern = settings["section_patterns"]["structure"]
    body = section_body(facts.readme_text or "", pattern)
    if body is None:  # _section_status already matched, so this is unreachable in practice
        return "FAIL", "structure heading matched but its section could not be read"
    if not looks_like_a_tree(body):
        return "FAIL", "structure section has no directory tree in a fenced block, only prose"
    return "PASS", detail


def check_readme_links_resolve(facts: RepoFacts, settings: dict[str, Any]) -> tuple[str, str]:
    if facts.readme_text is None:
        return "PASS", "no README, nothing to check (see readme_present)"
    # base_dir "" - the README lives at the repo root, so its relative links
    # are already relative to root and need no adjustment.
    broken = find_broken_links(facts.readme_text, "", facts.tree_paths)
    if not broken:
        return "PASS", "all relative links resolve"
    return "FAIL", "; ".join(raw for raw, _resolved in broken)


# Cap on how many docs/**/*.md files docs_links_resolve fetches content for.
# Fetching a file's content costs one API call, so this trades completeness
# for a bounded, predictable number of calls on a repo with a very large docs
# tree. Files past the cap are named in the check's own detail (never
# silently dropped) rather than truncated without a trace.
DOCS_LINKS_MAX_FILES = 50

# no_work_identifiers reads every eligible tracked file, which is a much
# larger set than docs/**/*.md, so it gets its own (larger) cap rather than
# reusing DOCS_LINKS_MAX_FILES. Extensions here are skipped from selection
# entirely (never fetched, never counted against the cap) because they are
# either not text (a regex can't meaningfully match decoded image/archive
# bytes) or, for the ones that are technically decodable, not files anyone
# writes a stray comment into.
BINARY_FILE_EXTENSIONS = frozenset(
    {
        ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp",
        ".pdf", ".zip", ".tar", ".gz", ".tgz", ".bz2", ".7z", ".rar",
        ".exe", ".dll", ".so", ".dylib", ".bin", ".class", ".jar", ".war",
        ".woff", ".woff2", ".ttf", ".otf", ".eot",
        ".mp3", ".mp4", ".mov", ".avi", ".wav", ".flac", ".ogg", ".webm",
        ".pyc", ".o", ".a", ".obj",
        ".db", ".sqlite", ".sqlite3", ".parquet", ".pkl", ".npy", ".npz",
    }
)
WORK_IDENTIFIER_MAX_FILES = 300


def _is_vendor_path(path: str) -> bool:
    """True if any path component is literally "vendor" - the one directory
    no_work_identifiers is told to skip outright: third-party code isn't this
    repo's vocabulary to police, and scanning it would just burn calls and
    risk false positives on someone else's comments."""
    return "vendor" in path.split("/")


def _looks_binary(path: str) -> bool:
    return Path(path).suffix.lower() in BINARY_FILE_EXTENSIONS


def blob_paths_from_tree_entries(entries: list[dict[str, Any]]) -> list[str]:
    """Keep only `type: "blob"` (regular file) entries from a raw git-tree API
    response - GitHub's `git/trees` returns a `type` per entry ("blob" for a
    file, "tree" for a directory, plus "commit" for a submodule). This is the
    fix for a real bug: a directory entry reaching a content-fetch call makes
    GitHub's contents API return a JSON array where a file object was
    expected (`.content` extraction then blows up, see gh_api's error path).
    Any check that fetches file CONTENT (
    no_work_identifiers, docs_links_resolve) must select from blob paths
    only; checks that merely pattern-match against tracked PATHS (secrets,
    CI globs, root .md budget, ...) are unaffected and keep reading the full
    tree_paths list, which still includes directory entries exactly as
    before this fix. A pure function so the type-filtering itself - not just
    the selection logic built on top of it - is covered by --self-test
    without a live tree to page through."""
    return [entry["path"] for entry in entries if entry.get("type") == "blob" and "path" in entry]


def select_work_scan_paths(blob_paths: list[str]) -> tuple[list[str], list[str]]:
    """Every tracked FILE no_work_identifiers should read - not under a
    vendor/ directory, not a known-binary extension - capped at
    WORK_IDENTIFIER_MAX_FILES so a large repo can't turn one enabled check
    into an unbounded number of API calls. `blob_paths` must already be
    filtered to blob (file) entries by the caller (see
    blob_paths_from_tree_entries) - this function has no tree-entry `type` to
    check against a bare path string, so passing a mixed blob+tree list back
    in is exactly the bug this whole fix closes. Returns (paths to fetch,
    paths dropped past the cap). Paths excluded for being vendor/binary are
    not reported anywhere: they were never eligible to begin with, unlike the
    cap, which drops otherwise-eligible files and must say so (see
    check_no_work_identifiers's "inconclusive" path). A pure function so this
    selection logic - including the pagination-shaped cap behavior - is
    covered by --self-test without any network access."""
    eligible = sorted(p for p in blob_paths if not _is_vendor_path(p) and not _looks_binary(p))
    return eligible[:WORK_IDENTIFIER_MAX_FILES], eligible[WORK_IDENTIFIER_MAX_FILES:]


def check_docs_links_resolve(facts: RepoFacts, settings: dict[str, Any]) -> tuple[str, str]:
    if facts.docs_file_texts is None:
        # This check ran (tier wasn't OFF) but the gathering step that
        # populates docs_file_texts never ran either - a real bug (gathering
        # and the profile's checks disagreeing), not "no docs/ directory",
        # so it stays a loud FAIL rather than a quiet SKIP.
        return "FAIL", "docs file facts were not gathered"
    if not facts.docs_file_texts:
        # A repo with no docs/ directory (or none with any .md in it) has
        # nothing to check - reporting PASS here would claim link hygiene
        # that was never actually evaluated.
        return "SKIP", "no markdown files under docs/ (nothing to check)"

    broken_entries: list[str] = []
    for path in sorted(facts.docs_file_texts):
        text = facts.docs_file_texts[path]
        base_dir = posixpath.dirname(path)
        for raw, resolved in find_broken_links(text, base_dir, facts.tree_paths):
            broken_entries.append(f"{path} -> {raw} (resolves to {resolved})")

    detail = "; ".join(broken_entries) if broken_entries else "all relative links resolve"
    if facts.docs_files_skipped:
        detail += (
            f" || NOTE: {len(facts.docs_files_skipped)} docs file(s) not checked "
            f"(over the {DOCS_LINKS_MAX_FILES}-file cap): {', '.join(facts.docs_files_skipped)}"
        )
    return ("FAIL", detail) if broken_entries else ("PASS", detail)


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


def check_no_work_identifiers(facts: RepoFacts, settings: dict[str, Any]) -> tuple[str, str]:
    """Catches work vocabulary (employer name, client names, internal group/
    project names) drifting into a personal public repo through a comment, a
    fixture, or an example - the exact failure mode that leaked the employer
    name into this project's own history.

    The pattern list (settings["work_identifier_patterns"]) MUST live only in
    the private overlay, never in the published checklist.yaml - see
    SKILL.md/checklist.yaml's own comments on the overlay split. This
    function enforces that indirectly: it never writes the pattern list
    anywhere, and it never echoes a matched term into its own report (detail
    strings below name a file and a pattern INDEX, never the matched text or
    the pattern's source text) - a report is exactly the kind of thing that
    gets pasted somewhere, and quoting the leak back would just relocate it.
    """
    patterns = settings.get("work_identifier_patterns")
    if not patterns:
        # No private overlay, or an overlay that hasn't configured this yet
        # (the CI-on-a-public-repo shape, where only checklist.yaml exists) -
        # SKIP, never PASS. A silent PASS here would report "clean" on
        # exactly the repos that have no way to actually check.
        return (
            "SKIP",
            "no work_identifier_patterns configured (private overlay only) - this check needs a "
            "private pattern list and cannot run without one",
        )

    compiled: list[re.Pattern[str]] = []
    for idx, pattern in enumerate(patterns):
        try:
            compiled.append(re.compile(pattern))
        except re.error as e:
            # Never interpolate `pattern` itself into the message: the
            # pattern IS the sensitive term (e.g. "(?i)acmecorp"), so naming
            # the index plus the regex engine's structural complaint is the
            # only safe way to report a bad entry.
            return "FAIL", f"work_identifier_patterns[{idx}] is not a valid regex: {e}"

    if facts.work_scan_file_texts is None:
        # Check was enabled and patterns exist, so gathering should have run
        # - if it didn't, that's a real gathering/profile bug, same "loud
        # FAIL, not a quiet SKIP" precedent as docs_links_resolve.
        return "FAIL", "file contents were not gathered for this check"

    hits: list[str] = []
    for path in sorted(facts.work_scan_file_texts):
        text = facts.work_scan_file_texts[path]
        for idx, rx in enumerate(compiled):
            if rx.search(text):
                hits.append(f"file {path} matches work pattern #{idx}")
                break  # one hit is enough to fail this file; no need to pile on every pattern

    cap_note = (
        f" || NOTE: {len(facts.work_scan_files_skipped)} file(s) not scanned "
        f"(over the {WORK_IDENTIFIER_MAX_FILES}-file cap)"
        if facts.work_scan_files_skipped
        else ""
    )

    if hits:
        return "FAIL", "; ".join(hits) + cap_note

    if facts.work_scan_files_skipped:
        # The cap was hit and nothing turned up in what WAS scanned - but
        # "nothing found in a partial scan" is not the same claim as "clean,"
        # and reporting PASS here would be exactly that overclaim. FAIL
        # (there is no separate "inconclusive" status) forces a human look
        # instead of blending into ordinary SKIP noise.
        return (
            "FAIL",
            f"inconclusive: {len(facts.work_scan_files_skipped)} tracked file(s) were not scanned "
            f"(over the {WORK_IDENTIFIER_MAX_FILES}-file cap) - not reported as clean",
        )

    return "PASS", f"{len(facts.work_scan_file_texts)} file(s) scanned, no work identifiers found"


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
    "docs_links_resolve": check_docs_links_resolve,
    "no_work_identifiers": check_no_work_identifiers,
    "readme_has_structure": check_readme_has_structure,
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


def suggest_readme_has_structure(facts: RepoFacts, result: CheckResult, settings: dict[str, Any]) -> str:
    # Richer than the generic _section_suggestion one-liner on purpose: a
    # structure block is only worth having if it's accurate, so the remedy
    # has to say how to keep it that way, not just that a heading is missing.
    pattern = settings.get("section_patterns", {}).get("structure")
    pattern_note = f" (matches the required pattern /{pattern}/i)" if pattern else ""
    return (
        f'Add a "## Project structure" heading to the README{pattern_note}, with a fenced tree of the '
        "top two levels of the repo and a short note on what lives where, so a stranger can see where "
        "the logic lives before reading any code. Generate the tree from the actual layout rather than "
        "writing it from memory - a structure block that drifts from the real tree is worse than no "
        "structure block at all, since it actively misleads instead of just being absent."
    )


def suggest_readme_links_resolve(facts: RepoFacts, result: CheckResult, settings: dict[str, Any]) -> str:
    targets = _split_detail(result.detail)
    if not targets:
        return "Fix the broken relative link(s), or restore the missing file(s) they point to."
    return "\n".join(f'Broken link target "{t}": fix the link, or restore the file at that path.' for t in targets)


def suggest_docs_links_resolve(facts: RepoFacts, result: CheckResult, settings: dict[str, Any]) -> str:
    # The cap note (if any) is appended after " || " in the detail, not as
    # another ";"-joined broken-link entry - strip it before splitting so it
    # doesn't get misread as one.
    link_part = result.detail.split(" || NOTE:", 1)[0]
    entries = _split_detail(link_part)
    if not entries:
        return "Fix the broken relative link(s) under docs/, or restore the missing file(s) they point to."
    return "\n".join(f"Broken link in {e}: fix the link, or restore the file at that path." for e in entries)


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


def suggest_no_work_identifiers(facts: RepoFacts, result: CheckResult, settings: dict[str, Any]) -> str:
    # Never read result.detail for anything but file names / pattern indices
    # / the fixed strings this check itself writes - it was already built to
    # never contain the matched term, so this function inherits that
    # guarantee for free as long as it never adds the term back in.
    detail = result.detail
    if detail.startswith("work_identifier_patterns["):
        return (
            "This is a regex syntax error in the private overlay's work_identifier_patterns (see the "
            "check detail for which index), not a leak - fix the pattern there. No repo file needs "
            "changing for this one."
        )
    if detail.startswith("inconclusive:"):
        return (
            f"This repo has more eligible tracked files than the {WORK_IDENTIFIER_MAX_FILES}-file scan "
            "cap, so this run did not cover everything and a PASS was deliberately withheld. Re-run "
            "scoped more narrowly, raise WORK_IDENTIFIER_MAX_FILES in audit.py, or review the untouched "
            "files by hand before trusting a clean result here."
        )
    # A real hit (or several): strip the cap-note suffix first (same
    # convention as suggest_docs_links_resolve) so it isn't misread as one
    # more semicolon-joined file entry.
    hit_part = detail.split(" || NOTE:", 1)[0]
    entries = _split_detail(hit_part)
    if not entries:
        return "Remove the matching term from the flagged file, then check git history for the same term."
    lines = [f"{e}: open this file and remove the matching term." for e in entries]
    lines.append(
        "Then check git history for the same term - removing it from HEAD does not remove it from any "
        "earlier commit. If this was ever pushed, treat it like a leaked secret: a history rewrite (and "
        "force-push) is what actually scrubs it, not a new commit on top."
    )
    return "\n".join(lines)


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
    "docs_links_resolve": suggest_docs_links_resolve,
    "no_work_identifiers": suggest_no_work_identifiers,
    "readme_has_structure": suggest_readme_has_structure,
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


def _fetch_github_file_text(repo: str, path: str) -> str | None:
    """Fetch and base64-decode one file's content via the GitHub contents
    API. Shared by every GitHub-side check that needs file text
    (docs_links_resolve, no_work_identifiers), so the decode/error handling
    can never quietly drift between them. Returns None on a 404 or genuinely
    empty content - nothing to check in that case, not a broken fetch."""
    content_b64 = gh_api(f"repos/{repo}/contents/{path}", jq=".content")
    if not content_b64:
        return None
    try:
        return base64.b64decode(content_b64.replace("\n", "")).decode("utf-8", errors="replace")
    except (ValueError, UnicodeDecodeError):
        return None


def _fetch_docs_file_texts(repo: str, blob_paths: list[str]) -> tuple[dict[str, str], list[str]]:
    """Fetch content only for tracked markdown files under docs/ (recursive) -
    every other tree entry costs an API call this check does not need, and
    fetching indiscriminately would multiply the per-repo call count for a
    check that may not even be enabled. This deliberately trades completeness
    for a bounded, predictable number of calls: past DOCS_LINKS_MAX_FILES,
    remaining files are named in the return value (and, from there, in the
    check's own detail) rather than silently dropped without a trace.

    Returns (path -> text for fetched files, paths skipped past the cap)."""
    docs_md_paths = sorted(p for p in blob_paths if p.startswith("docs/") and p.lower().endswith(".md"))
    fetch_paths = docs_md_paths[:DOCS_LINKS_MAX_FILES]
    skipped_paths = docs_md_paths[DOCS_LINKS_MAX_FILES:]

    texts: dict[str, str] = {}
    for path in fetch_paths:
        text = _fetch_github_file_text(repo, path)
        if text is not None:
            texts[path] = text
    return texts, skipped_paths


def _fetch_work_scan_file_texts_github(repo: str, blob_paths: list[str]) -> tuple[dict[str, str], list[str]]:
    """no_work_identifiers's GitHub-side fetch: every eligible tracked file
    (see select_work_scan_paths), not just docs/**/*.md - the leak this check
    exists to catch was in a YAML comment and, before that, a Python function
    name, neither of which docs_links_resolve's file selection would ever
    see."""
    fetch_paths, skipped_paths = select_work_scan_paths(blob_paths)
    texts: dict[str, str] = {}
    for path in fetch_paths:
        text = _fetch_github_file_text(repo, path)
        if text is not None:
            texts[path] = text
    return texts, skipped_paths


def _check_enabled(profile_checks: dict[str, str], check_id: str) -> bool:
    """Same YAML-bool-OFF coercion handled in run_checks (a bare OFF parses
    as False, not the string "OFF") - reused here so gather_repo_facts asks
    the same question the same way, instead of a second, potentially
    diverging notion of "is this check on for this profile"."""
    raw_tier = profile_checks.get(check_id)
    tier = "OFF" if raw_tier is False else raw_tier
    return tier is not None and tier != "OFF"

def gather_repo_facts_github(
    repo: str, profile: str, profile_checks: dict[str, str], settings: dict[str, Any]
) -> RepoFacts:
    meta = gh_api(f"repos/{repo}")
    if meta is None:
        raise GhError(f"repo {repo} not found or inaccessible")
    default_branch = meta.get("default_branch")

    tree_paths: list[str] = []
    blob_paths: list[str] = []
    if default_branch:  # a truly empty repo has no default branch to list a tree for
        tree = gh_api(f"repos/{repo}/git/trees/{default_branch}?recursive=1")
        if isinstance(tree, dict):
            entries = [e for e in tree.get("tree", []) if "path" in e]
            tree_paths = [e["path"] for e in entries]
            blob_paths = blob_paths_from_tree_entries(entries)

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
        blob_paths=blob_paths,
        readme_text=readme_text,
        platform="github",
    )

    # Only spend the extra API calls when a profile actually turns this check
    # on - a profile that doesn't list docs_links_resolve (or sets it OFF)
    # must not pay for content it will never look at.
    if _check_enabled(profile_checks, "docs_links_resolve"):
        facts.docs_file_texts, facts.docs_files_skipped = _fetch_docs_file_texts(repo, blob_paths)

    # Same rule again, plus a second gate: even when the profile enables
    # no_work_identifiers, there is nothing to fetch for if the private
    # overlay hasn't configured work_identifier_patterns (the public-checklist
    # CI shape) - the check will SKIP regardless, so paying for the content
    # would be pure waste.
    if _check_enabled(profile_checks, "no_work_identifiers") and settings.get("work_identifier_patterns"):
        facts.work_scan_file_texts, facts.work_scan_files_skipped = _fetch_work_scan_file_texts_github(
            repo, blob_paths
        )

    return facts


def gather_repo_facts(
    repo: str,
    profile: str,
    profile_checks: dict[str, str],
    settings: dict[str, Any],
) -> RepoFacts:
    """Thin wrapper around the one backend this build has, kept (rather than
    calling gather_repo_facts_github directly from main()) so the
    skills-repo local-facts attachment below has one place to live instead
    of every caller remembering it."""
    facts = gather_repo_facts_github(repo, profile, profile_checks, settings)

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
        lines.append(f"{run.repo} [{run.profile}/{facts.platform}] {vis}")
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
                "platform": run.facts.platform if run.facts is not None else None,
                "error": run.error,
                "checks": [check_payload(run, c) for c in run.checks],
                "must_failed": sum(1 for c in run.checks if c.tier == "MUST" and c.status == "FAIL"),
                "should_failed": sum(1 for c in run.checks if c.tier == "SHOULD" and c.status == "FAIL"),
            }
            for run in results
        ],
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

    # Resolve each target's profile before doing any I/O. Config-shaped
    # problems (bad checklist reference, unknown profile, an explicit
    # --profile the checklist doesn't have) are genuinely global - they mean
    # this invocation is malformed, not that one repo had a bad day - so they
    # still abort the whole run, exactly as before this was split into its
    # own pass.
    resolved_targets: list[tuple[str, str]] = []
    for repo in targets:
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
        resolved_targets.append((repo, profile_name))

    if resolved_targets:
        try:
            check_gh_auth()
        except GhError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2

    # Enumeration reconciliation only makes sense on a full sweep (targeting
    # one repo already names it explicitly, so there's nothing to reconcile),
    # only when an overlay actually supplied assignments to reconcile against,
    # and only when not explicitly skipped.
    enumeration: EnumerationResult | None = None
    if is_full_sweep and overlay_loaded and not args.no_enumerate:
        owners = owners_from_assignments(assignments)
        try:
            enumerated: list[dict[str, Any]] = []
            for owner in owners:
                enumerated.extend(list_owner_repos(owner))
        except GhError as e:
            print(f"error: repo enumeration failed: {e}", file=sys.stderr)
            return 2
        unassigned, informational = reconcile_enumeration(enumerated, assignments, excluded)
        enumeration = EnumerationResult(ran=True, unassigned=unassigned, informational=informational)

    results: list[RepoRun] = []
    for repo, profile_name in resolved_targets:
        profile_def = profiles[profile_name]
        try:
            facts = gather_repo_facts(repo, profile_name, profile_def["checks"], settings)
        except GhError as e:
            # Specific to this one repo (404, transient API failure, ...).
            # Recording it and moving on is what keeps a bad repo #3 of 11
            # from erasing the other 10.
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

    # -- readme_has_structure: same shared _section_status machinery as
    #    quickstart/stack/running_for_real/how_built above - heading-only,
    #    body prose does not count - plus its own defensive path: an older
    #    checklist.yaml without a `structure` entry in section_patterns must
    #    SKIP, never crash and never claim an unearned PASS. The pattern
    #    itself lives in checklist.yaml (Willian's to add, under `structure`)
    #    - this self-test supplies its own copy so the assertions hold
    #    whether or not that entry exists in the real file yet. -------------
    structure_pattern = r"(project structure|repo structure|directory structure|folder structure|layout)"
    settings_with_structure = {
        **settings,
        "section_patterns": {**settings["section_patterns"], "structure": structure_pattern},
    }
    settings_without_structure = {
        **settings,
        "section_patterns": {k: v for k, v in settings["section_patterns"].items() if k != "structure"},
    }
    structure_heading_readme = (
        "# Widget\n\n## Project structure\n\n```\nwidget/\n"
        "\u251c\u2500\u2500 src/          # the logic\n"
        "\u251c\u2500\u2500 tests/\n"
        "\u2514\u2500\u2500 README.md\n```\n\nWhere the logic lives.\n"
    )
    # A heading with a real section under it, but written as a flat list
    # instead of a tree. This is the exact shape three of Willian's repos had
    # on 2026-08-16: readable, but you cannot see the shape of the repo at a
    # glance, which is the whole point of the section.
    structure_list_not_tree_readme = (
        "# Widget\n\n## Project structure\n\n"
        "- `src/main.py` does the work\n- `tests/` covers it\n\nThat is all.\n"
    )
    # A fenced block that is not a tree either - prose in a code fence must
    # not buy a pass.
    structure_fence_without_paths_readme = (
        "# Widget\n\n## Project structure\n\n```\nit is organised sensibly\n```\n"
    )
    structure_prose_readme = (
        "# Widget\n\nA paragraph that describes the project structure informally, in passing.\n\n"
        "## Something Else\n\ntext\n"
    )
    structure_missing_readme = "# Widget\n\nJust a paragraph, no structure section at all.\n"
    expect(
        "readme_has_structure pass (real heading)",
        check_readme_has_structure(_base_facts(readme_text=structure_heading_readme), settings_with_structure)[0],
        "PASS",
    )
    expect(
        "readme_has_structure fail (heading, but a flat list instead of a tree)",
        check_readme_has_structure(_base_facts(readme_text=structure_list_not_tree_readme), settings_with_structure)[0],
        "FAIL",
    )
    expect(
        "readme_has_structure fail (fenced block with no paths in it)",
        check_readme_has_structure(_base_facts(readme_text=structure_fence_without_paths_readme), settings_with_structure)[0],
        "FAIL",
    )
    expect(
        "looks_like_a_tree accepts an indented path listing without box characters",
        str(looks_like_a_tree("```\napp/\n  main.py\ndocs/\n  x.md\ntests/\n```")),
        "True",
    )
    expect(
        "readme_has_structure fail (no matching heading at all)",
        check_readme_has_structure(_base_facts(readme_text=structure_missing_readme), settings_with_structure)[0],
        "FAIL",
    )
    expect(
        "readme_has_structure fail (word only in body prose, not a heading)",
        check_readme_has_structure(_base_facts(readme_text=structure_prose_readme), settings_with_structure)[0],
        "FAIL",
    )
    expect(
        "readme_has_structure SKIP when settings.section_patterns has no `structure` entry (older checklist.yaml)",
        check_readme_has_structure(_base_facts(readme_text=structure_heading_readme), settings_without_structure)[0],
        "SKIP",
    )
    # Every OTHER section check must be completely unaffected by
    # _section_status's new defensive branch - same fixtures, same expected
    # outcomes as they had before readme_has_structure existed.
    expect(
        "readme_has_quickstart still passes on its own fixture, unaffected by the structure addition",
        check_readme_has_quickstart(_base_facts(readme_text=heading_readme), settings)[0],
        "PASS",
    )
    expect(
        "readme_has_stack still evaluates normally (checklist.yaml's real `stack` pattern is unaffected)",
        check_readme_has_stack(_base_facts(readme_text=structure_missing_readme), settings)[0],
        "FAIL",
    )
    structure_fail_result = CheckResult("readme_has_structure", "SHOULD", "FAIL", "no heading matches /.../i")
    structure_suggestion = suggest_readme_has_structure(_base_facts(), structure_fail_result, settings_with_structure)
    expect(
        "suggestion for readme_has_structure: names the heading, a fenced tree of the top two levels, "
        "and generating it from the real layout",
        str(
            '## Project structure' in structure_suggestion
            and "fenced tree" in structure_suggestion.lower()
            and "top two levels" in structure_suggestion.lower()
            and "actual layout" in structure_suggestion.lower()
        ),
        "True",
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

    # -- resolve_link_target: the piece that makes docs_links_resolve safe -
    #    a link must resolve relative to the file that contains it, not the
    #    repo root, or every ../../ link in a real docs/ tree floods false
    #    failures. ------------------------------------------------------------
    expect(
        "resolve_link_target: ../../ from a nested docs file resolves to repo root",
        resolve_link_target("../../README.md", "docs/history"),
        "README.md",
    )
    expect(
        "resolve_link_target: base_dir '' (repo root) leaves a plain target alone",
        resolve_link_target("docs/guide.md", ""),
        "docs/guide.md",
    )
    expect(
        "resolve_link_target: sibling reference from one docs subdir to another",
        resolve_link_target("../archive/old.md", "docs/history"),
        "docs/archive/old.md",
    )

    # -- docs_links_resolve: same rule as readme_links_resolve, one level
    #    deeper - every tracked docs/**/*.md file's relative links, resolved
    #    relative to the file that contains them. --------------------------
    docs_tree = ["README.md", "docs/guide.md", "docs/history/foo.md", "docs/archive/old.md"]
    expect(
        "docs_links_resolve pass (good relative link)",
        check_docs_links_resolve(
            _base_facts(
                # docs/guide.md's directory is "docs" - archive/old.md is a
                # same-level sibling of guide.md within docs/, so no ".."
                # is needed to reach docs/archive/old.md from here.
                tree_paths=docs_tree,
                docs_file_texts={"docs/guide.md": "# Guide\n\nSee the [archive](archive/old.md) too.\n"},
            ),
            settings,
        )[0],
        "PASS",
    )
    docs_broken_result = check_docs_links_resolve(
        _base_facts(
            tree_paths=docs_tree,
            docs_file_texts={"docs/guide.md": "# Guide\n\nSee [gone](../removed.md) for more.\n"},
        ),
        settings,
    )
    expect("docs_links_resolve fail (broken target)", docs_broken_result[0], "FAIL")
    expect(
        "docs_links_resolve fail detail names both the file and the target",
        str("docs/guide.md" in docs_broken_result[1] and "removed.md" in docs_broken_result[1]),
        "True",
    )
    # The case the coordinator specifically warned about: a ../../ link in a
    # nested file must resolve against ITS directory, not the repo root, or
    # every genuine docs/history/*.md and docs/archive/*.md file would
    # falsely fail this check.
    expect(
        "docs_links_resolve pass (../../ from a nested file resolves correctly)",
        check_docs_links_resolve(
            _base_facts(
                tree_paths=docs_tree,
                docs_file_texts={"docs/history/foo.md": "# Foo\n\nBack to the [README](../../README.md).\n"},
            ),
            settings,
        )[0],
        "PASS",
    )
    # #anchor-only and external links are out of scope, same as the README
    # check - reusing find_broken_links is what guarantees this for free.
    expect(
        "docs_links_resolve pass (#anchor-only and external links ignored)",
        check_docs_links_resolve(
            _base_facts(
                tree_paths=docs_tree,
                docs_file_texts={
                    "docs/guide.md": "# Guide\n\nSee [section](#install) and [upstream](https://example.com).\n"
                },
            ),
            settings,
        )[0],
        "PASS",
    )
    # No docs/ directory (or none with any markdown in it) at all: SKIP, not
    # a silent PASS that would claim link hygiene nobody actually checked.
    expect(
        "docs_links_resolve SKIP (no docs/ directory)",
        check_docs_links_resolve(_base_facts(tree_paths=["README.md"], docs_file_texts={}), settings)[0],
        "SKIP",
    )
    # Facts never gathered at all (a real bug: the check ran but the
    # gathering step that populates docs_file_texts didn't) stays a loud
    # FAIL, same pattern as index_covers_skills/level_with_origin.
    expect(
        "docs_links_resolve FAIL (facts never gathered)",
        check_docs_links_resolve(_base_facts(tree_paths=["README.md"], docs_file_texts=None), settings)[0],
        "FAIL",
    )

    # -- suggestion: docs_links_resolve names the file and the target, and
    #    ignores the cap-note suffix rather than mangling it into a fake
    #    "broken link" entry. ------------------------------------------------
    docs_fail_result = CheckResult(
        "docs_links_resolve", "MUST", "FAIL", "docs/guide.md -> ../removed.md (resolves to removed.md)"
    )
    docs_suggestion = suggestion_for(_base_facts(), docs_fail_result, settings) or ""
    expect(
        "suggestion: docs_links_resolve names the file and the target",
        str("docs/guide.md" in docs_suggestion and "removed.md" in docs_suggestion),
        "True",
    )
    docs_fail_with_cap_note = CheckResult(
        "docs_links_resolve",
        "MUST",
        "FAIL",
        "docs/guide.md -> ../removed.md (resolves to removed.md) || NOTE: 3 docs file(s) not checked "
        "(over the 50-file cap): docs/z1.md, docs/z2.md, docs/z3.md",
    )
    docs_suggestion_with_note = suggestion_for(_base_facts(), docs_fail_with_cap_note, settings) or ""
    expect(
        "suggestion: docs_links_resolve ignores the cap-note suffix rather than treating it as a broken link",
        str("docs/z1.md" not in docs_suggestion_with_note and "docs/guide.md" in docs_suggestion_with_note),
        "True",
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

    # -- owners_from_assignments: unique, sorted owners derived from the
    #    overlay's assignments - the piece that keeps a full sweep from
    #    being structurally blind to a repo that was created and never
    #    added (see main()'s enumeration block). --------------------------------
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

    # ======================================================================
    # ======================================================================
    # no_work_identifiers. The pattern list is entirely fictitious here
    # ("AcmeCorpInternal") - this self-test must never contain a real work
    # identifier itself, for the same reason the check exists. Every FAIL
    # detail and every --suggest string below is asserted to be free of that
    # fictitious term too, which is the same guarantee that must hold for a
    # real employer/client name at runtime.
    # ======================================================================

    SENSITIVE_TERM = "AcmeCorpInternal"
    one_pattern_settings = {**settings, "work_identifier_patterns": [f"(?i){SENSITIVE_TERM.lower()}"]}

    # -- select_work_scan_paths: vendor dirs and binary extensions are
    #    excluded from selection entirely (not reported as "skipped" - they
    #    were never eligible), and the cap drops otherwise-eligible files and
    #    must be able to say so. -----------------------------------------------
    fetch, skipped = select_work_scan_paths(["vendor/lib.py", "src/a.py"])
    expect("select_work_scan_paths: vendor/ files are excluded from selection", str(fetch), "['src/a.py']")
    expect("select_work_scan_paths: vendor/ exclusion is not reported as a cap-skip", str(skipped), "[]")
    fetch, skipped = select_work_scan_paths(["logo.png", "archive.zip", "a.py"])
    expect(
        "select_work_scan_paths: known-binary extensions are excluded from selection",
        str(fetch),
        "['a.py']",
    )
    many_paths = [f"src/file_{i:04d}.py" for i in range(WORK_IDENTIFIER_MAX_FILES + 5)]
    fetch, skipped = select_work_scan_paths(many_paths)
    expect(
        "select_work_scan_paths: fetch list is capped at WORK_IDENTIFIER_MAX_FILES",
        str(len(fetch)),
        str(WORK_IDENTIFIER_MAX_FILES),
    )
    expect(
        "select_work_scan_paths: files past the cap are named as skipped, not silently dropped",
        str(len(skipped)),
        "5",
    )

    # -- check_no_work_identifiers: no patterns configured -> SKIP, never
    #    PASS. Uses the REAL checklist.yaml settings, unmodified - the actual
    #    shape of a public-checklist-only run (CI on the public repo, or any
    #    machine without the private overlay), since work_identifier_patterns
    #    must never be added to checklist.yaml. -------------------------------
    expect(
        "no_work_identifiers SKIP when no work_identifier_patterns are configured (the public-checklist shape)",
        check_no_work_identifiers(_base_facts(), settings)[0],
        "SKIP",
    )

    # -- a real hit: FAILs, and the report never contains the matched term,
    #    only the file name and the pattern's index. --------------------------
    hit_facts = _base_facts(
        work_scan_file_texts={"README.md": f"An internal note mentions {SENSITIVE_TERM} by name."}
    )
    hit_status, hit_detail = check_no_work_identifiers(hit_facts, one_pattern_settings)
    expect("no_work_identifiers FAILs on a real hit", hit_status, "FAIL")
    expect(
        "no_work_identifiers FAIL detail names the file and the pattern index",
        str("README.md" in hit_detail and "#0" in hit_detail),
        "True",
    )
    expect(
        "no_work_identifiers FAIL detail does NOT echo the matched term",
        str(SENSITIVE_TERM.lower() not in hit_detail.lower()),
        "True",
    )

    # -- a clean tree: PASSes, and genuinely evaluated (not SKIP). ------------
    clean_facts = _base_facts(work_scan_file_texts={"README.md": "Nothing sensitive in here at all."})
    expect(
        "no_work_identifiers PASSes on a clean tree",
        check_no_work_identifiers(clean_facts, one_pattern_settings)[0],
        "PASS",
    )

    # -- invalid regex: a clear FAIL naming the pattern INDEX, never a
    #    traceback and never the pattern's own (sensitive) text. -------------
    bad_regex_settings = {**settings, "work_identifier_patterns": [f"(?i){SENSITIVE_TERM.lower()}", "(unclosed"]}
    bad_status, bad_detail = check_no_work_identifiers(clean_facts, bad_regex_settings)
    expect("no_work_identifiers FAILs clearly on an invalid regex, not a traceback", bad_status, "FAIL")
    expect(
        "no_work_identifiers invalid-regex detail names the offending index",
        str("work_identifier_patterns[1]" in bad_detail),
        "True",
    )
    expect(
        "no_work_identifiers invalid-regex detail does not echo the bad pattern's own text",
        str("(unclosed" not in bad_detail),
        "True",
    )

    # -- the cap path: files were skipped, nothing found in what WAS scanned -
    #    must report inconclusive, never PASS (a partial scan is not a clean
    #    scan). ------------------------------------------------------------
    capped_clean_facts = _base_facts(
        work_scan_file_texts={"README.md": "clean"}, work_scan_files_skipped=["src/z.py"]
    )
    capped_status, capped_detail = check_no_work_identifiers(capped_clean_facts, one_pattern_settings)
    expect("no_work_identifiers: capped scan with no hits is FAIL (inconclusive), never PASS", capped_status, "FAIL")
    expect(
        "no_work_identifiers: capped-clean detail says inconclusive",
        str("inconclusive" in capped_detail),
        "True",
    )
    # A real hit AND a cap both present: the hit still fails it, and the NOTE
    # about the cap rides along rather than being lost.
    capped_hit_facts = _base_facts(
        work_scan_file_texts={"README.md": f"mentions {SENSITIVE_TERM}"}, work_scan_files_skipped=["src/z.py"]
    )
    capped_hit_status, capped_hit_detail = check_no_work_identifiers(capped_hit_facts, one_pattern_settings)
    expect("no_work_identifiers: a real hit under the cap is still FAIL", capped_hit_status, "FAIL")
    expect(
        "no_work_identifiers: a real hit's detail still carries the cap NOTE",
        str("README.md" in capped_hit_detail and "NOTE" in capped_hit_detail),
        "True",
    )
    expect(
        "no_work_identifiers: hit+cap detail still never echoes the matched term",
        str(SENSITIVE_TERM.lower() not in capped_hit_detail.lower()),
        "True",
    )

    # -- gathering-bug path: check enabled with valid patterns, but facts were
    #    never populated - loud FAIL, same precedent as docs_links_resolve. --
    expect(
        "no_work_identifiers FAILs loudly (not SKIP) when facts were never gathered at all",
        check_no_work_identifiers(_base_facts(work_scan_file_texts=None), one_pattern_settings)[0],
        "FAIL",
    )

    # -- suggest_no_work_identifiers / suggestion_for: every rendered remedy
    #    stays free of the matched term, and each detail shape gets sensible,
    #    distinct guidance. ---------------------------------------------------
    hit_suggestion = suggestion_for(hit_facts, CheckResult("no_work_identifiers", "MUST", "FAIL", hit_detail), settings) or ""
    expect(
        "suggestion for no_work_identifiers hit: names the file and mentions removing it from history",
        str("README.md" in hit_suggestion and "history" in hit_suggestion.lower()),
        "True",
    )
    expect(
        "suggestion for no_work_identifiers hit: never echoes the matched term",
        str(SENSITIVE_TERM.lower() not in hit_suggestion.lower()),
        "True",
    )
    invalid_regex_suggestion = suggestion_for(clean_facts, CheckResult("no_work_identifiers", "MUST", "FAIL", bad_detail), settings) or ""
    expect(
        "suggestion for no_work_identifiers invalid regex: points at the overlay pattern, not a file to edit",
        str("work_identifier_patterns" in invalid_regex_suggestion and "(unclosed" not in invalid_regex_suggestion),
        "True",
    )
    inconclusive_suggestion = suggestion_for(capped_clean_facts, CheckResult("no_work_identifiers", "MUST", "FAIL", capped_detail), settings) or ""
    expect(
        "suggestion for no_work_identifiers inconclusive: talks about the cap, not a file to fix",
        str("cap" in inconclusive_suggestion.lower()),
        "True",
    )

    # -- CHECKS/SUGGESTIONS registration: explicit, even though the generic
    #    "every check has a suggestion template" assertion earlier in this
    #    self-test already covers the pairing. ---------------------------------
    expect("no_work_identifiers is registered in CHECKS", str("no_work_identifiers" in CHECKS), "True")
    expect("no_work_identifiers is registered in SUGGESTIONS", str("no_work_identifiers" in SUGGESTIONS), "True")

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
