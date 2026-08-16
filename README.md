# Repo Hygiene

I built this to answer one question about any repository: if a stranger lands on it cold, with no access to the machine or network it normally runs on, can they tell what it is, run it, and trust it. It is a small, read-only Python script plus one YAML checklist that check a repo against a minimum bar of project infrastructure, then get out of the way.

## Quick start

Requirements: Python 3.11+, [`gh`](https://cli.github.com/) authenticated (`gh auth login`), and `pyyaml`.

```bash
pip install pyyaml

# Audit a single repo by owner/name
python3 scripts/audit.py --repo octocat/hello-world --profile public-portfolio

# Audit the repo you are standing in, resolved from its git remote
python3 scripts/audit.py --here --profile public-portfolio

# Same thing, via make
make check

# Offline, no gh, no network: exercises every check against hand built fixtures
make self-test
```

Add `--json` to either form for a machine readable report instead of the table.

## How it works

`checklist.yaml` is the standard. `scripts/audit.py` executes it. The check implementations are Python, but everything that constitutes the bar lives in the checklist: which checks apply to a given profile, at what tier, and against which thresholds. So raising or relaxing the standard means editing YAML, never the script, and a second consumer of the checklist (a pre-publish hook, for instance) is held to exactly the same bar as CI without a copy of it existing anywhere.

Every check reports one of three tiers:

| Tier | Meaning |
|---|---|
| `MUST` | a failure is a real defect; the run exits non-zero and this blocks going public |
| `SHOULD` | worth fixing, reported in the output, does not fail the run |
| `OFF` | the check does not apply to this profile |

A repo is held to a named profile, never an inferred one, so a repo flipping from private to public requires a deliberate edit rather than a silent change of bar:

| Profile | For | Bar |
|---|---|---|
| `public-portfolio` | a repo published for strangers to read, clone and judge | full: license, README structure, working links, a runnable quickstart, CI |
| `private-work` | a repo you work in but never publish | lighter: description, a README that orients, working links, no tracked secrets |
| `skills-repo` | a directory of reusable, versioned building blocks | drift checks: does an index cover every block, is the tree level with origin |
| `vault` | a notes or knowledge vault, markdown all the way down | metadata and safety only: almost every documentation check is OFF, because a vault is supposed to look like that |
| `gitlab-work` | reserved, not implemented | the script refuses this profile with a clear message rather than silently applying the GitHub bar |

Assignments (which repo gets which profile) and any repo-specific exclusions are meant to live in a small private overlay file that gets deep-merged onto `checklist.yaml` at run time, so one person's private repo list never has to live in a public checklist. `--overlay PATH` points at one, `--no-overlay` ignores it. In CI there normally is no overlay at all, which is exactly why `--profile` is a required flag there: with no assignments to fall back on, the tool refuses to guess a bar for you.

## Coverage is checked too

A sweep does not only audit the repos it was told about. It enumerates the owner's repos from the API and reconciles them against the overlay, so a repo that was created and never added is reported as UNASSIGNED and fails the run. Without that, coverage would silently depend on somebody remembering to update a list, which is the same failure mode as having no standard at all.

An archived repo counts as accounted for only when it is in `excluded`, not merely assigned a profile: a profile implies ongoing work that an archived, read-only repo no longer has.

## Read-only, by design

The audit never writes. Not to the repo, not to GitHub settings, not to a file in the tree. It reads and reports, which is what makes it safe to point at a dozen repos in one run without a second thought.

Fixing is a deliberate, separate step: read the report, fix one thing, then re-run the audit and require it green again. "I fixed it" is not evidence, a clean re-run is. The tool never bulk-fixes anything either: write access to many repos at once is exactly the kind of shortcut that turns one bad assumption into a dozen broken repos.

## Suggesting fixes

Add `--suggest` and every failing or warning check prints the exact command, or the exact edit, that would fix it: a `gh repo edit` invocation, a `git rm --cached` plus a `.gitignore` line, the heading to add to the README, which file is missing. It only prints, it never runs anything, so the read-only guarantee above still holds: the tool tells you the fix, you decide whether and when to apply it.

```bash
python3 scripts/audit.py --repo octocat/hello-world --profile public-portfolio --suggest
```

`--suggest` works with `--json` too: each non-passing check gets a `suggestion` field in the output (`null` on a pass). Every check id in the tool has a suggestion template mapped to it in `scripts/audit.py`, and `make self-test` asserts that mapping is complete, so a new check cannot ship without a remedy attached to it.

## Running it for real (CI)

The short way is the composite action, pinned to a tag so a change here cannot
turn your repos red without you choosing it:

```yaml
- uses: willianpripp/repo-hygiene@v1
  with:
    profile: public-portfolio
```

That is the whole job. It sets up Python, installs the one dependency, runs the
auditor's own self-test offline first, then audits the calling repository. Add
`suggest: "true"` to have every finding print the exact command that fixes it.

### The long way, without the action



Copy [`WORKFLOW_TEMPLATE.yml`](WORKFLOW_TEMPLATE.yml) into `.github/workflows/hygiene.yml` in the repo you want checked. It checks out this repo alongside yours and runs:

```bash
python3 .repo-hygiene/scripts/audit.py --repo ${{ github.repository }} --profile public-portfolio
```

`GH_TOKEN` is set to `${{ github.token }}` on that step, which is enough for the preinstalled `gh` to authenticate read-only calls against your own repo. The workflow triggers on push, on pull request, on manual dispatch, and once a week: metadata checks like description or topics have no push event to react to, so a schedule is the only thing that ever catches a repo going stale on GitHub's side with zero commits.

This repository checks itself the same way, see [`.github/workflows/hygiene.yml`](.github/workflows/hygiene.yml): it runs `--self-test` first as its own step, offline, before ever calling out to `gh`.

## How I built this

I wrote this after a sweep of my own repos found several with no CI at all, meaning their test suites never ran on a push. The checklist's shape (README sections, a runnable quickstart artifact, a `.gitignore`, no tracked secrets) came from the repos of mine that already agreed on one, not from a generic open source template. I built it with Claude Code and extracted it into its own repo once four separate projects needed the same standard and copying the script four times stopped making sense.

## License

[MIT](LICENSE).
