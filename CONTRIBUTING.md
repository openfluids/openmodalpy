# Contributing to openmodalpy

Contributions are genuinely welcome, and that includes the ones that are not
code. A bug report, a confusing docstring, a README paragraph that turned out to
be wrong, a question that took you an hour to answer yourself — all of those are
worth opening an [issue](https://github.com/openfluids/openmodalpy/issues) for.

If you are unsure whether something is worth reporting, it probably is. Open the
issue.

## Getting set up

```bash
git clone https://github.com/openfluids/openmodalpy.git
cd openmodalpy
uv sync
```

## Before you open a pull request

The same checks CI runs:

```bash
uv run --group test pytest -q
uv run --group lint ruff check .
uv run --group lint ruff format --check .
uv lock --check
# Last step, after the local checks pass: refuse green while CI is red.
scripts/check_ci_status.sh
```

CI also enforces a coverage floor. The number lives in `pyproject.toml`
(`[tool.coverage.report] fail_under`), so the same gate runs locally:

```bash
uv run --group test pytest -q --cov=openmodalpy
```

### Coverage floor ratchet

The floor is a ratchet, not a target: it sits a couple of points under the
measured coverage and moves up only when someone has measured again and
decided to move it. Concretely:

- Raise it when actual coverage exceeds the floor comfortably (say, by 3
  points or more) after your change. Measure with the command above, set
  `fail_under` just under what you measured, and say so in the pull request.
- Nobody lowers the floor to land a change. If your change drops coverage
  below the floor, the fix is tests for the new code — not a smaller number.
- The floor is measured on Linux; treat the exact percentage as
  platform-specific and keep the margin in the pyproject comment honest.
- The floor is aggregate-only, deliberately: coverage has no native
  per-module gate, so one would mean custom scripting in CI. The numerical
  core modules sit well above the aggregate (welch 100%, threads 98%,
  decomposition 94%, pod 86%); io.py and commands.py drag it down, and both
  are glue where the marginal test is worth less than one on the numerics.
  Revisit per-module floors if the aggregate ever reaches 85%.

If one fails for a reason you think is unrelated to your change, say so in the
pull request rather than working around it — that is useful information, and
sometimes it is CI that is wrong.

## What makes a pull request easy to review

- **One thing at a time.** A focused change gets reviewed quickly. A change that
  also reformats fifty unrelated lines is hard to read and slow to merge.
- **Say what you verified.** A pasted command and its output is worth more than
  "tested locally".
- **Ask early.** For anything substantial, open an issue first. It is much
  better to disagree about an approach before you have written it than after.
- **Draft PRs are fine.** Opening one early to ask "is this the right
  direction?" is welcome and costs nothing.

Reviews may take a few days — one maintainer, research alongside. A nudge on a
quiet pull request is welcome, not annoying.

## Conventions

Only the ones that are actually enforced:

- Decomposition results must stay reproducible: seed explicitly with
  `np.random.default_rng(seed)` rather than the legacy module-level
  `np.random` calls.
- Anything touching the parallel path needs a test that runs serially too. A
  result that depends on worker count is a bug.
- Resolve paths relative to `__file__`. No hardcoded absolute paths.
- Formatting and import order are handled by `ruff` — do not hand-tune them.
- New user-facing behaviour gets a `CHANGELOG.md` entry.

## Conduct and licence

Everyone taking part is asked to follow the
[openfluids Code of Conduct](https://github.com/openfluids/.github/blob/main/CODE_OF_CONDUCT.md).
It is short.

openmodalpy is licensed under Apache-2.0, and contributions are accepted under
the same licence. See `LICENSE` and `NOTICE`.

Found a security problem? Please do not open a public issue — see the
[security policy](https://github.com/openfluids/openmodalpy/security/policy).
