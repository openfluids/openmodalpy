#!/usr/bin/env bash
# On-demand mutation-testing harness for the numerical core (mutmut 3, scope
# set in pyproject.toml [tool.mutmut]: core/decomposition.py and core/welch.py
# only). Never run from the per-push CI path; invoke by hand or via the
# `mutation` workflow's workflow_dispatch trigger.
#
# On a shared machine run it under the heavy-run lock from the caller, e.g.
#   timeout 2700 flock -w 900 ~/.heavy.lock scripts/mutation.sh
# (the script does not take the lock itself, so CI runners need nothing).
# Writes the full `mutmut results` report to $1 (default: mutation-report.txt in the
# current directory; gitignored), and prints the killed/survived/timeout/
# suspicious counts on stdout.
set -euo pipefail

report_path="${1:-mutation-report.txt}"
# mutmut forks one test process per mutant up to --max-children, which
# defaults to the CPU count (24 on the shared dev box). Cap it; a lock held
# by the caller only serialises against other lock holders, it does not limit
# parallelism.
max_children="${MUTMUT_MAX_CHILDREN:-4}"

uv run --group mutation mutmut run --max-children "${max_children}"

uv run --group mutation mutmut results --all true > "$report_path"

killed=$(grep -c ': killed$' "$report_path" || true)
survived=$(grep -c ': survived$' "$report_path" || true)
timeout=$(grep -c ': timeout$' "$report_path" || true)
suspicious=$(grep -c ': suspicious$' "$report_path" || true)

echo "killed=${killed} survived=${survived} timeout=${timeout} suspicious=${suspicious}"
echo "full report: ${report_path}"
