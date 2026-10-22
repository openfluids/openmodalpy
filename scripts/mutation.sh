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

# `mutmut results` prints one "    <mutant>: <status>" line per mutant. Count
# every status it can emit, not only the four interesting ones: a mutant that
# lands in "no tests" is a mutated line the suite never reaches, which is the
# signal this scoped run exists to find. The status strings contain spaces
# ("no tests", "not checked"); they are not underscored.
count_status() { grep -c ": $1\$" "$report_path" || true; }

killed=$(count_status killed)
survived=$(count_status survived)
timeout=$(count_status timeout)
suspicious=$(count_status suspicious)
no_tests=$(count_status "no tests")
skipped=$(count_status skipped)
not_checked=$(count_status "not checked")

echo "killed=${killed} survived=${survived} timeout=${timeout} suspicious=${suspicious}"
echo "no_tests=${no_tests} skipped=${skipped} not_checked=${not_checked}"
echo "full report: ${report_path}"
