#!/usr/bin/env bash
# Refuse to report green while GitHub CI is red for the commit under review.
#
# Why this exists: the local gate used to run pytest/ruff/mypy locally on
# Linux only and never asked GitHub what CI thought, so main stayed red for
# over a week (runs 31075084440, 31076229770, 31748285337) while every check
# passed locally. This script is the process fix: call it as the last step of
# any gate or release pre-flight, after the local checks pass.
#
# Usage:
#   scripts/check_ci_status.sh [ref]
#
#   ref    Branch name or full commit SHA. Default: the current branch, with
#          fallback to main when the branch has no CI runs yet (the gate may
#          legitimately run before anything was pushed).
#
# Exit codes: 0 = latest CI run for the ref concluded success;
#             1 = refusal (red/failed/timed out run, no runs at all, or gh
#                 unavailable). The refusing run's URL is always printed.
#
# Notes:
#   - Filters to the ci.yml workflow ON PURPOSE. A bare `gh run list` mixes in
#     Dependabot Updates and other workflows, so one green Dependabot run can
#     mask a red CI run newer than it.
#   - Fails closed: an environment where gh cannot answer is not evidence of
#     green, it is no evidence at all.
#   - Uses gh's built-in --jq; no jq binary required on the host.
set -uo pipefail

WORKFLOW="ci.yml"
REF="${1:-}"

fail() {
    echo "CI GATE REFUSAL: $*" >&2
    exit 1
}

command -v gh >/dev/null 2>&1 || fail "gh CLI not found; cannot verify CI status"

if [[ "$REF" =~ ^[0-9a-f]{7,40}$ ]]; then
    sha_mode=1
else
    sha_mode=0
    if [[ -z "$REF" ]]; then
        REF="$(git rev-parse --abbrev-ref HEAD 2>/dev/null)" || REF=""
        [[ -n "$REF" ]] || fail "no ref given and none could be detected from git"
    fi
fi

fallback_note=""
if (( ! sha_mode )); then
    run_meta="$(gh run list --branch "$REF" --workflow "$WORKFLOW" --limit 1 \
        --json conclusion,headSha,url --jq '.[0] | [.conclusion, .url, .headSha] | @tsv' 2>&1)" \
        || fail "gh run list failed: $run_meta"
    if [[ -z "$run_meta" || "$run_meta" == "null" ]] && [[ "$REF" != "main" ]]; then
        fallback_note=" (no CI runs on '$REF' yet; falling back to main)"
        REF="main"
        run_meta="$(gh run list --branch "$REF" --workflow "$WORKFLOW" --limit 1 \
            --json conclusion,headSha,url --jq '.[0] | [.conclusion, .url, .headSha] | @tsv' 2>&1)" \
            || fail "gh run list failed: $run_meta"
    fi
else
    run_meta="$(gh run list --commit "$REF" --workflow "$WORKFLOW" --limit 1 \
        --json conclusion,headSha,url --jq '.[0] | [.conclusion, .url, .headSha] | @tsv' 2>&1)" \
        || fail "gh run list failed: $run_meta"
fi

[[ -n "$run_meta" && "$run_meta" != "null" ]] || \
    fail "no '$WORKFLOW' runs found for '$REF'$fallback_note — no CI evidence exists"

IFS=$'\t' read -r conclusion run_url run_head <<<"$run_meta"

if [[ "$conclusion" != "success" ]]; then
    fail "latest CI run for '$REF'$fallback_note concluded '$conclusion': $run_url"
fi

if (( ! sha_mode )); then
    local_head="$(git rev-parse HEAD 2>/dev/null || true)"
    if [[ -n "$local_head" && -n "$run_head" && "$local_head" != "$run_head" ]]; then
        echo "NOTE: CI success is for an older commit ($run_head), not local HEAD ($local_head)." >&2
    fi
fi

echo "CI GATE OK: '$WORKFLOW' on '$REF'$fallback_note concluded success: $run_url"
