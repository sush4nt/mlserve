#!/usr/bin/env bash
# Runs one Makefile "stage" with a visible banner, live terminal output, AND a
# persisted per-stage log file — so when `make pipeline`/`make stack-up` fails
# three commands deep, it's immediately obvious *which* stage failed and where
# to find its full output, instead of scrolling back through one giant log.
#
# Usage: scripts/run_stage.sh <stage-name> <command> [args...]
# Example: scripts/run_stage.sh prepare uv run mlserve-prepare --dataset all
#
# Every stage's output is captured in full (stdout+stderr interleaved, exactly
# as printed) to logs/make/<stage-name>.log, overwritten on each run. Colored
# banners are best-effort: only used on a real terminal, and always safe in
# case output is being piped/redirected (e.g. `make pipeline > run.log 2>&1`).

set -uo pipefail

stage="${1:?usage: run_stage.sh <stage-name> <command...>}"
shift
cmd=("$@")

log_dir="${MLSERVE_LOGS_DIR:-logs}/make"
mkdir -p "$log_dir"
log_file="$log_dir/${stage}.log"

if [ -t 1 ]; then
  c_cyan=$'\033[1;36m'; c_green=$'\033[1;32m'; c_red=$'\033[1;31m'; c_reset=$'\033[0m'
else
  c_cyan=""; c_green=""; c_red=""; c_reset=""
fi

now() { date "+%H:%M:%S"; }

printf '\n%s==> [%s] STAGE: %s%s\n' "$c_cyan" "$(now)" "$stage" "$c_reset"
printf '$ %s\n' "${cmd[*]}" | tee "$log_file"

start_ts=$(date +%s)
# `tee` captures the full transcript while still streaming live to the
# terminal; PIPESTATUS[0] recovers the *command's* exit code (not tee's),
# which is what actually needs to fail the Makefile.
"${cmd[@]}" 2>&1 | tee -a "$log_file"
status=${PIPESTATUS[0]}
elapsed=$(( $(date +%s) - start_ts ))

if [ "$status" -eq 0 ]; then
  printf '%s<== [%s] STAGE OK: %s (%ss)%s\n' "$c_green" "$(now)" "$stage" "$elapsed" "$c_reset"
else
  printf '%s<== [%s] STAGE FAILED: %s (exit %s, %ss) -- see %s%s\n' \
    "$c_red" "$(now)" "$stage" "$status" "$elapsed" "$log_file" "$c_reset"
fi
exit "$status"
