#!/bin/bash
# ANNA v2 watchdog — ensures `python3 -m anna_v2 resume` is always running.
# If the orchestrator exits (normal target_reached / max_iter / crash), wait
# then restart. Also auto-resets status if it ended in a stuck state.
#
# Safety features:
#   - flock-based single-instance guard (prevents duplicate watchdogs)
#   - crash-loop backoff using data/.restart_log
#   - hard giveup after >20 restarts in last 24h (data/.watchdog_giveup)
#
# Run with:   nohup bash watchdog.sh > watchdog.log 2>&1 &  disown

set -u
cd /home/jaeseokhan/2025-02/khu/research-2026-01

ANNA_DIR=/home/jaeseokhan/2025-02/khu/research-2026-01/anna_v2
DATA_DIR="$ANNA_DIR/data"
STATE_FILE="$DATA_DIR/loop_state.json"
LOCK_FILE="$DATA_DIR/.watchdog.lock"
RESTART_LOG="$DATA_DIR/.restart_log"
GIVEUP_FILE="$DATA_DIR/.watchdog_giveup"
LOG="$ANNA_DIR/anna_resume_$(date +%Y%m%d).log"

mkdir -p "$DATA_DIR"

# --- Single-instance guard via flock -----------------------------------------
# Open fd 200 against the lock file; non-blocking acquire. If another watchdog
# already holds it, exit silently so cron / manual nohup invocations don't
# stack up.
exec 200>"$LOCK_FILE"
flock -n 200 || exit 0

# --- Crash-loop backoff helpers ----------------------------------------------
# Restart log is one Unix epoch timestamp per line. We append before each
# launch attempt, then count occurrences in the last 1h / 24h windows.

record_restart() {
  date +%s >> "$RESTART_LOG"
  # Cap file at last 1000 lines to avoid unbounded growth.
  if [ -f "$RESTART_LOG" ]; then
    tail -n 1000 "$RESTART_LOG" > "$RESTART_LOG.tmp" && mv "$RESTART_LOG.tmp" "$RESTART_LOG"
  fi
}

count_restarts_within() {
  # $1 = window in seconds
  local window=$1
  local now
  now=$(date +%s)
  local cutoff=$((now - window))
  if [ ! -f "$RESTART_LOG" ]; then
    echo 0
    return
  fi
  awk -v c="$cutoff" '$1 >= c { n++ } END { print n+0 }' "$RESTART_LOG"
}

compute_backoff_seconds() {
  # Exponential backoff once last-hour restarts exceed 5.
  # 6th restart -> 60s, 7th -> 120s, 8th -> 240s, ... capped at 1h.
  local hr_count=$1
  if [ "$hr_count" -le 5 ]; then
    echo 60
    return
  fi
  local over=$((hr_count - 5))
  local backoff=$((60 * (1 << (over - 1))))
  if [ "$backoff" -gt 3600 ]; then
    backoff=3600
  fi
  echo "$backoff"
}

# Reset stuck state to allow resume. When watchdog is about to launch ANNA,
# any non-(awaiting_review|idle) value is stale and should be reset; in
# particular, status='running' left by a SIGTERM'd orchestrator would make
# resume() print 'Cannot resume' and exit immediately, causing an infinite
# respawn loop.
reset_state_if_stuck() {
  python3 - <<'PY'
import json, sys
sf = '/home/jaeseokhan/2025-02/khu/research-2026-01/anna_v2/data/loop_state.json'
try:
    with open(sf) as f: s = json.load(f)
except Exception:
    sys.exit(0)
allowed = {"awaiting_review", "idle"}
if s.get("status") not in allowed:
    print(f"[watchdog] resetting state from '{s.get('status')}' -> 'awaiting_review'")
    s["status"] = "awaiting_review"
    with open(sf, "w") as f:
        json.dump(s, f, indent=2, ensure_ascii=False)
PY
}

# --- Main loop ---------------------------------------------------------------
while true; do
  # Hard giveup if a previous iteration tripped the 24h ceiling.
  if [ -f "$GIVEUP_FILE" ]; then
    echo "[watchdog] giveup file present ($GIVEUP_FILE) — exiting at $(date -Is)"
    exit 0
  fi

  # Only launch if no resume/run is running (avoid duplicates).
  # Pattern covers both `resume` and `run` subcommands, and is anchored to
  # `python.* -m anna_v2` so unrelated python processes don't match.
  if pgrep -f "python.* -m anna_v2 (resume|run)" >/dev/null; then
    sleep 120
    continue
  fi

  # Check loop_state — if target_reached, stop the watchdog.
  STATUS=$(python3 -c "import json; print(json.load(open('$STATE_FILE'))['status'])" 2>/dev/null || echo "unknown")
  if [ "$STATUS" = "target_reached" ]; then
    echo "[watchdog] target_reached — stopping watchdog at $(date -Is)"
    exit 0
  fi

  # Crash-loop guardrails.
  DAY_COUNT=$(count_restarts_within 86400)
  if [ "$DAY_COUNT" -gt 20 ]; then
    echo "[watchdog] >20 restarts in last 24h ($DAY_COUNT) — writing giveup and exiting at $(date -Is)"
    {
      echo "giveup_at=$(date -Is)"
      echo "restarts_24h=$DAY_COUNT"
    } > "$GIVEUP_FILE"
    exit 0
  fi

  HR_COUNT=$(count_restarts_within 3600)
  BACKOFF=$(compute_backoff_seconds "$HR_COUNT")

  reset_state_if_stuck

  echo "[watchdog] starting ANNA resume at $(date -Is)  (status was: $STATUS, restarts: 1h=$HR_COUNT 24h=$DAY_COUNT, next backoff=${BACKOFF}s)"
  record_restart
  PYTHONUNBUFFERED=1 python3 -m anna_v2 resume >> "$LOG" 2>&1
  EC=$?
  echo "[watchdog] ANNA exited (code=$EC) at $(date -Is). Waiting ${BACKOFF}s before next launch."
  sleep "$BACKOFF"
done
