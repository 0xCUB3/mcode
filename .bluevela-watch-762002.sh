#!/usr/bin/env bash
set -euo pipefail

JOB_ID="762002"
LOGIN="skula@login3.bluevela.rmf.ibm.com"
STATUS_FILE="/Users/skula/Documents/mcode/.bluevela-watch-762002.status"
DONE_FILE="/Users/skula/Documents/mcode/.bluevela-watch-762002.done"
LOG_FILE="/Users/skula/Documents/mcode/.bluevela-watch-762002.log"
POLL_S="120"

log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$1" | tee -a "$LOG_FILE"
}

snapshot() {
  {
    printf 'timestamp=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    ssh "$LOGIN" "bjobs $JOB_ID 2>&1 || true"
    printf '\n== shard logs ==\n'
    ssh "$LOGIN" 'for idx in 1 2 3 4 5 6 7; do f=/u/skula/mcode/results/logs/live-m25-verified-full-main-20260331-b15-shard-${idx}.log; if [ -f "$f" ]; then echo "==== shard ${idx} ===="; tail -n 12 "$f"; fi; done' 2>&1 || true
    printf '\n== shard db sizes ==\n'
    ssh "$LOGIN" 'ls -lh /u/skula/mcode/results/live-m25-verified-full-main-20260331-b15-shard-*.db 2>/dev/null || true' 2>&1 || true
  } > "$STATUS_FILE"
}

log "watcher starting for job $JOB_ID"
while true; do
  snapshot
  if ssh "$LOGIN" "bjobs $JOB_ID >/dev/null 2>&1"; then
    log "job $JOB_ID still present"
    sleep "$POLL_S"
    continue
  fi

  log "job $JOB_ID no longer present in bjobs, collecting final state"
  {
    printf 'finished_at=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    printf '\n== bhist ==\n'
    ssh "$LOGIN" "bhist $JOB_ID 2>&1 || true"
    printf '\n== shard logs tail ==\n'
    ssh "$LOGIN" 'for idx in 1 2 3 4 5 6 7; do f=/u/skula/mcode/results/logs/live-m25-verified-full-main-20260331-b15-shard-${idx}.log; if [ -f "$f" ]; then echo "==== shard ${idx} ===="; tail -n 40 "$f"; fi; done' 2>&1 || true
    printf '\n== shard db sizes ==\n'
    ssh "$LOGIN" 'ls -lh /u/skula/mcode/results/live-m25-verified-full-main-20260331-b15-shard-*.db 2>/dev/null || true' 2>&1 || true
  } > "$DONE_FILE"
  log "final state written to $DONE_FILE"
  if command -v osascript >/dev/null 2>&1; then
    osascript -e 'display notification "Blue Vela job 762002 finished. Check .bluevela-watch-762002.done" with title "mcode monitor"' >/dev/null 2>&1 || true
  fi
  exit 0
done
