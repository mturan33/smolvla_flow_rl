#!/usr/bin/env bash
# Copyright 2026 Mehmet Turan Yardimci
#
# Licensed under the Apache License, Version 2.0. You may obtain a copy of the License in the LICENSE file at the
# root of this repository or at http://www.apache.org/licenses/LICENSE-2.0
# Helpers for the canonical evaluation scan: completeness by SHAPE, and never scanning into a dead attempt.
#
# WHY BOTH OF THESE EXIST, and neither is theoretical.
#
# A canonical scan writes one episode per line. If the scan is interrupted and relaunched, and the writer
# APPENDS rather than truncates, the file ends up holding two scans: the dead attempt's episodes followed by
# the live one's. Every completeness check that counts LINES then calls that file complete, twice over, first
# when the mixture crosses the expected episode count and again when the second scan finishes. A success rate
# read from it double counts part of the evaluation and mixes two runs of the same weights.
#
#   1. `canon_archive_previous` moves a previous attempt aside before a scan starts. It is MOVED, never
#      deleted: the record of what happened is worth keeping.
#   2. `canon_episodes` asks whether the file has the SHAPE of a canonical scan, N batches of K tasks with
#      each (batch, task) pair appearing exactly once, and reports 0 when it does not. A file that is not one
#      scan is not a partial scan either; it is not a measurement.
#
# Line count is not completeness. That sentence is the whole point of this file.

CANON_BATCHES=${CANON_BATCHES:-30}
CANON_TASKS=${CANON_TASKS:-7}

# Print the episode count of a CLEAN canonical scan in $1, or 0 with a reason on stderr.
canon_episodes () {
  local d=$1
  CANON_BATCHES="$CANON_BATCHES" CANON_TASKS="$CANON_TASKS" python3 - "$d" <<'PYEOF'
import collections, io, json, os, sys
d = sys.argv[1]
NB = int(os.environ.get("CANON_BATCHES", 30))
NT = int(os.environ.get("CANON_TASKS", 7))
want = NB * NT
for rel in ("canon/events.jsonl", "canon_events.jsonl"):
    p = os.path.join(d, rel)
    if not os.path.exists(p):
        continue
    try:
        rows = [json.loads(l) for l in io.open(p) if l.strip()]
    except Exception as e:
        print("unreadable %s: %s" % (rel, e), file=sys.stderr)
        continue
    if not rows:
        continue
    pairs = collections.Counter((r.get("batch"), r.get("task_idx")) for r in rows)
    dup = sum(1 for v in pairs.values() if v > 1)
    batches = collections.Counter(r.get("batch") for r in rows)
    if dup:
        print("%s holds more than one scan: %d duplicated (batch, task) pairs" % (rel, dup), file=sys.stderr)
        continue
    if len(rows) != want or sorted(batches) != list(range(NB)) or set(batches.values()) != {NT}:
        print("%s is %d episodes over batches %s, not a complete %dx%d scan"
              % (rel, len(rows), sorted(batches)[:4], NB, NT), file=sys.stderr)
        continue
    print(len(rows))
    sys.exit(0)
print(0)
PYEOF
}

# Move any previous canonical output aside so a scan never writes into a dead one's file.
canon_archive_previous () {
  local d=$1 stamp
  stamp=$(date +%Y%m%d_%H%M%S)
  for p in "$d/canon/events.jsonl" "$d/canon_events.jsonl"; do
    if [ -s "$p" ]; then
      mv "$p" "${p%.jsonl}.attempt_$stamp.jsonl"
      echo "  archived a previous attempt: $(basename "${p%.jsonl}.attempt_$stamp.jsonl")"
    fi
  done
}
