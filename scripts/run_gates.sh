#!/bin/bash
# Copyright 2026 Mehmet Turan Yardimci
# Licensed under the Apache License, Version 2.0. You may obtain a copy of the License in the LICENSE file at the
# root of this repository or at http://www.apache.org/licenses/LICENSE-2.0
#
# Run the mechanical verification gates and report one table.
#
# A gate that only runs when someone remembers its name is a file, not a gate. This is the single entry point,
# and it is what continuous integration or a scheduled job should call.
#
# One rule worth stating, because it is easy to get wrong and does not fail loudly: a check must not exempt
# itself from its own walk, or it will certify itself.
#
# Exit code is the number of failing gates, so zero means everything passed.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Some distributions ship only python3, and this is the one command the README's verification section gives.
PY="${GATE_PY:-}"
[ -n "$PY" ] || { command -v python >/dev/null 2>&1 && PY=python || PY=python3; }
cd "$ROOT" || exit 99

# Write no bytecode while checking. Importing a module normally leaves a cache beside it, so a suite that imports
# the tree in order to test it also fills that tree with compiled copies. A cache outlives the source it was
# built from, and the two can then disagree with nothing reporting it. The gate that verifies a tree must not be
# the thing that puts unreviewable content in it.
export PYTHONDONTWRITEBYTECODE=1

fails=0
results=""

run_gate () {
  local label="$1"; shift
  local out rc
  out=$("$@" 2>&1); rc=$?
  if [ $rc -eq 0 ]; then
    results="${results}  ok    ${label}\n"
  else
    results="${results}  FAIL  ${label} (exit ${rc})\n"
    fails=$((fails+1))
    echo "----- ${label} -----"
    echo "$out" | tail -20
  fi
}

echo "verification gates"
echo

run_gate "syntax, and no compiled artefact" $PY scripts/check_syntax.py
run_gate "import closure is committed"     $PY scripts/check_import_closure.py

# The invariants import the trainer, which reaches the simulator package, which asks an interactive question the
# first time it is imported and writes the answer to a config file. Under this runner the child's output is
# captured, so that prompt is invisible: on a terminal the gates would sit there forever showing nothing, and
# with no terminal they fail on an end of file naming the simulator rather than the cause. Refused with the one
# line remedy instead, which is the same step Getting started asks for before anything else.
if [ ! -f "${LIBERO_CONFIG_PATH:-$HOME/.libero}/config.yaml" ]; then
  results="${results}  SKIP  invariants and gate self tests (simulator not configured yet)\n"
  fails=$((fails+1))
  echo "----- invariants and gate self tests -----"
  echo "the simulator asks a one time question on first import and has not been answered on this machine."
  echo "run this once, interactively, then run the gates again:"
  echo "  python -c \"import libero.libero\""
else
  run_gate "invariants and gate self tests"  $PY tests/test_invariants.py
fi

echo
printf "%b" "$results"
echo
if [ $fails -eq 0 ]; then echo "ALL_GATES_PASS"; else echo "GATES_FAILED=${fails}"; fi
exit $fails
