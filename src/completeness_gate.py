# Copyright 2026 Mehmet Turan Yardimci
#
# Licensed under the Apache License, Version 2.0. You may obtain a copy of the License in the LICENSE file at the
# root of this repository or at http://www.apache.org/licenses/LICENSE-2.0
"""Refuse to produce a number from an evaluation that has not finished.

An evaluation writes one record per episode as it goes, so a partially finished run looks exactly like a
finished one to anything that simply reads the file. The failure this guards against is specific and was met in
practice: a reader checked that every task had appeared at least once, which in a design that cycles through the
task list is satisfied almost immediately, and produced a confident number from a small fraction of the intended
episodes.

The only criterion that works is the total count, derived from the configuration rather than written down as a
literal at each call site. A reader that hardcodes its own expectation drifts from the launcher the moment either
changes, and the drift is invisible because both still produce a number.

The gate answers with an object rather than a boolean so callers can print what is missing. Incomplete readings
return no value at all, instead of a value with a warning attached, because a warning next to a number does not
survive being copied into a table.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence


class EvalReading:
    """One evaluation file, read with its completeness known.

    Duplicate records are tolerated and reported rather than counted twice. A resumed evaluation can rewrite a
    batch it already finished, and counting raw lines would let the same episode contribute to a rate more than
    once. Within one cycle, episodes are keyed by their task and batch, which is unique by construction.

    Records are grouped by their cycle label first. A training run that evaluates periodically appends every
    evaluation to the same file, so a reader keyed on task and batch alone would let a later cycle overwrite an
    earlier one and still call the file complete, reporting a rate assembled from whichever weights happened to
    write last. When a file holds more than one cycle the caller must say which one it means; otherwise the
    reading refuses, because picking one silently is the failure this class exists to prevent.
    """

    def __init__(self, path: str, tasks: Sequence[int], episodes_per_task: int,
                 cycle: Optional[int] = None):
        self.path = path
        self.tasks = list(tasks)
        self.episodes_per_task = int(episodes_per_task)
        self.expected = len(self.tasks) * self.episodes_per_task
        self.exists = os.path.isfile(path)
        self.requested_cycle = cycle
        self._groups: Dict[object, Dict[int, Dict[int, dict]]] = {}
        self._rows: Dict[object, int] = {}

        if self.exists:
            with open(path, "r", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    idx = rec.get("task_idx")
                    if idx is None or idx >= len(self.tasks):
                        continue
                    task = self.tasks[idx]
                    key = rec.get("cycle")
                    self._groups.setdefault(key, {}).setdefault(task, {})[rec.get("batch")] = rec
                    self._rows[key] = self._rows.get(key, 0) + 1

        self.selected: Optional[object] = None
        self.ambiguous = False
        if cycle is not None:
            self.selected = cycle
        elif len(self._groups) == 1:
            self.selected = next(iter(self._groups))
        elif len(self._groups) > 1:
            self.ambiguous = True

    @property
    def cycles(self) -> List[object]:
        """Cycle labels present in the file, in the order they were first seen."""
        return list(self._groups.keys())

    @property
    def _by_key(self) -> Dict[int, Dict[int, dict]]:
        if self.ambiguous:
            return {}
        return self._groups.get(self.selected, {})

    @property
    def rows(self) -> int:
        if self.ambiguous:
            return sum(self._rows.values())
        return self._rows.get(self.selected, 0)

    @property
    def episodes(self) -> int:
        return sum(len(v) for v in self._by_key.values())

    @property
    def duplicates(self) -> int:
        return self.rows - self.episodes

    @property
    def complete(self) -> bool:
        # An expectation of zero is not an evaluation that finished, it is one that was never asked for, and
        # every other clause below is satisfied by whatever happens to be in the file already. Without this the
        # gate certifies a previous run's records under this run's name.
        return (self.exists
                and self.expected > 0
                and not self.ambiguous
                and len(self._by_key) == len(self.tasks)
                and self.episodes >= self.expected)

    @property
    def status(self) -> str:
        if not self.exists:
            return "MISSING"
        if self.ambiguous:
            return "AMBIGUOUS %d cycles present, none requested" % len(self._groups)
        if not self.complete:
            return "PARTIAL %d/%d" % (self.episodes, self.expected)
        if self.duplicates:
            return "COMPLETE (+%d duplicate rows ignored)" % self.duplicates
        return "COMPLETE"

    def success_rate(self, task: Optional[int] = None, budget: Optional[int] = None) -> Optional[float]:
        """Success rate over one task or over all of them, or None when the reading is incomplete.

        The budget argument counts an episode as a success only if it finished within that many steps, which is
        what makes a single evaluation readable at several horizons without running it again.
        """
        if not self.complete:
            return None
        tasks = [task] if task is not None else self.tasks
        rates = []
        for t in tasks:
            recs = list(self._by_key.get(t, {}).values())
            if not recs:
                return None
            hits = 0
            for r in recs:
                step = int(r.get("succ_step", -1))
                if step > 0 and (budget is None or step <= budget):
                    hits += 1
            rates.append(hits / len(recs))
        return sum(rates) / len(rates)


def gate(readings: Dict[str, EvalReading]) -> bool:
    """Print a completeness ledger and say whether any number may be reported at all.

    Returns True only when every reading is complete. A caller scoring more than one checkpoint should stop on
    False rather than report whichever readings happen to be ready, since a partial reading is not a smaller
    version of a complete one.
    """
    print("completeness")
    print("-" * 60)
    ok = True
    for name, r in readings.items():
        flag = "ok " if r.complete else "NO "
        print("  %s %-28s %s" % (flag, name[:28], r.status))
        ok &= r.complete
    print()
    if not ok:
        print("Incomplete readings present. No number is produced.")
    return ok
