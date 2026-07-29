# Copyright 2026 Mehmet Turan Yardimci
#
# Licensed under the Apache License, Version 2.0. See the LICENSE file at the root of this repository.
"""Parse every module without producing bytecode, and fail if any compiled artefact is present.

Both halves address the same hazard. A syntax gate that invokes the byte compiler writes caches inside the tree
it is checking, so the gate that verifies a repository also fills it with content nobody will read. A cache
outlives the source it came from, which means an edited file and its cache can disagree while nothing reports a
difference, and the stale one is the one that gets imported.

Parsing rather than compiling gives the same syntax guarantee and writes nothing.

Line endings are checked in the same pass. A shell script saved with carriage returns fails on the first line
under a Unix shell, with an error that names an option rather than the real cause, so it is a defect that lands on
every user and points nowhere useful. It is easy to introduce by editing from another platform and invisible in a
diff, which is exactly the combination that warrants a gate.

Which files count as text is decided by looking at their bytes, not by their extension. An extension allowlist
covers the spellings whoever wrote it thought of, silently skips the rest, and then reports that the whole tree is
consistent. That is the same shape as the defect the gate exists to catch, one level up.

Usage: python scripts/check_syntax.py [root]
Exit code 1 on a syntax error, a carriage return, or any compiled artefact found.
"""

import ast
import os
import sys

ARTEFACT_EXT = {".pyc", ".pyo", ".pyd"}
# Directories this gate is not about. Two kinds: places a reader's own tooling writes, and places a run writes.
# Without this the gate reports a virtual environment created inside the clone, which .gitignore anticipates, as
# compiled artefacts of this repository, and syntax checks every dependency along with it. It also reports a
# carriage return in a file a run produced as a defect in the tree.
#
# Not driven by .gitignore, deliberately. That file ignores __pycache__ and *.pyc, which are exactly what this
# gate exists to find, so honouring it would switch the gate off.
SKIP_DIRS = {".git", ".venv", "venv", "env", "site-packages", "node_modules", "runs", "outputs"}

# Enough bytes to classify a file. A null byte in the first block is the usual signal that content is binary and
# that carriage returns in it mean nothing.
SNIFF_BYTES = 8192


def looks_like_text(path):
    try:
        with open(path, "rb") as fh:
            head = fh.read(SNIFF_BYTES)
    except OSError:
        return False
    return b"\x00" not in head


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    errors, artefacts, crlf, parsed = [], [], [], 0

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        if os.path.basename(dirpath) == "__pycache__":
            artefacts.append(os.path.relpath(dirpath, root))
            continue
        for fn in sorted(filenames):
            path = os.path.join(dirpath, fn)
            ext = os.path.splitext(fn)[1].lower()
            if ext in ARTEFACT_EXT:
                artefacts.append(os.path.relpath(path, root))
                continue
            if looks_like_text(path):
                try:
                    if b"\r\n" in open(path, "rb").read():
                        crlf.append(os.path.relpath(path, root))
                except OSError:
                    pass
            if ext != ".py":
                continue
            try:
                ast.parse(open(path, "r", encoding="utf-8", errors="replace").read(), filename=path)
                parsed += 1
            except SyntaxError as exc:
                errors.append("%s:%s %s" % (os.path.relpath(path, root), exc.lineno, exc.msg))

    print("parsed %d modules, wrote no bytecode" % parsed)
    for e in errors:
        print("  SYNTAX    %s" % e)
    for a in artefacts:
        print("  ARTEFACT  %s  (compiled content cannot be reviewed by reading)" % a)
    for c in crlf:
        print("  LINE END  %s  (carriage returns break shell scripts on a Unix host)" % c)

    if errors or artefacts or crlf:
        print("FAIL: %d syntax error(s), %d compiled artefact(s), %d file(s) with carriage returns"
              % (len(errors), len(artefacts), len(crlf)))
        return 1
    print("PASS: every module parses, the tree holds no compiled artefact, and line endings are consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
