# Copyright 2026 Mehmet Turan Yardimci
#
# Licensed under the Apache License, Version 2.0. See the LICENSE file at the root of this repository.
"""Walk what every entry point imports, and fail if it reaches something this repository does not contain.

A repository can look complete and still not be. Every module is present, every gate passes, and the whole thing
runs perfectly on the machine it was written on, because one import resolves to a file that lives beside the
checkout and was never committed. Nobody else can run it, and the file listing gives no hint of that.

The walk is static: modules are parsed, not imported. Importing would need the dependencies installed and would
run module level code, and a gate that only works on a fully provisioned machine is a gate that stops being run.

"Inside this repository" means tracked by git, not merely present on disk, whenever the tree is a checkout. The
distinction is the entire point: an uncommitted module beside the checkout satisfies a walk of the working tree
and is absent from every clone. When the tree is not a checkout the walk falls back to the filesystem and says so,
rather than reporting a stronger result than it can support.

Three outcomes per imported name. It resolves to a file inside this repository, and the walk continues into it.
It is part of the standard library. Or it is a third party package, in which case it must be named in
requirements.txt, since an undeclared dependency fails for a new user in exactly the same way a missing file
does. Anything else is a defect and fails the gate.

This file walks itself along with everything else. A checker that skips its own directory certifies itself.

Usage: python scripts/check_import_closure.py [root]
Exit code 1 if any import cannot be accounted for.
"""

import ast
import os
import subprocess
import sys

# Import names that differ from the distribution name that provides them. Kept short and explicit rather than
# resolved through package metadata, which would need the packages installed and defeat the point of a static walk.
IMPORT_ALIASES = {
    "PIL": "pillow",
    "yaml": "pyyaml",
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "hydra": "hydra-core",
    "libero": "hf-libero",
}


def tracked_files(root):
    """The set of paths git would place in a fresh clone, or None when this is not a checkout.

    Asking git rather than the filesystem is the whole point of the gate. A module that exists beside the checkout
    and was never committed satisfies a filesystem walk and is absent from everyone else's clone, which is exactly
    the failure this file was written to catch, and a walk of the working tree cannot see it.
    """
    try:
        out = subprocess.run(["git", "-C", root, "ls-files"], capture_output=True, text=True, timeout=60)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return {os.path.normpath(line.strip()) for line in out.stdout.splitlines() if line.strip()}


def repo_modules(root, tracked):
    """Map every importable module in the tree to its path, by the names an entry point would use.

    When the tree is a git checkout, only tracked files count, so an uncommitted module cannot satisfy an import.
    """
    found = {}
    for sub in ("src", "scripts", "tests", "."):
        d = os.path.join(root, sub)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".py"):
                continue
            rel = os.path.normpath(fn if sub == "." else os.path.join(sub, fn))
            if tracked is not None and rel not in tracked:
                continue
            found.setdefault(fn[:-3], os.path.join(d, fn))
    return found


def declared_requirements(root):
    names = set()
    path = os.path.join(root, "requirements.txt")
    if not os.path.isfile(path):
        return names
    for line in open(path, "r", encoding="utf-8", errors="replace"):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        for sep in ("==", ">=", "<=", "~=", ">", "<", "["):
            line = line.split(sep, 1)[0]
        names.add(line.strip().lower().replace("_", "-"))
    return names


def imported_names(path):
    """Top level names imported by one module, split into relative and absolute.

    The split matters for the verdict. A relative import can only ever mean a module of this package, so one that
    does not resolve here is a file that is missing from the repository. An absolute import may legitimately mean
    a third party package, and is judged against the declared dependencies instead.
    """
    relative, absolute = set(), set()
    try:
        tree = ast.parse(open(path, "r", encoding="utf-8", errors="replace").read(), filename=path)
    except SyntaxError:
        return relative, absolute
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                absolute.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                (relative if node.level else absolute).add(node.module.split(".")[0])
            elif node.level:
                # `from . import a, b`: there is no module attribute, and each imported name is itself a sibling
                # module. Skipping this form leaves the walk blind to a whole class of local import.
                for alias in node.names:
                    relative.add(alias.name.split(".")[0])
    return relative, absolute


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tracked = tracked_files(root)
    local = repo_modules(root, tracked)
    required = declared_requirements(root)
    # Without this the gate cannot tell the standard library from a third party package, and the degraded answer
    # is not a weaker pass but a page of untrue findings: every import of os and json becomes undeclared. Refused
    # rather than degraded, because a gate that reports nonsense is worse than one that says it cannot run.
    if not hasattr(sys, "stdlib_module_names"):
        print("this gate needs Python 3.10 or newer, which is where sys.stdlib_module_names arrived; without it "
              "the standard library cannot be recognised and every import of it would be reported as undeclared")
        return 1
    stdlib = set(sys.stdlib_module_names)

    entry_points = sorted(p for p in local.values()
                          if os.path.basename(os.path.dirname(p)) in ("src", "scripts", "tests"))

    seen, missing, undeclared, external = set(), [], [], set()
    queue = list(entry_points)
    while queue:
        path = queue.pop()
        if path in seen:
            continue
        seen.add(path)
        relative, absolute = imported_names(path)

        for name in sorted(relative):
            if name in local:
                queue.append(local[name])
            else:
                missing.append("%s imports .%s relatively, and no such module is in this repository"
                               % (os.path.relpath(path, root), name))

        for name in sorted(absolute):
            if name in local:
                queue.append(local[name])
            elif name in stdlib or name == "__future__":
                continue
            else:
                dist = IMPORT_ALIASES.get(name, name).lower().replace("_", "-")
                external.add(name)
                if dist not in required:
                    undeclared.append("%s imports %s, which is neither in this repository, nor in the standard "
                                      "library, nor provided by any line of requirements.txt"
                                      % (os.path.relpath(path, root), name))

    scope = ("what git tracks, so an uncommitted module counts as missing" if tracked is not None
             else "the working tree, since this is not a git checkout, so an uncommitted module would pass")
    print("import closure over %d modules from %d entry points" % (len(seen), len(entry_points)))
    print("  resolved against %s" % scope)
    print("  external packages reached: %s" % ", ".join(sorted(external)))
    for m in sorted(set(missing)):
        print("  MISSING     %s" % m)
    for u in sorted(set(undeclared)):
        print("  UNACCOUNTED %s" % u)

    if missing or undeclared:
        print("FAIL: %d unresolved local import(s), %d unaccounted import(s)"
              % (len(set(missing)), len(set(undeclared))))
        return 1
    print("PASS: every import resolves to this repository, the standard library, or a declared dependency.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
