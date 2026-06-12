"""Stage-0 preflight: print the three resolved roots and assert every
INPUT file exists for ALL 3 models. Errors with the missing ABSOLUTE path; never skips a
model. Run this first on any new machine.
"""

import sys

from nmscrc import paths


def check_inputs():
    roots = paths.resolved_roots()
    print("[paths] resolved roots:")
    for k, v in roots.items():
        print(f"    {k:13s} = {v}")

    required = []
    for m in paths.MODELS:
        required.append(paths.data_full(m))
        required.append(paths.data_full_hs(m))
    required.append(paths.METHODS_JSON)

    missing = [p for p in required if not p.exists()]
    print(f"\n[preflight] checking {len(required)} INPUT files for models {paths.MODELS} ...")
    for p in required:
        mark = "ok " if p.exists() else "MISSING"
        print(f"    [{mark}] {p.resolve()}")

    if missing:
        lines = "\n".join(f"  - {p.resolve()}" for p in missing)
        raise FileNotFoundError(
            f"{len(missing)} required INPUT file(s) missing (never skip a model):\n{lines}"
        )
    print(f"\n[preflight] OK — all {len(required)} INPUT files present for all {len(paths.MODELS)} models.")
    return True


if __name__ == "__main__":
    try:
        check_inputs()
    except FileNotFoundError as e:
        print(f"\n[preflight] FAILED:\n{e}", file=sys.stderr)
        sys.exit(1)
