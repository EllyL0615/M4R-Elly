"""Freeze + sha256 of Stage-0 artifacts. Downstream verifies the hash on
startup: a physical guarantee that experiments read frozen artifacts and never silently drift.
"""

import hashlib
import json
from pathlib import Path


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def write_artifact_hash(files, out_path, base):
    """Hash every file in `files`; write 'sha256  relpath-from-base' lines + a combined digest.
    `base` is the common root (DATA_ROOT) so relpaths resolve on any machine.

    Returns the combined digest (sha256 of the per-file lines), used as each json's split_hash.
    """
    out_path, base = Path(out_path), Path(base)
    lines = []
    for p in sorted(Path(f) for f in files):
        lines.append(f"{sha256_file(p)}  {Path(p).relative_to(base).as_posix()}")
    body = "\n".join(lines) + "\n"
    combined = hashlib.sha256(body.encode("utf-8")).hexdigest()
    out_path.write_text(f"# combined={combined}\n{body}", encoding="utf-8")
    return combined


def read_combined_hash(hash_path):
    """Read the combined digest written by write_artifact_hash (the '# combined=' header)."""
    first = Path(hash_path).read_text(encoding="utf-8").splitlines()[0]
    if not first.startswith("# combined="):
        raise RuntimeError(f"{hash_path} missing combined-hash header")
    return first.split("=", 1)[1].strip()


def check_hash(hash_path, base):
    """Recompute every listed file's sha256 and assert it matches ARTIFACT_HASH.txt. Returns the
    combined digest. Raises on any mismatch / missing file."""
    hash_path, base = Path(hash_path), Path(base)
    lines = hash_path.read_text(encoding="utf-8").splitlines()
    mismatches = []
    for line in lines:
        if line.startswith("#") or not line.strip():
            continue
        want, rel = line.split("  ", 1)
        p = base / rel
        if not p.exists():
            mismatches.append(f"MISSING {rel}")
        elif sha256_file(p) != want:
            mismatches.append(f"CHANGED {rel}")
    if mismatches:
        raise RuntimeError("artifact hash check FAILED:\n  " + "\n  ".join(mismatches))
    return read_combined_hash(hash_path)
