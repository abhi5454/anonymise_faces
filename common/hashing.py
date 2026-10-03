"""Hashing, atomic writes, and content addressing."""

import hashlib
import json
import os
import tempfile

CHUNK = 1024 * 1024


def sha256_file(path):
    """Stream a file and return its hex sha256 digest."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_rename(src, dst):
    """Atomically publish a finished artifact.

    The parent directory is created if needed. ``os.replace`` is atomic on
    POSIX when src and dst live on the same filesystem, so retries overwrite
    with identical bytes instead of creating duplicates.
    """
    parent = os.path.dirname(os.path.abspath(dst))
    os.makedirs(parent, exist_ok=True)
    os.replace(src, dst)
    return dst


def content_addressed_path(base_dir, digest, suffix):
    """Return ``<base_dir>/<digest><suffix>`` for content-addressed outputs."""
    if not suffix.startswith("."):
        suffix = "." + suffix
    return os.path.join(base_dir, digest + suffix)


def write_json_atomic(path, payload):
    """Write JSON to ``path`` atomically (temp file in the same dir -> rename).

    Used for the side-car artifacts the eval harness reads (tracks, verifier
    flags): a reader never sees a half-written file, and a retry overwrites with
    identical bytes instead of appending duplicates.
    """
    path = os.path.abspath(path)
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=parent, prefix=".tmp_",
                                   suffix=os.path.basename(path))
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            json.dump(payload, out, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    return path
