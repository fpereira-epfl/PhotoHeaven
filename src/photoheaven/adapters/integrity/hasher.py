"""Blake3-based file hasher."""

from __future__ import annotations

from pathlib import Path

import blake3

from photoheaven.application.ports import Hasher


class Blake3Hasher(Hasher):
    """Compute a Blake3 checksum for a file."""

    def hash_file(self, path: Path) -> str:
        hasher = blake3.blake3()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
