"""Application service for hierarchical duplicate detection."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from photoheaven.adapters.integrity.perceptual_hasher import PerceptualHasher
from photoheaven.application.ports import MediaRepository
from photoheaven.domain.models import MediaFile, MediaType

logger = logging.getLogger(__name__)


class _UnionFind:
    """Simple union-find for grouping duplicate media ids."""

    def __init__(self, items: list[str]) -> None:
        self._parent: dict[str, str] = {item: item for item in items}

    def find(self, item: str) -> str:
        parent = self._parent[item]
        if parent != item:
            self._parent[item] = self.find(parent)
        return self._parent[item]

    def union(self, a: str, b: str) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return
        self._parent[root_b] = root_a


@dataclass
class DedupeResult:
    """Result of a duplicate-detection run."""

    groups_created: int = 0
    total_duplicates: int = 0
    hashes_computed: int = 0
    checksum_matches: int = 0
    perceptual_matches: int = 0


@dataclass
class DedupeProgress:
    """Live progress snapshot emitted during duplicate detection."""

    total_media: int = 0
    hashes_total: int = 0
    hashes_done: int = 0
    candidate_pairs_checked: int = 0
    groups_found: int = 0
    duplicate_files_found: int = 0


@dataclass
class _DedupeItem:
    """Lightweight item used during duplicate detection."""

    media: MediaFile
    perceptual_hash: str | None = None


class DedupeService:
    """Find duplicate photos/videos using metadata, checksum and pHash."""

    def __init__(
        self,
        repository: MediaRepository,
        perceptual_hasher: PerceptualHasher,
    ) -> None:
        self.repository = repository
        self.perceptual_hasher = perceptual_hasher

    def find_duplicates(
        self,
        *,
        reset: bool = False,
        max_distance: int = 5,
        progress_callback: Optional[Callable[[DedupeProgress], None]] = None,
    ) -> DedupeResult:
        """Scan the library and store duplicate groups.

        Matches are based only on strong evidence: identical checksums or
        similar perceptual hashes. Metadata alone (capture time, camera,
        size) is intentionally NOT used, because burst-mode photos and
        near-simultaneous shots share those properties but are different
        images.
        """
        if reset:
            self.repository.clear_duplicate_groups()

        result = DedupeResult()
        progress = DedupeProgress()
        items = self._load_items()
        if not items:
            return result

        progress.total_media = len(items)
        self._notify(progress_callback, progress)

        self._ensure_perceptual_hashes(items, result, progress, progress_callback)

        buckets = self._bucket_items(items)
        edges: list[tuple[str, str, str]] = []
        # Track best match level per media id for storage.
        best_level: dict[str, str] = {}

        for bucket in buckets.values():
            for i in range(len(bucket)):
                for j in range(i + 1, len(bucket)):
                    a = bucket[i]
                    b = bucket[j]
                    progress.candidate_pairs_checked += 1
                    level = self._match_level(a, b, max_distance)
                    if level is None:
                        self._notify(progress_callback, progress)
                        continue
                    edges.append((a.media.id, b.media.id, level))
                    for item, lvl in ((a, level), (b, level)):
                        if _level_rank(lvl) > _level_rank(best_level.get(item.media.id, "")):
                            best_level[item.media.id] = lvl
                    self._notify(progress_callback, progress)

        result.checksum_matches = sum(1 for _, _, lvl in edges if lvl == "checksum")
        result.perceptual_matches = sum(
            1 for _, _, lvl in edges if lvl == "perceptual"
        )

        groups = self._build_groups(items, edges)
        for group_items in groups:
            members = self._prepare_group_members(group_items, best_level)
            self.repository.save_duplicate_group(str(uuid.uuid4()), members)
            result.groups_created += 1
            result.total_duplicates += len(members) - 1
            progress.groups_found = result.groups_created
            progress.duplicate_files_found = result.total_duplicates
            self._notify(progress_callback, progress)

        return result

    @staticmethod
    def _notify(
        callback: Optional[Callable[[DedupeProgress], None]],
        progress: DedupeProgress,
    ) -> None:
        if callback is not None:
            callback(progress)

    def list_duplicate_groups(self) -> list[dict]:
        """Return stored duplicate groups ordered by quality.

        Each group's members are sorted so the primary/best file is first.
        """
        groups = self.repository.list_duplicate_groups()
        for group in groups:
            group["members"] = sorted(
                group["members"], key=self._member_sort_key, reverse=True
            )
        return sorted(groups, key=self._group_sort_key, reverse=True)

    def _member_sort_key(self, member: dict) -> tuple:
        """Sort key for a group member dict (higher = better)."""
        path = Path(member["path"])
        ext = path.suffix.lower()
        return (
            ext in {".heic", ".heif"},
            member["size_bytes"],
            member.get("is_primary", False),
            path.name,
        )

    def _load_items(self) -> list[_DedupeItem]:
        """Load all media files into lightweight dedupe items."""
        items: list[_DedupeItem] = []
        offset = 0
        limit = 1000
        while True:
            batch = self.repository.list_media(limit=limit, offset=offset)
            if not batch:
                break
            for media in batch:
                items.append(
                    _DedupeItem(media=media, perceptual_hash=media.perceptual_hash)
                )
            if len(batch) < limit:
                break
            offset += limit
        return items

    def _ensure_perceptual_hashes(
        self,
        items: list[_DedupeItem],
        result: DedupeResult,
        progress: DedupeProgress,
        progress_callback: Optional[Callable[[DedupeProgress], None]],
    ) -> None:
        """Compute and persist missing perceptual hashes for images."""
        image_items = [
            item
            for item in items
            if item.media.media_type == MediaType.IMAGE and not item.perceptual_hash
        ]
        progress.hashes_total = len(image_items)
        self._notify(progress_callback, progress)
        for item in image_items:
            path = Path(item.media.path)
            if not path.exists():
                progress.hashes_done += 1
                self._notify(progress_callback, progress)
                continue
            hash_value = self.perceptual_hasher.compute(path)
            if hash_value is not None:
                item.perceptual_hash = hash_value
                self.repository.update_media_perceptual_hash(
                    item.media.id, hash_value
                )
                result.hashes_computed += 1
            progress.hashes_done += 1
            self._notify(progress_callback, progress)

    def _bucket_items(
        self, items: list[_DedupeItem]
    ) -> dict[datetime | str, list[_DedupeItem]]:
        """Bucket items by capture datetime (second precision).

        Items without a capture datetime go into a ``"unknown"`` bucket where
        only checksum matches are considered.
        """
        buckets: dict[datetime | str, list[_DedupeItem]] = {}
        for item in items:
            dt = item.media.capture_datetime
            if dt is not None:
                key = dt.replace(microsecond=0)
            else:
                key = "unknown"
            buckets.setdefault(key, []).append(item)
        return buckets

    def _match_level(
        self, a: _DedupeItem, b: _DedupeItem, max_distance: int
    ) -> str | None:
        """Return the strongest duplicate match level, or None.

        Only checksums and perceptual hashes are considered. Metadata such as
        capture datetime and camera model is deliberately ignored, because it
        produces false positives for burst photos and rapid sequences.
        """
        if a.media.checksum == b.media.checksum:
            return "checksum"

        if (
            a.media.media_type == MediaType.IMAGE
            and b.media.media_type == MediaType.IMAGE
            and a.perceptual_hash
            and b.perceptual_hash
        ):
            try:
                distance = self.perceptual_hasher.distance(
                    a.perceptual_hash, b.perceptual_hash
                )
                if distance <= max_distance:
                    return "perceptual"
            except Exception:
                logger.warning(
                    "Could not compare perceptual hashes for %s and %s",
                    a.media.id,
                    b.media.id,
                )

        return None

    def _build_groups(
        self, items: list[_DedupeItem], edges: list[tuple[str, str, str]]
    ) -> list[list[_DedupeItem]]:
        """Use union-find to group media connected by duplicate edges."""
        item_map = {item.media.id: item for item in items}
        uf = _UnionFind(list(item_map.keys()))
        for a_id, b_id, _ in edges:
            uf.union(a_id, b_id)

        groups: dict[str, list[_DedupeItem]] = {}
        for item in items:
            root = uf.find(item.media.id)
            groups.setdefault(root, []).append(item)

        return [group for group in groups.values() if len(group) > 1]

    def _prepare_group_members(
        self, group_items: list[_DedupeItem], best_level: dict[str, str]
    ) -> list[dict]:
        """Sort group by quality and build member dicts."""
        sorted_items = sorted(group_items, key=self._quality_key)
        members: list[dict] = []
        for index, item in enumerate(sorted_items):
            members.append(
                {
                    "media_id": item.media.id,
                    "is_primary": index == 0,
                    "match_level": best_level.get(item.media.id, "metadata"),
                }
            )
        return members

    def _quality_key(self, item: _DedupeItem) -> tuple:
        """Sort key where the best item comes first."""
        path = Path(item.media.path)
        ext = path.suffix.lower()
        # HEIC/HEIF first.
        is_heic = ext in {".heic", ".heif"}
        dt = item.media.capture_datetime
        ts = -dt.timestamp() if dt is not None else 0.0
        return (
            not is_heic,
            -item.media.size_bytes,
            ts,
            len(path.name),
            path.name,
        )

    def _group_sort_key(self, group: dict) -> tuple:
        """Sort groups by primary item quality (largest/best first)."""
        primary = next(
            (m for m in group.get("members", []) if m.get("is_primary")),
            None,
        )
        if primary is None:
            return (True, 0, "")
        path = Path(primary["path"])
        ext = path.suffix.lower()
        return (ext not in {".heic", ".heif"}, primary["size_bytes"], path.name)


def _level_rank(level: str) -> int:
    """Higher rank = stronger evidence."""
    return {"perceptual": 2, "checksum": 3}.get(level, 0)
