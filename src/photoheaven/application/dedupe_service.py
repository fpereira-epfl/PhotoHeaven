"""Application service for hierarchical duplicate detection."""

from __future__ import annotations

import logging
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from photoheaven.adapters.integrity.hasher import Blake3Hasher
from photoheaven.adapters.integrity.perceptual_hasher import PerceptualHasher
from photoheaven.adapters.integrity.video_frame_hasher import VideoFrameHasher
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
class DedupeMoveResult:
    """Result of moving duplicate files to the duplicates folder."""

    groups_processed: int = 0
    files_moved: int = 0
    files_missing: int = 0
    files_reconciled: int = 0
    groups_with_errors: int = 0


@dataclass
class DedupeMoveProgress:
    """Live progress snapshot emitted during duplicate moving."""

    total_groups: int = 0
    groups_done: int = 0
    files_moved: int = 0
    files_missing: int = 0


@dataclass
class DedupeProgress:
    """Live progress snapshot emitted during duplicate detection."""

    total_media: int = 0
    hashes_total: int = 0
    hashes_done: int = 0
    candidate_pairs_total: int = 0
    candidate_pairs_checked: int = 0
    groups_found: int = 0
    duplicate_files_found: int = 0


@dataclass
class _DedupeItem:
    """Lightweight item used during duplicate detection."""

    media: MediaFile
    perceptual_hash: str | None = None
    video_frame_hashes: list[str] | None = None


class DedupeService:
    """Find duplicate photos/videos using metadata, checksum and pHash."""

    def __init__(
        self,
        repository: MediaRepository,
        perceptual_hasher: PerceptualHasher,
        video_frame_hasher: VideoFrameHasher | None = None,
        hasher: Blake3Hasher | None = None,
    ) -> None:
        self.repository = repository
        self.perceptual_hasher = perceptual_hasher
        self.video_frame_hasher = video_frame_hasher
        self.hasher = hasher or Blake3Hasher()

    def find_duplicates(
        self,
        *,
        reset: bool = False,
        max_distance: int = 5,
        include_videos: bool = False,
        progress_callback: Optional[Callable[[DedupeProgress], None]] = None,
    ) -> DedupeResult:
        """Scan the library and store duplicate groups.

        Matches are based only on strong evidence: identical checksums or
        similar perceptual hashes. Metadata alone (capture time, camera,
        size) is intentionally NOT used, because burst-mode photos and
        near-simultaneous shots share those properties but are different
        images.

        Duplicate groups are replaced on every run so stale groups from
        previous scans do not accumulate.

        Videos are excluded unless ``include_videos`` is True. When included,
        video duplicates are detected by identical checksums or similar pHash
        values computed from sampled keyframes.
        """
        self.repository.clear_duplicate_groups()

        result = DedupeResult()
        progress = DedupeProgress()
        items = self._load_items(include_videos=include_videos)
        if not items:
            return result

        progress.total_media = len(items)
        self._notify(progress_callback, progress)

        self._ensure_perceptual_hashes(
            items, result, progress, progress_callback, include_videos=include_videos
        )

        buckets = self._bucket_items(items)
        progress.candidate_pairs_total = sum(
            len(bucket) * (len(bucket) - 1) // 2 for bucket in buckets.values()
        )
        # If there were no hashes to compute, the hash task should appear done.
        progress.hashes_done = progress.hashes_total
        self._notify(progress_callback, progress)

        edges: list[tuple[str, str, str]] = []
        # Track best match level per media id for storage.
        best_level: dict[str, str] = {}

        notify_interval = max(1, progress.candidate_pairs_total // 100)
        for bucket in buckets.values():
            for i in range(len(bucket)):
                for j in range(i + 1, len(bucket)):
                    a = bucket[i]
                    b = bucket[j]
                    progress.candidate_pairs_checked += 1
                    level = self._match_level(a, b, max_distance)
                    if level is None:
                        if progress.candidate_pairs_checked % notify_interval == 0:
                            self._notify(progress_callback, progress)
                        continue
                    edges.append((a.media.id, b.media.id, level))
                    for item, lvl in ((a, level), (b, level)):
                        if _level_rank(lvl) > _level_rank(best_level.get(item.media.id, "")):
                            best_level[item.media.id] = lvl
                    if progress.candidate_pairs_checked % notify_interval == 0:
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

        # Ensure the UI shows 100% even if work finished between notify ticks.
        progress.hashes_done = progress.hashes_total
        progress.candidate_pairs_checked = progress.candidate_pairs_total
        self._notify(progress_callback, progress)

        return result

    @staticmethod
    def _notify(
        callback: Optional[Callable[..., None]],
        progress: DedupeProgress | DedupeMoveProgress,
    ) -> None:
        if callback is not None:
            callback(progress)

    def list_duplicate_groups(
        self, *, only_faces: bool = False, only_videos: bool = False
    ) -> list[dict]:
        """Return stored duplicate groups ordered by quality.

        Each group's members are sorted so the primary/best file is first.
        When ``only_faces`` is True, only groups where at least one member has
        a detected face are returned. When ``only_videos`` is True, only groups
        where at least one member is a video are returned.
        """
        media_with_faces: set[str] | None = None
        if only_faces:
            media_with_faces = self.repository.get_media_ids_with_faces()

        groups = self.repository.list_duplicate_groups()
        filtered_groups: list[dict] = []
        for group in groups:
            group["members"] = sorted(
                group["members"], key=self._member_sort_key, reverse=True
            )
            if only_faces:
                member_ids = {m["media_id"] for m in group["members"]}
                if not member_ids.intersection(media_with_faces):
                    continue
            if only_videos:
                if not any(
                    m.get("media_type") == MediaType.VIDEO.value
                    for m in group["members"]
                ):
                    continue
            filtered_groups.append(group)

        return sorted(filtered_groups, key=self._group_sort_key, reverse=True)

    def move_duplicates(
        self,
        duplicates_root: Path,
        *,
        dry_run: bool = False,
        progress_callback: Optional[Callable[[DedupeMoveProgress], None]] = None,
    ) -> DedupeMoveResult:
        """Move non-primary duplicate files into ``duplicates_root/YYYY/MM``.

        The database is updated to reflect the new paths, so ``--list`` and
        future scans continue to treat the moved files as duplicates of the
        primary. The primary file is never moved. Renamed duplicates from
        earlier runs are reconciled by checksum after moving.
        """
        result = DedupeMoveResult()
        groups = self.list_duplicate_groups()
        if not groups:
            return result

        progress = DedupeMoveProgress(total_groups=len(groups))
        self._notify(progress_callback, progress)

        for group in groups:
            members = group["members"]
            if not members:
                progress.groups_done += 1
                self._notify(progress_callback, progress)
                continue
            primary = members[0]
            primary_path = Path(primary["path"])
            # Derive the YYYY/MM folder from the primary's current location.
            year, month = self._resolve_year_month(primary_path)
            group_target_dir = duplicates_root / year / month

            group_ok = True
            for member in members[1:]:
                source = Path(member["path"])
                target = self._unique_target_path(
                    group_target_dir, source.name, primary_path.name
                )
                if dry_run:
                    result.files_moved += 1
                    progress.files_moved += 1
                    continue

                try:
                    group_target_dir.mkdir(parents=True, exist_ok=True)
                    if not source.exists():
                        if target.exists():
                            # Already moved in a previous/interrupted run;
                            # just reconcile the database path.
                            self.repository.update_media_path(
                                member["media_id"], str(target)
                            )
                        else:
                            result.files_missing += 1
                            progress.files_missing += 1
                            group_ok = False
                        continue

                    if source.resolve() == target.resolve():
                        # Source and target are the same file; nothing to do.
                        continue

                    if source.parent.resolve() == group_target_dir.resolve():
                        # File is already in the correct duplicates folder,
                        # likely under a different name chosen in a previous
                        # run. Reconcile the path and leave it alone.
                        self.repository.update_media_path(
                            member["media_id"], str(source)
                        )
                        continue

                    shutil.move(str(source), str(target))
                    self.repository.update_media_path(
                        member["media_id"], str(target)
                    )
                except Exception as exc:
                    logger.error(
                        "Failed to move duplicate %s to %s: %s",
                        source,
                        target,
                        exc,
                    )
                    group_ok = False
                    continue
                result.files_moved += 1
                progress.files_moved += 1
            result.groups_processed += 1
            progress.groups_done += 1
            if not group_ok:
                result.groups_with_errors += 1
            self._notify(progress_callback, progress)

        # Reconcile any members whose recorded path no longer exists but whose
        # file is still present in the target folder under a different name
        # (legacy damage from earlier runs that renamed duplicates).
        result.files_reconciled = self._reconcile_renamed_duplicates(
            duplicates_root
        )

        return result

    def _reconcile_renamed_duplicates(self, duplicates_root: Path) -> int:
        """Update DB paths for duplicates that were renamed in place.

        Returns the number of paths reconciled.
        """
        reconciled = 0
        checksum_cache: dict[Path, str] = {}

        for group in self.list_duplicate_groups():
            for member in group["members"]:
                if member.get("is_primary"):
                    continue
                source = Path(member["path"])
                if source.exists():
                    continue

                primary = group["members"][0]
                primary_path = Path(primary["path"])
                year, month = self._resolve_year_month(primary_path)
                group_target_dir = duplicates_root / year / month
                if not group_target_dir.exists():
                    continue

                expected_checksum = member.get("checksum")
                if not expected_checksum:
                    continue

                for candidate in group_target_dir.iterdir():
                    if not candidate.is_file():
                        continue
                    if candidate.name == primary_path.name:
                        continue
                    if candidate not in checksum_cache:
                        try:
                            checksum_cache[candidate] = self.hasher.hash_file(
                                candidate
                            )
                        except Exception:
                            continue
                    if checksum_cache[candidate] != expected_checksum:
                        continue

                    # Make sure the candidate path is not already claimed by
                    # another media record.
                    existing_id = self.repository.get_media_id_by_path(
                        str(candidate)
                    )
                    if existing_id is not None and existing_id != member["media_id"]:
                        continue

                    self.repository.update_media_path(
                        member["media_id"], str(candidate)
                    )
                    reconciled += 1
                    break

        return reconciled

    @staticmethod
    def _resolve_year_month(path: Path) -> tuple[str, str]:
        """Return (YYYY, MM) from the nearest parent folder names if possible."""
        parts = path.parts
        for i in range(len(parts) - 2, -1, -1):
            if len(parts[i]) == 4 and parts[i].isdigit() and len(parts[i + 1]) == 2 and parts[i + 1].isdigit():
                return parts[i], parts[i + 1]
        # Fallback to the file's current modification/capture date would require
        # another DB lookup; use filesystem mtime as a reasonable approximation.
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        return f"{mtime.year:04d}", f"{mtime.month:02d}"

    @staticmethod
    def _unique_target_path(directory: Path, original_name: str, primary_name: str) -> Path:
        """Return a unique target path that avoids clashing with the primary."""
        candidate = directory / original_name
        if candidate.name.lower() != primary_name.lower() and not candidate.exists():
            return candidate

        stem = Path(original_name).stem
        suffix = Path(original_name).suffix
        n = 1
        while True:
            candidate = directory / f"{stem}_{n}{suffix}"
            if candidate.name.lower() != primary_name.lower() and not candidate.exists():
                return candidate
            n += 1

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

    def _load_items(self, *, include_videos: bool = False) -> list[_DedupeItem]:
        """Load all media files into lightweight dedupe items."""
        items: list[_DedupeItem] = []
        offset = 0
        limit = 1000
        while True:
            batch = self.repository.list_media(limit=limit, offset=offset)
            if not batch:
                break
            for media in batch:
                if media.media_type == MediaType.VIDEO and not include_videos:
                    continue
                items.append(
                    _DedupeItem(
                        media=media,
                        perceptual_hash=media.perceptual_hash,
                        video_frame_hashes=media.video_frame_hashes,
                    )
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
        *,
        include_videos: bool = False,
    ) -> None:
        """Compute and persist missing perceptual hashes for images and videos."""
        pending_items = [
            item
            for item in items
            if (
                item.media.media_type == MediaType.IMAGE
                and not item.perceptual_hash
            )
            or (
                include_videos
                and item.media.media_type == MediaType.VIDEO
                and not item.video_frame_hashes
            )
        ]
        progress.hashes_total = len(pending_items)
        self._notify(progress_callback, progress)
        for item in pending_items:
            path = Path(item.media.path)
            if not path.exists():
                progress.hashes_done += 1
                self._notify(progress_callback, progress)
                continue

            if item.media.media_type == MediaType.IMAGE:
                hash_value = self.perceptual_hasher.compute(path)
                if hash_value is not None:
                    item.perceptual_hash = hash_value
                    self.repository.update_media_perceptual_hash(
                        item.media.id, hash_value
                    )
                    result.hashes_computed += 1
            elif include_videos and item.media.media_type == MediaType.VIDEO:
                frame_hashes = self._compute_video_frame_hashes(path)
                if frame_hashes is not None:
                    item.video_frame_hashes = frame_hashes
                    self.repository.update_media_video_frame_hashes(
                        item.media.id, frame_hashes
                    )
                    result.hashes_computed += 1
            progress.hashes_done += 1
            self._notify(progress_callback, progress)

    def _compute_video_frame_hashes(self, path: Path) -> list[str] | None:
        """Compute keyframe pHashes for a video file."""
        if self.video_frame_hasher is None:
            return None
        return self.video_frame_hasher.compute(path)

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

        if (
            a.media.media_type == MediaType.VIDEO
            and b.media.media_type == MediaType.VIDEO
            and a.video_frame_hashes
            and b.video_frame_hashes
        ):
            try:
                if self._video_frames_match(
                    a.video_frame_hashes, b.video_frame_hashes, max_distance
                ):
                    return "perceptual"
            except Exception:
                logger.warning(
                    "Could not compare video frame hashes for %s and %s",
                    a.media.id,
                    b.media.id,
                )

        return None

    def _video_frames_match(
        self,
        hashes_a: list[str],
        hashes_b: list[str],
        max_distance: int,
    ) -> bool:
        """Return True if any sampled keyframe pair is similar enough."""
        if self.video_frame_hasher is None:
            return False
        for ha in hashes_a:
            for hb in hashes_b:
                try:
                    if self.video_frame_hasher.distance(ha, hb) <= max_distance:
                        return True
                except Exception:
                    continue
        return False

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
