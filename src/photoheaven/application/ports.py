"""Ports (interfaces) that the domain/application layer depends on.

Adapters live outside this layer and implement these interfaces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional

from photoheaven.domain.models import Face, GeoPoint, Identity, MediaFile, MediaType


class Hasher(ABC):
    """Computes a strong checksum for a file."""

    @abstractmethod
    def hash_file(self, path: Path) -> str:
        """Return a hex string checksum for the file at *path*."""
        raise NotImplementedError


class MediaMetadata:
    """Plain result object returned by MetadataExtractor.

    Mirrors optional fields of MediaFile so that the ingestion service can
    apply them without depending on a concrete extraction library.
    """

    def __init__(
        self,
        media_type: MediaType = MediaType.UNKNOWN,
        capture_datetime: Optional[datetime] = None,
        make: Optional[str] = None,
        model: Optional[str] = None,
        gps: Optional[GeoPoint] = None,
        extracted: bool = True,
    ) -> None:
        self.media_type = media_type
        self.capture_datetime = capture_datetime
        self.make = make
        self.model = model
        self.gps = gps
        self.extracted = extracted


class MetadataExtractor(ABC):
    """Extracts capture date, GPS, camera info and other metadata."""

    @abstractmethod
    def extract(self, path: Path, media_type: MediaType) -> MediaMetadata:
        """Return metadata for the given file."""
        raise NotImplementedError


class MediaRepository(ABC):
    """Persistence port for media files and faces."""

    @abstractmethod
    def get_by_checksum(self, checksum: str) -> Optional[MediaFile]:
        """Return the media file with the given checksum, if any."""
        raise NotImplementedError

    @abstractmethod
    def save_media(self, media: MediaFile) -> None:
        """Persist a media file."""
        raise NotImplementedError

    @abstractmethod
    def update_media_path(self, media_id: str, new_path: str) -> None:
        """Update the stored filesystem path for a media file."""
        raise NotImplementedError

    @abstractmethod
    def count_media(self) -> int:
        """Return the number of media files in the library."""
        raise NotImplementedError

    @abstractmethod
    def list_media(self, limit: int = 100, offset: int = 0) -> list[MediaFile]:
        """Return a paginated list of media files."""
        raise NotImplementedError

    @abstractmethod
    def get_all_media_paths(self) -> list[str]:
        """Return all stored media file paths.

        This is intended for path-analysis operations and may load a large
        result set into memory.
        """
        raise NotImplementedError

    @abstractmethod
    def save_face(self, face: Face) -> None:
        """Persist a detected face."""
        raise NotImplementedError

    @abstractmethod
    def count_faces(self) -> int:
        """Return the number of faces in the library."""
        raise NotImplementedError

    @abstractmethod
    def media_has_faces(self, media_id: str) -> bool:
        """Return True if at least one face record exists for the media file."""
        raise NotImplementedError

    @abstractmethod
    def get_media_ids_with_faces(self) -> set[str]:
        """Return all media ids that have at least one detected face."""
        raise NotImplementedError

    @abstractmethod
    def get_unprocessed_faces_media(
        self, limit: int = 100, offset: int = 0
    ) -> list[MediaFile]:
        """Return media files that have not yet had face analysis run."""
        raise NotImplementedError

    @abstractmethod
    def update_media_face_analysis(
        self, media_id: str, analyzed_at: datetime, version: str
    ) -> None:
        """Mark a media file as having been analysed for faces."""
        raise NotImplementedError

    @abstractmethod
    def list_faces_for_media(self, media_id: str) -> list[Face]:
        """Return all faces detected in the given media file."""
        raise NotImplementedError

    @abstractmethod
    def get_face_by_id(self, face_id: str) -> Face | None:
        """Return a single face by id, or None if not found."""
        raise NotImplementedError

    @abstractmethod
    def list_faces(self, limit: int = 100, offset: int = 0) -> list[Face]:
        """Return a paginated list of all faces."""
        raise NotImplementedError

    @abstractmethod
    def get_all_faces(self) -> list[Face]:
        """Return all faces with embeddings.

        This is intended for clustering and may load a large result set into
        memory. Callers should use ``list_faces`` for UI pagination.
        """
        raise NotImplementedError

    @abstractmethod
    def get_embedding_versions(self) -> set[str]:
        """Return the distinct embedding versions present in the library."""
        raise NotImplementedError

    @abstractmethod
    def update_face_cluster_label(
        self, face_id: str, cluster_label: int | None
    ) -> None:
        """Assign a cluster label to a face."""
        raise NotImplementedError

    @abstractmethod
    def update_face_identity_name_for_cluster(
        self, cluster_label: int, identity_name: str | None
    ) -> None:
        """Assign an identity name to all faces in a cluster."""
        raise NotImplementedError

    @abstractmethod
    def save_identity(self, identity: Identity) -> None:
        """Persist an identity (person)."""
        raise NotImplementedError

    @abstractmethod
    def get_identity_by_name(self, name: str) -> Identity | None:
        """Return an identity by its human-readable name."""
        raise NotImplementedError

    @abstractmethod
    def get_identity_by_id(self, identity_id: str) -> Identity | None:
        """Return an identity by its unique id."""
        raise NotImplementedError

    @abstractmethod
    def list_identities(self, limit: int = 100, offset: int = 0) -> list[Identity]:
        """Return a paginated list of identities."""
        raise NotImplementedError

    @abstractmethod
    def get_faces_for_cluster(self, cluster_label: int) -> list[Face]:
        """Return all faces belonging to a cluster."""
        raise NotImplementedError

    @abstractmethod
    def get_faces_for_identity(self, identity_id: str) -> list[Face]:
        """Return all faces linked to an identity."""
        raise NotImplementedError

    @abstractmethod
    def get_faces_without_identity(
        self, limit: int = 100, offset: int = 0
    ) -> list[Face]:
        """Return faces that are not yet linked to any identity."""
        raise NotImplementedError

    @abstractmethod
    def update_face_identity(
        self,
        face_id: str,
        *,
        identity_id: str | None,
        identity_name: str | None,
    ) -> None:
        """Assign (or clear) the persistent identity for a single face."""
        raise NotImplementedError

    @abstractmethod
    def get_media_paths_for_cluster(
        self,
        cluster_label: int,
        *,
        limit: int = 10,
        include_heic: bool = False,
    ) -> list[str]:
        """Return distinct media file paths for a cluster, randomly sampled.

        By default only JPEG files are returned. Set ``include_heic`` to True
        to also include HEIC files.
        """
        raise NotImplementedError

    @abstractmethod
    def get_cluster_summary(
        self, limit: int = 100, offset: int = 0
    ) -> list[dict]:
        """Return a summary of clusters ordered by distinct-photo count.

        Each item is a dict with keys:
        - ``cluster_label`` (int)
        - ``face_count`` (int)
        - ``photo_count`` (int)
        - ``identity_name`` (str | None)
        - ``sample_path`` (str | None)
        """
        raise NotImplementedError

    @abstractmethod
    def get_identity_summary(
        self, limit: int = 100, offset: int = 0
    ) -> list[dict]:
        """Return a summary of identities ordered by distinct-photo count.

        Each item is a dict with keys:
        - ``identity_id`` (str)
        - ``identity_name`` (str)
        - ``face_count`` (int)
        - ``photo_count`` (int)
        - ``sample_path`` (str | None)
        """
        raise NotImplementedError

    @abstractmethod
    def delete_media(self, media_id: str) -> None:
        """Delete a media file record and its linked faces."""
        raise NotImplementedError

    @abstractmethod
    def get_identity_photo_counts(self) -> dict[str, int]:
        """Return a mapping of identity name to distinct-photo count."""
        raise NotImplementedError

    @abstractmethod
    def update_media_perceptual_hash(
        self, media_id: str, perceptual_hash: str
    ) -> None:
        """Store the perceptual hash for a media file."""
        raise NotImplementedError

    @abstractmethod
    def clear_duplicate_groups(self) -> None:
        """Remove all stored duplicate groups."""
        raise NotImplementedError

    @abstractmethod
    def save_duplicate_group(
        self, group_id: str, members: list[dict]
    ) -> None:
        """Persist a duplicate group and its member links."""
        raise NotImplementedError

    @abstractmethod
    def list_duplicate_groups(self) -> list[dict]:
        """Return duplicate groups with member details.

        Each item is a dict with keys ``group_id``, ``created_at``, and
        ``members``. Each member dict contains ``media_id``, ``path``,
        ``size_bytes``, ``is_primary``, and ``match_level``.
        """
        raise NotImplementedError


class FaceAnalyzer(ABC):
    """Detects faces and computes embeddings for a media file."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Return an identifier for the detector/embedding pipeline."""
        raise NotImplementedError

    @abstractmethod
    def analyze(self, media: MediaFile) -> list[Face]:
        """Return all faces detected in *media*."""
        raise NotImplementedError
