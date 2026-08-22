"""Ports (interfaces) that the domain/application layer depends on.

Adapters live outside this layer and implement these interfaces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional

from photoheaven.domain.models import Face, GeoPoint, MediaFile, MediaType


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
    ) -> None:
        self.media_type = media_type
        self.capture_datetime = capture_datetime
        self.make = make
        self.model = model
        self.gps = gps


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
    def count_media(self) -> int:
        """Return the number of media files in the library."""
        raise NotImplementedError

    @abstractmethod
    def list_media(self, limit: int = 100, offset: int = 0) -> list[MediaFile]:
        """Return a paginated list of media files."""
        raise NotImplementedError

    @abstractmethod
    def save_face(self, face: Face) -> None:
        """Persist a detected face."""
        raise NotImplementedError

    @abstractmethod
    def count_faces(self) -> int:
        """Return the number of faces in the library."""
        raise NotImplementedError


class FaceAnalyzer(ABC):
    """Detects faces and computes embeddings for a media file."""

    @abstractmethod
    def analyze(self, media: MediaFile) -> list[Face]:
        """Return all faces detected in *media*."""
        raise NotImplementedError
