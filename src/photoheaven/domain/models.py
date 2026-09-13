"""Domain models for PhotoHeaven.

These are plain data objects with no dependencies on frameworks, databases, or
external libraries. Sensitive data such as embeddings and GPS coordinates are
stored here but must never be logged or printed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class MediaType(Enum):
    IMAGE = "image"
    VIDEO = "video"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class GeoPoint:
    """A latitude/longitude pair. Treat as sensitive data."""

    latitude: float
    longitude: float

    def __repr__(self) -> str:
        return "GeoPoint(<redacted>)"


@dataclass
class MediaFile:
    """A single photo or video file known to the library."""

    id: str
    path: str
    checksum: str
    size_bytes: int
    mtime: float
    media_type: MediaType = MediaType.UNKNOWN
    capture_datetime: datetime | None = None
    make: str | None = None
    model: str | None = None
    gps: GeoPoint | None = None
    face_analysis_at: datetime | None = None
    """When face detection was last run on this file. None means not yet analysed."""

    face_analysis_version: str | None = None
    """Identifier of the face analysis pipeline/version used."""

    place_analysis_at: datetime | None = None
    """When place/scene detection was last run on this file. None means not yet analysed."""

    place_analysis_version: str | None = None
    """Identifier of the place analysis pipeline/version used."""

    metadata_extracted: bool = True
    """False when metadata extraction failed (e.g. corrupt/unreadable file)."""

    perceptual_hash: str | None = None
    """Hex perceptual hash (e.g. pHash) used for near-duplicate detection."""

    duration_seconds: float | None = None
    """Video duration in seconds, if known."""

    video_frame_hashes: list[str] | None = None
    """Perceptual hashes of sampled video keyframes."""

    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def __repr__(self) -> str:
        return (
            f"MediaFile(id={self.id!r}, path=<redacted>, "
            f"checksum={self.checksum!r}, size_bytes={self.size_bytes}, "
            f"media_type={self.media_type}, gps=<redacted>, "
            f"face_analysis_at={self.face_analysis_at}, "
            f"place_analysis_at={self.place_analysis_at}, "
            f"metadata_extracted={self.metadata_extracted}, "
            f"perceptual_hash={self.perceptual_hash!r}, "
            f"duration_seconds={self.duration_seconds}, "
            f"video_frame_hashes=<redacted>)"
        )


@dataclass
class Identity:
    """A persistent identity (person) in the library."""

    id: str
    name: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Face:
    """A face detected in a media file."""

    id: str
    media_id: str
    bbox: tuple[int, int, int, int]
    """(x1, y1, x2, y2) in pixel coordinates, top-left origin."""

    embedding: list[float]
    """Face embedding vector (e.g. 512-dim ArcFace). Treat as sensitive."""

    embedding_version: str = "unknown"
    """Identifier of the model/pipeline that produced the embedding."""

    detection_confidence: float = 0.0
    cluster_label: int | None = None
    identity_id: str | None = None
    """Persistent identity id; survives re-clustering."""

    identity_name: str | None = None
    """Denormalised display name for convenience."""

    created_at: datetime = field(default_factory=datetime.utcnow)

    def __repr__(self) -> str:
        return (
            f"Face(id={self.id!r}, media_id={self.media_id!r}, "
            f"bbox={self.bbox}, embedding=<redacted>, "
            f"embedding_version={self.embedding_version!r}, "
            f"detection_confidence={self.detection_confidence}, "
            f"cluster_label={self.cluster_label}, "
            f"identity_id={self.identity_id!r}, "
            f"identity_name={self.identity_name!r})"
        )


@dataclass
class PlaceRecord:
    """A country/scene classification for a media file."""

    media_id: str
    country: str | None = None
    country_source: str | None = None
    """How the country was determined, e.g. 'gps' or 'visual'."""

    country_confidence: float | None = None
    scene: str | None = None
    scene_confidence: float | None = None
    analyzed_at: datetime = field(default_factory=datetime.utcnow)
    version: str = "unknown"

    def __repr__(self) -> str:
        return (
            f"PlaceRecord(media_id={self.media_id!r}, "
            f"country={self.country!r}, country_source={self.country_source!r}, "
            f"country_confidence={self.country_confidence}, "
            f"scene={self.scene!r}, scene_confidence={self.scene_confidence}, "
            f"version={self.version!r})"
        )
