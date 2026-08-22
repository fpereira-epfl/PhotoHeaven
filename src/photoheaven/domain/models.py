"""Domain models for PhotoHeaven.

These are plain data objects with no dependencies on frameworks, databases, or
external libraries. Sensitive data such as embeddings and GPS coordinates are
stored here but must never be logged or printed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class MediaType(Enum):
    IMAGE = "image"
    VIDEO = "video"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class GeoPoint:
    """A latitude/longitude pair. Treat as sensitive data."""

    latitude: float
    longitude: float


@dataclass
class MediaFile:
    """A single photo or video file known to the library."""

    id: str
    path: str
    checksum: str
    size_bytes: int
    mtime: float
    media_type: MediaType = MediaType.UNKNOWN
    capture_datetime: Optional[datetime] = None
    make: Optional[str] = None
    model: Optional[str] = None
    gps: Optional[GeoPoint] = None
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
    cluster_label: Optional[int] = None
    identity_name: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
