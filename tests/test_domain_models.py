"""Tests for domain model safety invariants."""

from __future__ import annotations

from photoheaven.domain.models import Face, GeoPoint, MediaFile, MediaType


def test_face_repr_does_not_leak_embedding() -> None:
    face = Face(
        id="face-1",
        media_id="media-1",
        bbox=(0, 0, 1, 1),
        embedding=[0.123] * 512,
        identity_name="Alice",
    )
    representation = repr(face)

    assert "embedding=<redacted>" in representation
    assert "0.123" not in representation
    assert "Alice" in representation
    assert "face-1" in representation


def test_geopoint_repr_redacts_coordinates() -> None:
    point = GeoPoint(latitude=38.7, longitude=-9.1)
    representation = repr(point)

    assert representation == "GeoPoint(<redacted>)"
    assert "38.7" not in representation
    assert "-9.1" not in representation


def test_mediafile_repr_redacts_path_and_gps() -> None:
    media = MediaFile(
        id="media-1",
        path="/home/user/private/photo.jpg",
        checksum="abc",
        size_bytes=100,
        mtime=1.0,
        media_type=MediaType.IMAGE,
        gps=GeoPoint(latitude=38.7, longitude=-9.1),
    )
    representation = repr(media)

    assert "<redacted>" in representation
    assert "/home/user/private/photo.jpg" not in representation
    assert "38.7" not in representation
    assert "media-1" in representation


def test_attributes_remain_accessible() -> None:
    """Redaction only affects repr; the actual data is still available."""
    face = Face(
        id="face-1",
        media_id="media-1",
        bbox=(0, 0, 1, 1),
        embedding=[0.5, 0.5],
    )
    assert face.embedding == [0.5, 0.5]

    point = GeoPoint(latitude=1.0, longitude=2.0)
    assert point.latitude == 1.0
    assert point.longitude == 2.0
