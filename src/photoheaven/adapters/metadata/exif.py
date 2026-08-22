"""Image and video metadata extraction adapters."""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from PIL import ExifTags, Image

# Register HEIC/HEIF opener with Pillow if the dependency is present.
try:
    import pillow_heif  # type: ignore

    pillow_heif.register_heif_opener()
except Exception:  # pragma: no cover - optional dependency
    pass

from photoheaven.application.ports import MediaMetadata, MetadataExtractor
from photoheaven.domain.models import GeoPoint, MediaType

logger = logging.getLogger(__name__)

_EXIF_DATE_TAGS = ["DateTimeOriginal", "DateTimeDigitized", "DateTime"]


def _dms_to_decimal(dms: tuple, ref: str) -> float:
    """Convert EXIF DMS tuple to decimal degrees."""
    degrees, minutes, seconds = dms
    decimal = float(degrees) + float(minutes) / 60 + float(seconds) / 3600
    if ref in {"S", "W"}:
        decimal = -decimal
    return decimal


def _gps_value(gps_info: dict, *keys):
    """Return the first matching key from a GPSInfo dict."""
    for key in keys:
        if key in gps_info:
            return gps_info[key]
    return None


def _extract_exif_gps(exif: dict) -> Optional[GeoPoint]:
    gps_info = exif.get("GPSInfo")
    if not gps_info or not isinstance(gps_info, dict):
        return None
    try:
        lat = _dms_to_decimal(
            _gps_value(gps_info, 2, "GPSLatitude"),
            _gps_value(gps_info, 1, "GPSLatitudeRef"),
        )
        lon = _dms_to_decimal(
            _gps_value(gps_info, 4, "GPSLongitude"),
            _gps_value(gps_info, 3, "GPSLongitudeRef"),
        )
        return GeoPoint(latitude=lat, longitude=lon)
    except Exception:
        return None


def _parse_exif_datetime(value: str) -> Optional[datetime]:
    """Parse common EXIF datetime string formats."""
    if not value or value in {"0000:00:00 00:00:00", "    :  :     :  :  "}:
        return None
    try:
        return datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
    except ValueError:
        pass
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def _extract_exif_datetime(exif: dict) -> Optional[datetime]:
    for tag_name in _EXIF_DATE_TAGS:
        value = exif.get(tag_name)
        if value:
            parsed = _parse_exif_datetime(value)
            if parsed:
                return parsed
    return None


def _filesystem_datetime(path: Path) -> datetime:
    stat = path.stat()
    return datetime.fromtimestamp(stat.st_mtime)


class PillowExifExtractor(MetadataExtractor):
    """Extract image metadata using Pillow (JPEG/PNG/TIFF/HEIC/WebP/etc.)."""

    def extract(self, path: Path, media_type: MediaType) -> MediaMetadata:
        try:
            with Image.open(path) as img:
                exif = self._read_exif(img)

                capture_datetime = _extract_exif_datetime(exif)
                make = exif.get("Make")
                model = exif.get("Model")
                gps = _extract_exif_gps(exif)

                return MediaMetadata(
                    media_type=MediaType.IMAGE,
                    capture_datetime=capture_datetime,
                    make=make,
                    model=model,
                    gps=gps,
                )
        except Exception as exc:
            logger.warning("Pillow metadata extraction failed for %s: %s", path, exc)
            # If we didn't already know this was an image, leave the type as
            # unknown so the fallback extractor can try video/container parsing.
            fallback_type = MediaType.IMAGE if media_type is MediaType.IMAGE else MediaType.UNKNOWN
            return MediaMetadata(media_type=fallback_type)

    @staticmethod
    def _read_exif(img: Image.Image) -> dict:
        """Read EXIF tags using Pillow's modern API, with HEIF fallback."""
        exif: dict = {}
        raw_exif = img.getexif()
        if raw_exif:
            for tag_id in raw_exif:
                tag_name = ExifTags.TAGS.get(tag_id, tag_id)
                value = raw_exif[tag_id]
                if tag_name == "GPSInfo" and isinstance(value, dict):
                    # Convert numeric GPS tag ids to readable names while
                    # keeping numeric keys available too.
                    gps: dict = {}
                    for gps_key, gps_value in value.items():
                        gps[gps_key] = gps_value
                        gps_name = ExifTags.GPSTAGS.get(gps_key, gps_key)
                        gps[gps_name] = gps_value
                    exif[tag_name] = gps
                else:
                    exif[tag_name] = value

        # Pillow's getexif() may be empty for HEIC even though the raw EXIF
        # bytes are available in img.info. Try a byte-level fallback.
        if not exif and "exif" in img.info:
            try:
                exif_data = img.info["exif"]
                if isinstance(exif_data, (bytes, bytearray)):
                    from PIL import Image as PILImage

                    fallback_exif = PILImage.Exif()
                    fallback_exif.load(exif_data)
                    for tag_id in fallback_exif:
                        tag_name = ExifTags.TAGS.get(tag_id, tag_id)
                        exif[tag_name] = fallback_exif[tag_id]
            except Exception:
                pass

        return exif


class VideoMetadataExtractor(MetadataExtractor):
    """Extract video/container metadata using pymediainfo (MediaInfo binary required)."""

    def __init__(self) -> None:
        self._available: bool | None = None

    def _is_available(self) -> bool:
        if self._available is None:
            try:
                from pymediainfo import MediaInfo  # type: ignore

                MediaInfo.parse(__file__)  # quick smoke test
                self._available = True
            except Exception:
                logger.warning(
                    "pymediainfo/MediaInfo is not available; video metadata will be limited."
                )
                self._available = False
        return self._available

    def extract(self, path: Path, media_type: MediaType) -> MediaMetadata:
        if not self._is_available():
            return MediaMetadata(media_type=MediaType.VIDEO)

        try:
            from pymediainfo import MediaInfo  # type: ignore

            info = MediaInfo.parse(str(path))
            capture_datetime: Optional[datetime] = None
            make: Optional[str] = None
            model: Optional[str] = None

            for track in info.tracks:
                if track.track_type == "General":
                    if not capture_datetime and track.encoded_date:
                        parsed = self._parse_track_date(track.encoded_date)
                        if parsed:
                            capture_datetime = parsed
                    if not capture_datetime and track.tagged_date:
                        parsed = self._parse_track_date(track.tagged_date)
                        if parsed:
                            capture_datetime = parsed
                elif track.track_type in {"Video", "Image"}:
                    if not capture_datetime and track.encoded_date:
                        parsed = self._parse_track_date(track.encoded_date)
                        if parsed:
                            capture_datetime = parsed
                    # Some phones store Make/Model in the video track.
                    if track.manufacturer:
                        make = track.manufacturer
                    if track.model:
                        model = track.model

            return MediaMetadata(
                media_type=MediaType.VIDEO,
                capture_datetime=capture_datetime,
                make=make,
                model=model,
                gps=None,
            )
        except Exception as exc:
            logger.warning("Video metadata extraction failed for %s: %s", path, exc)
            return MediaMetadata(media_type=MediaType.VIDEO)

    @staticmethod
    def _parse_track_date(value: str) -> Optional[datetime]:
        # MediaInfo dates often look like "UTC 2023-01-15 10:30:00"
        # or "2023-01-15 10:30:00".
        value = value.strip()
        value = re.sub(r"^UTC\s+", "", value)
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y:%m:%d %H:%M:%S"):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
        return None


class FallbackMetadataExtractor(MetadataExtractor):
    """Tries image extraction first, then video extraction, falling back to filesystem dates."""

    def __init__(self) -> None:
        self._image_extractor = PillowExifExtractor()
        self._video_extractor = VideoMetadataExtractor()

    def extract(self, path: Path, media_type: MediaType) -> MediaMetadata:
        if media_type is MediaType.IMAGE:
            metadata = self._image_extractor.extract(path, media_type)
        elif media_type is MediaType.VIDEO:
            metadata = self._video_extractor.extract(path, media_type)
        else:
            # Guess by trying Pillow first, then video/container parsing.
            metadata = self._image_extractor.extract(path, media_type)
            if metadata.media_type is MediaType.UNKNOWN:
                metadata = self._video_extractor.extract(path, media_type)
            # If neither could identify it, still treat it as a video file if
            # the extension suggests video, so it gets a sensible media type.
            if metadata.media_type is MediaType.UNKNOWN:
                suffix = path.suffix.lower()
                if suffix in {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".3gp",
                              ".webm", ".mts", ".m2ts", ".ts", ".mpg", ".mpeg"}:
                    metadata.media_type = MediaType.VIDEO

        if metadata.capture_datetime is None:
            metadata.capture_datetime = _filesystem_datetime(path)
            logger.debug("Using filesystem mtime for capture date: %s", path)

        return metadata
