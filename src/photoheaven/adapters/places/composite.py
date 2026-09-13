"""Composite place identifier.

Country is derived from EXIF GPS + offline reverse geocoding only.
Scene is derived from a Places365 CNN with a configurable confidence threshold.
"""

from __future__ import annotations

import logging
from pathlib import Path

from photoheaven.adapters.places.places365 import Places365SceneIdentifier
from photoheaven.application.ports import PlaceIdentifier
from photoheaven.domain.models import MediaFile, MediaType, PlaceRecord

logger = logging.getLogger(__name__)


class CompositePlaceIdentifier(PlaceIdentifier):
    """Identify places using GPS for country and Places365 for scene."""

    def __init__(
        self,
        scene_arch: str = "resnet50",
        scene_threshold: float = 0.1,
    ) -> None:
        self._scene_identifier = Places365SceneIdentifier(
            arch=scene_arch,
            threshold=scene_threshold,
        )

    @property
    def version(self) -> str:
        return f"gps+{self._scene_identifier.version}"

    def _country_from_gps(self, media: MediaFile) -> tuple[str, float] | None:
        if media.gps is None:
            return None

        try:
            import reverse_geocoder  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "reverse_geocoder is required for GPS place identification. "
                "Install with: pip install reverse-geocoder"
            ) from exc

        try:
            # reverse_geocoder expects (lat, lon).
            results = reverse_geocoder.search(
                (media.gps.latitude, media.gps.longitude), mode=1
            )
        except Exception:
            logger.exception("Reverse geocoding failed for %s", media.path)
            return None

        if not results:
            return None

        result = results[0] if isinstance(results, list) else results
        country_code = result.get("cc")
        if not country_code:
            return None

        try:
            import pycountry  # type: ignore[import-not-found]

            country = pycountry.countries.get(alpha_2=country_code)
            country_name = (
                getattr(country, "common_name", None)
                or getattr(country, "name", None)
                or country_code
            )
        except Exception:
            country_name = country_code

        # Confidence for GPS-based country is high but not absolute.
        return country_name, 0.95

    def identify(self, media: MediaFile) -> PlaceRecord:
        if media.media_type is not MediaType.IMAGE:
            return PlaceRecord(
                media_id=media.id,
                country=None,
                country_source=None,
                country_confidence=None,
                scene=None,
                scene_confidence=None,
                version=self.version,
            )

        country: str | None = None
        country_source: str | None = None
        country_confidence: float | None = None

        gps_result = self._country_from_gps(media)
        if gps_result is not None:
            country, country_confidence = gps_result
            country_source = "gps"

        scene: str | None = None
        scene_confidence: float | None = None

        try:
            scene, scene_confidence = self._scene_identifier.identify(
                Path(media.path)
            )
        except Exception:
            logger.exception(
                "Places365 scene classification failed for %s", media.path
            )

        return PlaceRecord(
            media_id=media.id,
            country=country,
            country_source=country_source,
            country_confidence=country_confidence,
            scene=scene,
            scene_confidence=scene_confidence,
            version=self.version,
        )
