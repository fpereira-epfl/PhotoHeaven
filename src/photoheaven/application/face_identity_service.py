"""Application service for viewing and naming face clusters/identities."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from photoheaven.application.ports import MediaRepository
from photoheaven.domain.models import Identity

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClusterSummary:
    """Human-readable summary of a face cluster."""

    cluster_label: int
    face_count: int
    photo_count: int
    identity_name: str | None
    sample_path: str | None


@dataclass(frozen=True)
class IdentitySummary:
    """Human-readable summary of a named identity."""

    identity_id: str
    identity_name: str
    face_count: int
    photo_count: int
    sample_path: str | None


class FaceIdentityService:
    """List, name, and search face clusters without exposing embeddings."""

    def __init__(self, repository: MediaRepository) -> None:
        self.repository = repository

    def list_clusters(
        self, *, limit: int = 100, offset: int = 0
    ) -> list[ClusterSummary]:
        """Return clusters ordered by size (largest first)."""
        rows = self.repository.get_cluster_summary(limit=limit, offset=offset)
        return [
            ClusterSummary(
                cluster_label=row["cluster_label"],
                face_count=row["face_count"],
                photo_count=row["photo_count"],
                identity_name=row["identity_name"],
                sample_path=row["sample_path"],
            )
            for row in rows
        ]

    def list_identities(
        self, *, limit: int = 100, offset: int = 0
    ) -> list[IdentitySummary]:
        """Return named identities ordered by distinct-photo count."""
        rows = self.repository.get_identity_summary(
            limit=limit, offset=offset
        )
        return [
            IdentitySummary(
                identity_id=row["identity_id"],
                identity_name=row["identity_name"] or "—",
                face_count=row["face_count"],
                photo_count=row["photo_count"],
                sample_path=row["sample_path"],
            )
            for row in rows
        ]

    _MAX_IDENTITY_NAME_LENGTH = 100

    def _sanitize_identity_name(self, name: str) -> str:
        """Strip whitespace, collapse spaces, and cap length.

        Raises ``ValueError`` for empty or overly long names.
        """
        sanitized = " ".join(name.split())
        if not sanitized:
            raise ValueError("Identity name cannot be empty or only whitespace")
        if len(sanitized) > self._MAX_IDENTITY_NAME_LENGTH:
            raise ValueError(
                f"Identity name exceeds {self._MAX_IDENTITY_NAME_LENGTH} characters"
            )
        return sanitized

    def name_cluster(self, cluster_label: int, identity_name: str) -> int:
        """Assign a human name to a cluster.

        Creates a persistent identity if one does not already exist, links
        every face in the cluster to it, and returns the number of faces
        updated.
        """
        identity_name = self._sanitize_identity_name(identity_name)
        logger.info(
            "Naming cluster %d as '%s'", cluster_label, identity_name
        )
        identity = self.repository.get_identity_by_name(identity_name)
        if identity is None:
            identity = Identity(
                id=str(uuid.uuid4()),
                name=identity_name,
            )
            self.repository.save_identity(identity)

        faces = self.repository.get_faces_for_cluster(cluster_label)
        for face in faces:
            self.repository.update_face_identity(
                face.id,
                identity_id=identity.id,
                identity_name=identity.name,
            )
        return len(faces)

    def get_sample_photos(
        self,
        cluster_label: int,
        *,
        limit: int = 10,
        include_heic: bool = False,
    ) -> list[str]:
        """Return up to *limit* random sample photos for a cluster.

        JPEG files are always included. HEIC files are included only when
        ``include_heic`` is True.
        """
        return self.repository.get_media_paths_for_cluster(
            cluster_label, limit=limit, include_heic=include_heic
        )
