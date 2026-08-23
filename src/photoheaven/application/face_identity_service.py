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

    def name_cluster(self, cluster_label: int, identity_name: str) -> int:
        """Assign a human name to a cluster.

        Creates a persistent identity if one does not already exist, links
        every face in the cluster to it, and returns the number of faces
        updated.
        """
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
