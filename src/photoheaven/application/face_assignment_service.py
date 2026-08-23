"""Application service for incrementally assigning faces to known identities."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from photoheaven.application.ports import MediaRepository
from photoheaven.domain.models import Face

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FaceAssignmentResult:
    """Result summary for an incremental identity assignment run."""

    identities_used: int
    unassigned_faces: int
    assigned_faces: int


class FaceAssignmentService:
    """Assign unlabelled faces to known identities using centroid distance.

    This is the incremental counterpart to full DBSCAN clustering. It computes
    a centroid embedding for each named identity from all faces already linked
    to that identity, then assigns unlinked faces whose nearest centroid is
    within the configured cosine-distance threshold.
    """

    def __init__(self, repository: MediaRepository) -> None:
        self.repository = repository

    def _compute_centroids(self) -> dict[str, tuple[str, np.ndarray]]:
        """Return a mapping identity_id -> (name, centroid_vector)."""
        identities = self.repository.list_identities(limit=10_000)
        centroids: dict[str, tuple[str, np.ndarray]] = {}
        for identity in identities:
            faces = self.repository.get_faces_for_identity(identity.id)
            if not faces:
                continue
            matrix = np.array(
                [face.embedding for face in faces], dtype=np.float32
            )
            centroid = matrix.mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            centroids[identity.id] = (identity.name, centroid)
        return centroids

    def assign(
        self, *, eps: float = 0.4, batch_size: int = 1000
    ) -> FaceAssignmentResult:
        """Assign faces without an identity to the nearest known identity.

        Args:
            eps: Maximum cosine distance to a centroid for assignment.
            batch_size: Number of unassigned faces to process in one query.

        Returns:
            Summary of identities used and faces assigned / left unassigned.
        """
        centroids = self._compute_centroids()
        if not centroids:
            logger.info("No named identities exist; nothing to assign")
            return FaceAssignmentResult(
                identities_used=0,
                unassigned_faces=0,
                assigned_faces=0,
            )

        identity_ids = list(centroids.keys())
        identity_names = [centroids[iid][0] for iid in identity_ids]
        centroid_matrix = np.array(
            [centroids[iid][1] for iid in identity_ids], dtype=np.float32
        )

        assigned = 0
        unassigned = 0
        offset = 0
        while True:
            faces = self.repository.get_faces_without_identity(
                limit=batch_size, offset=offset
            )
            if not faces:
                break

            for face in faces:
                identity_id, identity_name = self._nearest_identity(
                    face,
                    identity_ids,
                    identity_names,
                    centroid_matrix,
                    eps=eps,
                )
                if identity_id is not None:
                    self.repository.update_face_identity(
                        face.id,
                        identity_id=identity_id,
                        identity_name=identity_name,
                    )
                    assigned += 1
                else:
                    unassigned += 1

            if len(faces) < batch_size:
                break
            offset += batch_size

        logger.info(
            "Incremental assignment complete: %d assigned, %d unassigned "
            "using %d identity centroids",
            assigned,
            unassigned,
            len(centroids),
        )
        return FaceAssignmentResult(
            identities_used=len(centroids),
            unassigned_faces=unassigned,
            assigned_faces=assigned,
        )

    def _nearest_identity(
        self,
        face: Face,
        identity_ids: list[str],
        identity_names: list[str | None],
        centroid_matrix: np.ndarray,
        *,
        eps: float,
    ) -> tuple[str | None, str | None]:
        """Return the nearest identity within ``eps``, or (None, None)."""
        embedding = np.array(face.embedding, dtype=np.float32)
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return None, None
        embedding = embedding / norm

        # Cosine distance = 1 - cosine similarity.
        similarities = centroid_matrix @ embedding
        distances = 1.0 - similarities
        nearest_idx = int(np.argmin(distances))
        nearest_distance = float(distances[nearest_idx])

        if nearest_distance <= eps:
            return (
                identity_ids[nearest_idx],
                identity_names[nearest_idx],
            )

        return None, None
