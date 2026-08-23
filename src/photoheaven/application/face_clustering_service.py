"""Application service that clusters face embeddings into identities."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.cluster import DBSCAN

from photoheaven.application.ports import MediaRepository
from photoheaven.domain.models import Identity

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FaceClusteringResult:
    """Result summary for a face-clustering run."""

    total_faces: int = 0
    clustered_faces: int = 0
    noise_faces: int = 0
    num_clusters: int = 0


class FaceClusteringService:
    """Group stored face embeddings into identity clusters.

    The service loads embeddings from the repository, runs a clustering
    algorithm, and writes cluster labels back. It never logs or exposes raw
    embeddings.
    """

    def __init__(self, repository: MediaRepository) -> None:
        self.repository = repository

    def _ensure_identities_for_named_faces(self, faces: list) -> None:
        """Convert legacy name-only faces to persistent identities.

        Older rows may store ``identity_name`` without an ``identity_id``.
        Before re-clustering, ensure every named face is linked to a persistent
        identity so its name survives the cluster relabel.
        """
        for face in faces:
            if face.identity_id is not None or face.identity_name is None:
                continue
            identity = self.repository.get_identity_by_name(face.identity_name)
            if identity is None:
                identity = Identity(
                    id=str(uuid.uuid4()),
                    name=face.identity_name,
                )
                self.repository.save_identity(identity)
                logger.info(
                    "Created identity %s for existing name '%s'",
                    identity.id,
                    identity.name,
                )
            self.repository.update_face_identity(
                face.id,
                identity_id=identity.id,
                identity_name=identity.name,
            )
            face.identity_id = identity.id

    def cluster(
        self,
        *,
        eps: float = 0.4,
        min_samples: int = 2,
        algorithm: str = "dbscan",
    ) -> FaceClusteringResult:
        """Cluster all stored faces.

        Args:
            eps: Maximum distance between two samples for them to be considered
                neighbours. For cosine distance on ArcFace embeddings, 0.4 is a
                reasonable starting point.
            min_samples: Minimum number of faces to form a dense cluster.
            algorithm: Clustering algorithm to use. Only ``dbscan`` is supported
                for now.

        Returns:
            Summary of clustered, noise, and total face counts.
        """
        if algorithm.lower() != "dbscan":
            raise ValueError(f"Unsupported clustering algorithm: {algorithm}")

        faces = self.repository.get_all_faces()
        if not faces:
            logger.info("No faces to cluster")
            return FaceClusteringResult()

        versions = self.repository.get_embedding_versions()
        if len(versions) > 1:
            raise ValueError(
                f"Cannot cluster faces with mixed embedding versions: {sorted(versions)}. "
                "Run with --force to re-analyse older media, or ensure all faces "
                "use the same face analysis model."
            )

        logger.info("Clustering %d face embedding(s)", len(faces))

        self._ensure_identities_for_named_faces(faces)

        matrix = np.array([face.embedding for face in faces], dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("Face embeddings must be vectors")

        clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
        labels = clusterer.fit_predict(matrix)

        # Preserve existing identity mappings so we can re-attach names after
        # DBSCAN assigns new, potentially different cluster labels.
        old_identity_by_face: dict[str, tuple[str | None, str | None]] = {
            face.id: (face.identity_id, face.identity_name) for face in faces
        }

        # Group faces by their new DBSCAN label.
        new_clusters: dict[int, list[Face]] = {}
        noise_faces: list[Face] = []
        for face, label in zip(faces, labels):
            if label >= 0:
                new_clusters.setdefault(int(label), []).append(face)
            else:
                noise_faces.append(face)

        # Determine the dominant identity for each new cluster.
        cluster_identities: dict[int, tuple[str | None, str | None]] = {}
        for cluster_label, cluster_faces in new_clusters.items():
            identity_counts: dict[str, tuple[int, str]] = {}
            for face in cluster_faces:
                identity_id, identity_name = old_identity_by_face[face.id]
                if identity_id is None:
                    continue
                count, _ = identity_counts.get(identity_id, (0, identity_name or ""))
                identity_counts[identity_id] = (count + 1, identity_name or "")
            if identity_counts:
                # Pick the identity with the most faces in this cluster.
                dominant_identity_id, (count, identity_name) = max(
                    identity_counts.items(), key=lambda item: item[1][0]
                )
                cluster_identities[cluster_label] = (dominant_identity_id, identity_name)

        clustered = 0
        noise = 0
        for cluster_label, cluster_faces in new_clusters.items():
            identity_id, identity_name = cluster_identities.get(
                cluster_label, (None, None)
            )
            for face in cluster_faces:
                db_label = int(cluster_label)
                clustered += 1
                try:
                    self.repository.update_face_cluster_label(face.id, db_label)
                    if identity_id is not None:
                        self.repository.update_face_identity(
                            face.id,
                            identity_id=identity_id,
                            identity_name=identity_name,
                        )
                except Exception:
                    logger.exception(
                        "Failed to update cluster label for face %s", face.id
                    )
                    return FaceClusteringResult(
                        total_faces=len(faces),
                        clustered_faces=clustered,
                        noise_faces=noise,
                        num_clusters=len(new_clusters),
                    )

        # Noise points keep their previous identity (if any) but lose their
        # cluster label, matching DBSCAN semantics.
        for face in noise_faces:
            noise += 1
            try:
                self.repository.update_face_cluster_label(face.id, None)
            except Exception:
                logger.exception(
                    "Failed to update cluster label for face %s", face.id
                )
                return FaceClusteringResult(
                    total_faces=len(faces),
                    clustered_faces=clustered,
                    noise_faces=noise,
                    num_clusters=len(new_clusters),
                )

        num_clusters = len(new_clusters)
        logger.info(
            "Clustering complete: %d cluster(s), %d clustered face(s), %d noise",
            num_clusters,
            clustered,
            noise,
        )
        return FaceClusteringResult(
            total_faces=len(faces),
            clustered_faces=clustered,
            noise_faces=noise,
            num_clusters=num_clusters,
        )
