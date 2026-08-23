"""Application service that clusters face embeddings into identities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.cluster import DBSCAN

from photoheaven.application.ports import MediaRepository

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

        logger.info("Clustering %d face embedding(s)", len(faces))

        matrix = np.array([face.embedding for face in faces], dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("Face embeddings must be vectors")

        clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
        labels = clusterer.fit_predict(matrix)

        clustered = 0
        noise = 0
        for face, label in zip(faces, labels):
            # DBSCAN labels noise points as -1. Store those as None in the
            # database so they are distinct from unclustered NULL rows only by
            # context (clustering has run).
            db_label: Optional[int] = int(label) if label >= 0 else None
            if db_label is not None:
                clustered += 1
            else:
                noise += 1
            try:
                self.repository.update_face_cluster_label(face.id, db_label)
            except Exception:
                logger.exception("Failed to update cluster label for face %s", face.id)
                return FaceClusteringResult(
                    total_faces=len(faces),
                    clustered_faces=clustered,
                    noise_faces=noise,
                    num_clusters=len({label for label in labels if label >= 0}),
                )

        num_clusters = len({label for label in labels if label >= 0})
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
