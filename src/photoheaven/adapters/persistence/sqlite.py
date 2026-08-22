"""SQLite persistence adapter for media files and faces."""

from __future__ import annotations

import array
import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    create_engine,
)
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from photoheaven.application.ports import MediaRepository
from photoheaven.domain.models import Face, GeoPoint, MediaFile, MediaType

logger = logging.getLogger(__name__)
Base = declarative_base()


class _MediaFileORM(Base):
    __tablename__ = "media_files"

    id = Column(String(36), primary_key=True)
    path = Column(String, nullable=False, unique=True)
    checksum = Column(String(64), nullable=False, unique=True, index=True)
    size_bytes = Column(Integer, nullable=False)
    mtime = Column(Float, nullable=False)
    media_type = Column(String(16), nullable=False, default="unknown")
    capture_datetime = Column(DateTime, nullable=True)
    make = Column(String, nullable=True)
    model = Column(String, nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    face_analysis_at = Column(DateTime, nullable=True)
    face_analysis_version = Column(String, nullable=True)
    metadata_extracted = Column(Integer, nullable=False, default=1)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class _FaceORM(Base):
    __tablename__ = "faces"

    id = Column(String(36), primary_key=True)
    media_id = Column(String(36), ForeignKey("media_files.id"), nullable=False, index=True)
    bbox_x1 = Column(Integer, nullable=False)
    bbox_y1 = Column(Integer, nullable=False)
    bbox_x2 = Column(Integer, nullable=False)
    bbox_y2 = Column(Integer, nullable=False)
    embedding_blob = Column(LargeBinary, nullable=False)
    embedding_version = Column(String, nullable=False, default="unknown")
    detection_confidence = Column(Float, nullable=False, default=0.0)
    cluster_label = Column(Integer, nullable=True)
    identity_name = Column(String, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


def _media_to_domain(row: _MediaFileORM) -> MediaFile:
    gps = None
    if row.latitude is not None and row.longitude is not None:
        gps = GeoPoint(latitude=row.latitude, longitude=row.longitude)
    return MediaFile(
        id=row.id,
        path=row.path,
        checksum=row.checksum,
        size_bytes=row.size_bytes,
        mtime=row.mtime,
        media_type=MediaType(row.media_type),
        capture_datetime=row.capture_datetime,
        make=row.make,
        model=row.model,
        gps=gps,
        face_analysis_at=row.face_analysis_at,
        face_analysis_version=row.face_analysis_version,
        metadata_extracted=bool(row.metadata_extracted),
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _media_to_orm(media: MediaFile) -> _MediaFileORM:
    return _MediaFileORM(
        id=media.id,
        path=media.path,
        checksum=media.checksum,
        size_bytes=media.size_bytes,
        mtime=media.mtime,
        media_type=media.media_type.value,
        capture_datetime=media.capture_datetime,
        make=media.make,
        model=media.model,
        latitude=media.gps.latitude if media.gps else None,
        longitude=media.gps.longitude if media.gps else None,
        face_analysis_at=media.face_analysis_at,
        face_analysis_version=media.face_analysis_version,
        metadata_extracted=1 if media.metadata_extracted else 0,
        created_at=media.created_at,
        updated_at=media.updated_at,
    )


def _embedding_to_bytes(embedding: list[float]) -> bytes:
    """Serialize a float32 vector to bytes."""
    return array.array("f", embedding).tobytes()


def _embedding_from_bytes(blob: bytes) -> list[float]:
    """Deserialize bytes back to a float32 vector."""
    arr = array.array("f")
    arr.frombytes(blob)
    return arr.tolist()


def _face_to_domain(row: _FaceORM) -> Face:
    return Face(
        id=row.id,
        media_id=row.media_id,
        bbox=(row.bbox_x1, row.bbox_y1, row.bbox_x2, row.bbox_y2),
        embedding=_embedding_from_bytes(row.embedding_blob),
        embedding_version=row.embedding_version,
        detection_confidence=row.detection_confidence,
        cluster_label=row.cluster_label,
        identity_name=row.identity_name,
        created_at=row.created_at,
    )


def _face_to_orm(face: Face) -> _FaceORM:
    return _FaceORM(
        id=face.id,
        media_id=face.media_id,
        bbox_x1=face.bbox[0],
        bbox_y1=face.bbox[1],
        bbox_x2=face.bbox[2],
        bbox_y2=face.bbox[3],
        embedding_blob=_embedding_to_bytes(face.embedding),
        embedding_version=face.embedding_version,
        detection_confidence=face.detection_confidence,
        cluster_label=face.cluster_label,
        identity_name=face.identity_name,
        created_at=face.created_at,
    )


def _migrate_schema(engine) -> None:
    """Add any columns that are missing from an older database file.

    SQLAlchemy's ``create_all`` creates missing tables but does not alter
    existing ones. This helper performs lightweight ``ALTER TABLE`` additions
    for columns introduced after the initial schema.
    """
    from sqlalchemy import inspect

    inspector = inspect(engine)

    def _add_column_if_missing(table_name: str, column: Column) -> None:
        if not inspector.has_table(table_name):
            return
        existing = {col["name"] for col in inspector.get_columns(table_name)}
        if column.name in existing:
            return
        column_type = column.type.compile(dialect=engine.dialect)
        nullable = "NULL" if column.nullable else "NOT NULL"
        default = ""
        if column.default is not None and column.default.is_scalar:
            default_value = column.default.arg
            if isinstance(default_value, str):
                default = f" DEFAULT '{default_value}'"
            else:
                default = f" DEFAULT {default_value}"
        sql = f"ALTER TABLE {table_name} ADD COLUMN {column.name} {column_type} {nullable}{default}"
        with engine.begin() as conn:
            conn.exec_driver_sql(sql)
        logger.info("Added missing column %s.%s", table_name, column.name)

    _add_column_if_missing(
        "media_files",
        Column("face_analysis_at", DateTime, nullable=True),
    )
    _add_column_if_missing(
        "media_files",
        Column("face_analysis_version", String, nullable=True),
    )
    _add_column_if_missing(
        "media_files",
        Column("metadata_extracted", Integer, nullable=False, default=1),
    )


class SqliteMediaRepository(MediaRepository):
    """SQLite-backed repository using SQLAlchemy."""

    def __init__(self, db_path: str) -> None:
        self.engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(self.engine)
        _migrate_schema(self.engine)
        self._session_factory = sessionmaker(self.engine)

    def _session(self) -> Session:
        return self._session_factory()

    def get_by_checksum(self, checksum: str) -> Optional[MediaFile]:
        with self._session() as session:
            row = session.query(_MediaFileORM).filter_by(checksum=checksum).first()
            return _media_to_domain(row) if row else None

    def save_media(self, media: MediaFile) -> None:
        with self._session() as session:
            existing = session.get(_MediaFileORM, media.id)
            if existing:
                session.delete(existing)
            session.add(_media_to_orm(media))
            session.commit()

    def count_media(self) -> int:
        with self._session() as session:
            return session.query(_MediaFileORM).count()

    def list_media(self, limit: int = 100, offset: int = 0) -> list[MediaFile]:
        with self._session() as session:
            rows = (
                session.query(_MediaFileORM)
                .order_by(_MediaFileORM.capture_datetime)
                .offset(offset)
                .limit(limit)
                .all()
            )
            return [_media_to_domain(row) for row in rows]

    def save_face(self, face: Face) -> None:
        with self._session() as session:
            existing = session.get(_FaceORM, face.id)
            if existing:
                session.delete(existing)
            session.add(_face_to_orm(face))
            session.commit()

    def count_faces(self) -> int:
        with self._session() as session:
            return session.query(_FaceORM).count()

    def media_has_faces(self, media_id: str) -> bool:
        with self._session() as session:
            return session.query(_FaceORM).filter_by(media_id=media_id).first() is not None

    def get_unprocessed_faces_media(
        self, limit: int = 100, offset: int = 0
    ) -> list[MediaFile]:
        with self._session() as session:
            rows = (
                session.query(_MediaFileORM)
                .filter(_MediaFileORM.face_analysis_at.is_(None))
                .order_by(_MediaFileORM.capture_datetime)
                .offset(offset)
                .limit(limit)
                .all()
            )
            return [_media_to_domain(row) for row in rows]

    def update_media_face_analysis(
        self, media_id: str, analyzed_at: datetime, version: str
    ) -> None:
        with self._session() as session:
            row = session.get(_MediaFileORM, media_id)
            if row is None:
                logger.warning("Cannot mark face analysis: media %s not found", media_id)
                return
            row.face_analysis_at = analyzed_at
            row.face_analysis_version = version
            row.updated_at = analyzed_at
            session.commit()

    def list_faces_for_media(self, media_id: str) -> list[Face]:
        with self._session() as session:
            rows = (
                session.query(_FaceORM)
                .filter_by(media_id=media_id)
                .order_by(_FaceORM.detection_confidence.desc())
                .all()
            )
            return [_face_to_domain(row) for row in rows]

    def get_face_by_id(self, face_id: str) -> Face | None:
        with self._session() as session:
            row = session.get(_FaceORM, face_id)
            return _face_to_domain(row) if row else None

    def list_faces(self, limit: int = 100, offset: int = 0) -> list[Face]:
        with self._session() as session:
            rows = (
                session.query(_FaceORM)
                .order_by(_FaceORM.created_at)
                .offset(offset)
                .limit(limit)
                .all()
            )
            return [_face_to_domain(row) for row in rows]

    def update_face_cluster_label(
        self, face_id: str, cluster_label: int | None
    ) -> None:
        with self._session() as session:
            row = session.get(_FaceORM, face_id)
            if row is None:
                logger.warning("Cannot update cluster label: face %s not found", face_id)
                return
            row.cluster_label = cluster_label
            session.commit()

    def update_face_identity_name_for_cluster(
        self, cluster_label: int, identity_name: str | None
    ) -> None:
        with self._session() as session:
            (
                session.query(_FaceORM)
                .filter_by(cluster_label=cluster_label)
                .update({"identity_name": identity_name})
            )
            session.commit()
