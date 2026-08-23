"""Application service for library initialisation."""

from __future__ import annotations

import logging
from pathlib import Path

from photoheaven.adapters.persistence.sqlite import SqliteMediaRepository
from photoheaven.application.ports import Hasher

logger = logging.getLogger(__name__)

DB_DIR = "db"
DB_NAME = "photoheaven.db"
FILES_DIR = "files"


class LibraryService:
    """Create and initialise PhotoHeaven library packages.

    A library package is a self-contained folder with the database under
    ``db/photoheaven.db`` and media files under ``files/``.
    """

    def __init__(self, hasher: Hasher) -> None:
        self.hasher = hasher

    def init_library(self, library_path: Path) -> Path:
        """Create a new, empty library folder with a database.

        Returns the path to the created database.
        """
        library_path = library_path.expanduser().resolve()
        library_path.mkdir(parents=True, exist_ok=True)
        (library_path / DB_DIR).mkdir(exist_ok=True)
        (library_path / FILES_DIR).mkdir(exist_ok=True)

        db_path = library_path / DB_DIR / DB_NAME

        if db_path.exists():
            logger.info("Library database already exists at %s", db_path)
        else:
            # Creating the repository initialises the schema.
            SqliteMediaRepository(str(db_path))
            logger.info("Created library database at %s", db_path)

        return db_path
