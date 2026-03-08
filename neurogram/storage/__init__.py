"""Storage backend abstract interface for Neurogram."""

from neurogram.storage.base import StorageBackend
from neurogram.storage.sqlite_backend import SQLiteBackend

__all__ = ["StorageBackend", "SQLiteBackend"]
