"""Unified RAG Manager facade for VAY project."""

from vay.rag.manager_admin import inspect_source, list_categories, list_sources, update, delete
from vay.rag.manager_create import create, create_from_markdown, save_markdown
from vay.rag.manager_read import read

__all__ = [
    "create",
    "create_from_markdown",
    "save_markdown",
    "read",
    "list_sources",
    "list_categories",
    "update",
    "delete",
    "inspect_source",
]
