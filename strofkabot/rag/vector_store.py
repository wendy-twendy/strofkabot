# Vector store implementation using LanceDB

"""LanceDB vector store wrapper for RAG.

Provides a simple interface for:
- Adding conversation chunks with embeddings and metadata
- Semantic search with optional metadata filtering
- Persistence across restarts

Note: Originally planned for ChromaDB, but using LanceDB due to
Python 3.14 compatibility (ChromaDB depends on onnxruntime which
doesn't have wheels for Python 3.14 yet).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import lancedb

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result from the vector store.

    Attributes:
        chunk_id: Unique identifier for the chunk.
        document: The stored document text.
        metadata: Associated metadata dictionary.
        distance: L2 distance from query (lower = more similar).
    """

    chunk_id: str
    document: str
    metadata: dict
    distance: float

    @property
    def similarity(self) -> float:
        """Convert distance to similarity score (1 - distance)."""
        return 1.0 - self.distance


class VectorStore:
    """LanceDB-backed vector store for conversation chunks.

    Args:
        persist_dir: Directory for LanceDB persistence.
        collection_name: Name of the table (collection).
    """

    def __init__(
        self,
        persist_dir: Path,
        collection_name: str = "strofka_messages",
    ):
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name

        # Create directory if needed
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Initialize LanceDB connection
        self._db = lancedb.connect(str(self.persist_dir))

        # Track whether table exists
        self._table = None
        if collection_name in self._db.table_names():
            self._table = self._db.open_table(collection_name)

    def add_chunks(
        self,
        chunk_ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        """Add chunks to the vector store.

        If a chunk_id already exists, it will be updated (upsert behavior).

        Args:
            chunk_ids: Unique identifiers for each chunk.
            embeddings: Vector embeddings for each chunk.
            documents: Document text for each chunk.
            metadatas: Metadata dictionaries for each chunk.
        """
        if not chunk_ids:
            return

        # Build records for LanceDB
        records = []
        for i, chunk_id in enumerate(chunk_ids):
            record = {
                "id": chunk_id,
                "vector": embeddings[i],
                "document": documents[i],
                **metadatas[i],
            }
            records.append(record)

        if self._table is None:
            # Create table with first batch
            self._table = self._db.create_table(
                self.collection_name,
                data=records,
                mode="overwrite",
            )
        else:
            # Upsert: delete existing IDs first, then add new
            existing_ids = set(self._get_all_ids())
            ids_to_delete = [cid for cid in chunk_ids if cid in existing_ids]

            if ids_to_delete:
                # Handle single ID case (tuple with trailing comma is invalid SQL)
                if len(ids_to_delete) == 1:
                    self._table.delete(f"id = '{ids_to_delete[0]}'")
                else:
                    self._table.delete(f"id IN {tuple(ids_to_delete)!r}")

            self._table.add(records)

        logger.debug(f"Added {len(chunk_ids)} chunks to vector store")

    def _get_all_ids(self) -> list[str]:
        """Get all chunk IDs in the store."""
        if self._table is None:
            return []
        return self._table.to_pandas()["id"].tolist()

    def search(
        self,
        query_embedding: list[float],
        k: int = 10,
        where: dict | None = None,
    ) -> list[SearchResult]:
        """Search for similar chunks.

        Args:
            query_embedding: Query vector to search for.
            k: Maximum number of results to return.
            where: Optional metadata filter (dict with field: value or ChromaDB-style $and).

        Returns:
            List of SearchResult objects, sorted by similarity (most similar first).
        """
        if self._table is None or self.count() == 0:
            return []

        # Limit k to actual count
        actual_k = min(k, self.count())

        # Build search query
        query = self._table.search(query_embedding).limit(actual_k)

        # Apply filters if provided
        if where:
            filter_str = self._build_filter(where)
            if filter_str:
                query = query.where(filter_str)

        try:
            results_df = query.to_pandas()
        except Exception as e:
            logger.warning(f"Search query failed: {e}")
            return []

        # Convert to SearchResult objects
        search_results = []
        for _, row in results_df.iterrows():
            # Extract metadata (all columns except id, vector, document, _distance)
            metadata = {
                k: v for k, v in row.items() if k not in ("id", "vector", "document", "_distance")
            }

            search_results.append(
                SearchResult(
                    chunk_id=row["id"],
                    document=row["document"],
                    metadata=metadata,
                    distance=row["_distance"],
                )
            )

        return search_results

    def _build_filter(self, where: dict) -> str:
        """Convert ChromaDB-style where clause to LanceDB SQL filter.

        Supports:
        - Simple equality: {"field": "value"}
        - $and: {"$and": [{"field1": "value1"}, {"field2": "value2"}]}
        - $contains: {"field": {"$contains": "substring"}} for LIKE matching
        """
        if "$and" in where:
            conditions = where["$and"]
            parts = []
            for cond in conditions:
                part = self._build_single_condition(cond)
                if part:
                    parts.append(part)
            return " AND ".join(parts)
        else:
            return self._build_single_condition(where)

    def _build_single_condition(self, cond: dict) -> str:
        """Build a single filter condition."""
        parts = []
        for key, value in cond.items():
            if key.startswith("$"):
                continue  # Skip operators at top level
            if isinstance(value, dict):
                # Handle operators like $contains
                if "$contains" in value:
                    # Use LIKE for substring matching
                    substr = value["$contains"]
                    parts.append(f"{key} LIKE '%{substr}%'")
            elif isinstance(value, str):
                parts.append(f"{key} = '{value}'")
            else:
                parts.append(f"{key} = {value}")
        return " AND ".join(parts)

    def get_chunk(self, chunk_id: str) -> dict | None:
        """Retrieve a single chunk by ID.

        Args:
            chunk_id: The unique identifier of the chunk.

        Returns:
            Dictionary with 'id', 'document', and 'metadata' keys,
            or None if not found.
        """
        if self._table is None:
            return None

        try:
            df = self._table.to_pandas()
            matches = df[df["id"] == chunk_id]

            if matches.empty:
                return None

            row = matches.iloc[0]
            metadata = {k: v for k, v in row.items() if k not in ("id", "vector", "document")}

            return {
                "id": row["id"],
                "document": row["document"],
                "metadata": metadata,
            }
        except Exception as e:
            logger.warning(f"get_chunk failed: {e}")
            return None

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        """Delete chunks by ID.

        Args:
            chunk_ids: List of chunk IDs to delete.
        """
        if not chunk_ids or self._table is None:
            return

        try:
            # LanceDB requires tuple for IN clause
            if len(chunk_ids) == 1:
                self._table.delete(f"id = '{chunk_ids[0]}'")
            else:
                self._table.delete(f"id IN {tuple(chunk_ids)!r}")
            logger.debug(f"Deleted {len(chunk_ids)} chunks from vector store")
        except Exception as e:
            logger.warning(f"Delete failed: {e}")

    def clear(self) -> None:
        """Remove all chunks from the collection."""
        if self._table is not None:
            self._db.drop_table(self.collection_name)
            self._table = None
        logger.info(f"Cleared collection '{self.collection_name}'")

    def count(self) -> int:
        """Return the number of chunks in the store."""
        if self._table is None:
            return 0
        return len(self._table)

    def get_all_documents(self) -> list[tuple[str, str]]:
        """Return all (chunk_id, document) pairs for BM25 indexing.

        Returns:
            List of (chunk_id, document_text) tuples.
        """
        if self._table is None:
            return []

        df = self._table.to_pandas()
        return list(zip(df["id"].tolist(), df["document"].tolist(), strict=True))

    def update_metadata(
        self,
        chunk_ids: list[str],
        new_metadata: list[dict],
    ) -> int:
        """Update metadata for existing chunks without re-embedding.

        This allows adding LLM-extracted metadata after initial ingestion.
        Only updates specified fields; existing metadata is preserved.

        Note: LanceDB has fixed schema. New fields in new_metadata will be
        added to the table schema if they don't exist yet (requires table
        recreation on first new field).

        Args:
            chunk_ids: List of chunk IDs to update.
            new_metadata: List of metadata dicts (one per chunk_id).
                Only specified fields are updated; others preserved.

        Returns:
            Number of chunks successfully updated.
        """
        if not chunk_ids or self._table is None:
            return 0

        if len(chunk_ids) != len(new_metadata):
            raise ValueError("chunk_ids and new_metadata must have same length")

        # Check if we need to add new columns to schema
        existing_columns = set(self._table.schema.names)
        new_columns = set()
        for meta in new_metadata:
            new_columns.update(meta.keys())

        columns_to_add = new_columns - existing_columns

        if columns_to_add:
            # Need to recreate table with new schema
            logger.info(f"Adding new columns to schema: {columns_to_add}")
            self._add_columns_to_schema(columns_to_add, new_metadata)

        # Now update the rows
        df = self._table.to_pandas()
        updated_count = 0

        for chunk_id, meta_updates in zip(chunk_ids, new_metadata, strict=True):
            matches = df[df["id"] == chunk_id]
            if matches.empty:
                logger.warning(f"Chunk {chunk_id} not found, skipping")
                continue

            row = matches.iloc[0]

            # Build updated record (preserve existing fields)
            record = {
                "id": row["id"],
                "vector": row["vector"],
                "document": row["document"],
            }

            # Copy existing metadata
            for k, v in row.items():
                if k not in ("id", "vector", "document"):
                    record[k] = v

            # Merge new metadata (overwrites existing keys)
            record.update(meta_updates)

            # Delete old and add updated
            self._table.delete(f"id = '{chunk_id}'")
            self._table.add([record])
            updated_count += 1

        logger.info(f"Updated metadata for {updated_count} chunks")
        return updated_count

    def _add_columns_to_schema(self, columns: set[str], sample_metadata: list[dict]) -> None:
        """Add new columns to table schema by recreating the table.

        Args:
            columns: Set of new column names to add.
            sample_metadata: Sample metadata to infer types from.
        """
        # Read all existing data
        df = self._table.to_pandas()

        # Infer types for new columns from sample metadata
        type_map = {}
        for col in columns:
            for meta in sample_metadata:
                if col in meta:
                    val = meta[col]
                    if isinstance(val, bool):
                        type_map[col] = False  # default bool
                    elif isinstance(val, int):
                        type_map[col] = 0  # default int
                    elif isinstance(val, float):
                        type_map[col] = 0.0  # default float
                    else:
                        type_map[col] = ""  # default string
                    break

        # Add new columns with defaults
        for col in columns:
            default = type_map.get(col, "")
            df[col] = default

        # Recreate table
        self._db.drop_table(self.collection_name)
        self._table = self._db.create_table(
            self.collection_name,
            data=df.to_dict(orient="records"),
            mode="overwrite",
        )
