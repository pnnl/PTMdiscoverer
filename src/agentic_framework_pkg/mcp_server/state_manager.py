import asyncio  # For to_thread
import datetime  # For timestamps
import json
import os
from collections.abc import Mapping
from typing import Any, cast

import chromadb
from dotenv import load_dotenv

from ..logger_config import get_logger  # Use centralized logger

load_dotenv()

logger = get_logger(__name__)

# ChromaDB client instance and collection names
_client: chromadb.ClientAPI | None = None
_sessions_collection: chromadb.Collection | None = None
_embeddings_collection: chromadb.Collection | None = None

SESSIONS_COLLECTION_NAME = "mcp_sessions"
EMBEDDINGS_COLLECTION_NAME = "mcp_embeddings"
DEFAULT_CHROMA_DB_PATH = "./chroma_data"


def _result_to_dict(result: Any) -> dict[str, Any]:
    return dict(cast(Mapping[str, Any], result))


def _metadata_to_dict(metadata: Any) -> dict[str, Any]:
    return dict(cast(Mapping[str, Any], metadata))


def _get_chroma_client_sync() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        chroma_db_path_env = os.getenv("CHROMA_DB_PATH")

        if chroma_db_path_env:
            chroma_db_path = chroma_db_path_env
        else:
            chroma_db_path = DEFAULT_CHROMA_DB_PATH
            logger.info(f"CHROMA_DB_PATH not set, defaulting to persistent storage at: {chroma_db_path}")

        if chroma_db_path.lower() != ":memory:":
            logger.info(f"Initializing persistent ChromaDB client at: {chroma_db_path}")
            os.makedirs(chroma_db_path, exist_ok=True)  # Ensure directory exists
            _client = chromadb.PersistentClient(path=chroma_db_path)
        else:
            logger.info("Initializing in-memory ChromaDB client.")
            _client = chromadb.Client()  # Ephemeral client
    return _client


async def initialize_db():
    global _sessions_collection, _embeddings_collection

    client_instance = await asyncio.to_thread(_get_chroma_client_sync)

    logger.info(f"Getting or creating ChromaDB collection: {SESSIONS_COLLECTION_NAME}")
    _sessions_collection = await asyncio.to_thread(
        client_instance.get_or_create_collection, name=SESSIONS_COLLECTION_NAME
    )

    logger.info(f"Getting or creating ChromaDB collection: {EMBEDDINGS_COLLECTION_NAME}")
    _embeddings_collection = await asyncio.to_thread(
        client_instance.get_or_create_collection, name=EMBEDDINGS_COLLECTION_NAME
    )

    logger.info("ChromaDB initialized and collections are ready.")


def _get_sessions_collection() -> chromadb.Collection:
    if _sessions_collection is None:
        raise RuntimeError("Sessions collection not initialized. Call initialize_db first.")
    return _sessions_collection


def _get_embeddings_collection() -> chromadb.Collection:
    if _embeddings_collection is None:
        raise RuntimeError("Embeddings collection not initialized. Call initialize_db first.")
    return _embeddings_collection


async def create_session_if_not_exists(
    session_id: str, client_id: str | None = None, initial_context: dict[str, Any] | None = None
) -> None:
    collection = _get_sessions_collection()

    existing_session_results = await asyncio.to_thread(collection.get, ids=[session_id])

    if not existing_session_results["ids"]:  # Check if id list is empty
        current_time = datetime.datetime.utcnow().isoformat()
        metadata = {
            "client_id": client_id or "",  # Ensure not None
            "context_data": json.dumps(initial_context or {}),
            "created_at": current_time,
            "updated_at": current_time,
        }
        await asyncio.to_thread(collection.add, ids=[session_id], documents=[session_id], metadatas=[metadata])
        logger.info("Created MCP session: %s", session_id)
    else:
        logger.debug("MCP Session %s already exists. Not creating.", session_id)


async def get_session_context(session_id: str) -> dict[str, Any]:
    collection = _get_sessions_collection()
    session_data = _result_to_dict(
        await asyncio.to_thread(collection.get, ids=[session_id], include=cast(Any, ["metadatas"]))
    )

    if session_data["ids"] and session_data["metadatas"]:
        metadata = _metadata_to_dict(session_data["metadatas"][0])
        context_str = metadata.get("context_data")
        if isinstance(context_str, str) and context_str:
            try:
                return cast(dict[str, Any], json.loads(context_str))
            except json.JSONDecodeError:
                logger.error(
                    "Failed to decode JSON context for session %s. Returning empty context.", session_id, exc_info=True
                )
                return {}
    return {}  # Return empty dict if no session or context found


async def update_session_context(session_id: str, new_context_data: dict[str, Any]) -> None:
    collection = _get_sessions_collection()
    current_time = datetime.datetime.utcnow().isoformat()

    # Fetch existing data to preserve fields not being updated (client_id, created_at) during upsert.
    existing_data_results = _result_to_dict(
        await asyncio.to_thread(collection.get, ids=[session_id], include=cast(Any, ["metadatas"]))
    )

    final_metadata = {"context_data": json.dumps(new_context_data), "updated_at": current_time}

    if existing_data_results["ids"] and existing_data_results["metadatas"]:
        # Session exists, it's an update. Preserve existing client_id and created_at.
        existing_metadata = _metadata_to_dict(existing_data_results["metadatas"][0])
        final_metadata["client_id"] = existing_metadata.get("client_id", "")
        final_metadata["created_at"] = existing_metadata.get("created_at", current_time)  # Should ideally exist
    else:
        # Session does not exist, it's an insert part of an upsert.
        # client_id is not provided to this function, so default to empty string.
        final_metadata["client_id"] = ""
        final_metadata["created_at"] = current_time

    await asyncio.to_thread(collection.upsert, ids=[session_id], documents=[session_id], metadatas=[final_metadata])
    logger.debug("Upserted context for MCP session: %s", session_id)


async def delete_session(session_id: str) -> None:
    sessions_coll = _get_sessions_collection()
    embeddings_coll = _get_embeddings_collection()

    await asyncio.to_thread(sessions_coll.delete, ids=[session_id])
    logger.info("Deleted MCP session: %s", session_id)

    # Delete associated embeddings (mimicking ON DELETE CASCADE)
    # This requires embeddings to have a "session_id" in their metadata.
    await asyncio.to_thread(embeddings_coll.delete, where={"session_id": session_id})
    logger.info(f"Attempted deletion of associated embeddings for session: {session_id} using where clause.")


# --- Generic Embedding Management Functions ---


async def add_embeddings_batch(
    ids: list[str], documents: list[str], embeddings: list[list[float]], metadatas: list[dict[str, Any]]
):
    """Adds a batch of embeddings to the embeddings collection."""
    collection = _get_embeddings_collection()
    await asyncio.to_thread(
        collection.add,
        ids=ids,
        embeddings=cast(Any, embeddings),
        documents=documents,
        metadatas=cast(Any, metadatas),
    )
    logger.debug(f"Added batch of {len(ids)} embeddings.")


async def get_embeddings(
    ids: list[str] | None = None, where: dict[str, Any] | None = None, include: list[str] | None = None
) -> dict[str, Any]:
    """Generic get function for embeddings from the embeddings collection."""
    collection = _get_embeddings_collection()
    if include is None:
        include = ["metadatas", "documents", "embeddings"]  # Default fields to retrieve

    results = await asyncio.to_thread(collection.get, ids=ids, where=where, include=cast(Any, include))
    return _result_to_dict(results)


async def query_embeddings(
    query_embeddings_list: list[list[float]],
    n_results: int = 5,
    where: dict[str, Any] | None = None,
    where_document: dict[str, Any] | None = None,
    include: list[str] | None = None,
) -> dict[str, Any]:
    """Generic query function for embeddings from the embeddings collection."""
    collection = _get_embeddings_collection()
    if include is None:
        include = ["metadatas", "documents", "distances"]  # Default fields for query results

    results = await asyncio.to_thread(
        collection.query,
        query_embeddings=cast(Any, query_embeddings_list),
        n_results=n_results,
        where=where,
        where_document=cast(Any, where_document),
        include=cast(Any, include),
    )
    return _result_to_dict(results)


async def delete_embeddings(ids: list[str] | None = None, where: dict[str, Any] | None = None) -> None:
    """Deletes embeddings from the embeddings collection based on ids or a where filter."""
    collection = _get_embeddings_collection()
    await asyncio.to_thread(collection.delete, ids=ids, where=where)
    logger.info(f"Attempted deletion of embeddings with ids: {ids} or where filter: {where}")


async def clear_embeddings() -> None:
    """Clears all embeddings from the embeddings collection."""
    collection = _get_embeddings_collection()
    await asyncio.to_thread(collection.delete, where={})  # Deletes all entries
    logger.info("Cleared all embeddings from the collection.")


async def get_all_embeddings() -> dict[str, Any]:
    """Retrieves all embeddings from the embeddings collection."""
    collection = _get_embeddings_collection()
    results = await asyncio.to_thread(collection.get, where={})  # Get all entries
    return _result_to_dict(results)


async def get_all_sessions() -> dict[str, Any]:
    """Retrieves all sessions from the sessions collection."""
    collection = _get_sessions_collection()
    results = await asyncio.to_thread(collection.get, where={})  # Get all entries
    return _result_to_dict(results)


async def clear_sessions() -> None:
    """Clears all sessions from the sessions collection."""
    collection = _get_sessions_collection()
    await asyncio.to_thread(collection.delete, where={})  # Deletes all entries
    logger.info("Cleared all sessions from the collection.")


async def get_session_ids() -> list[str]:
    """Retrieves all session IDs from the sessions collection."""
    collection = _get_sessions_collection()
    results = _result_to_dict(await asyncio.to_thread(collection.get, where={}))
    return cast(list[str], results.get("ids", []))  # Return list of session IDs or empty list if none


async def get_session_metadata(session_id: str) -> dict[str, Any]:
    """Retrieves metadata for a specific session by ID."""
    collection = _get_sessions_collection()
    results = _result_to_dict(
        await asyncio.to_thread(collection.get, ids=[session_id], include=cast(Any, ["metadatas"]))
    )

    if results["ids"] and results["metadatas"]:
        return _metadata_to_dict(results["metadatas"][0])
    return {}  # Return empty dict if no session found


async def update_session_metadata(session_id: str, new_metadata: dict[str, Any]) -> None:
    """Updates metadata for a specific session by ID."""
    collection = _get_sessions_collection()

    # Fetch existing metadata to preserve fields not being updated
    existing_data_results = _result_to_dict(
        await asyncio.to_thread(collection.get, ids=[session_id], include=cast(Any, ["metadatas"]))
    )

    if existing_data_results["ids"] and existing_data_results["metadatas"]:
        existing_metadata = _metadata_to_dict(existing_data_results["metadatas"][0])
        # Merge new metadata with existing, preserving existing fields
        final_metadata = {**existing_metadata, **new_metadata}
    else:
        final_metadata = new_metadata  # If no existing data, just use new metadata

    await asyncio.to_thread(collection.upsert, ids=[session_id], documents=[session_id], metadatas=[final_metadata])
    logger.debug("Updated metadata for MCP session: %s", session_id)


async def delete_session_metadata(session_id: str) -> None:
    """Deletes metadata for a specific session by ID."""
    collection = _get_sessions_collection()

    # Fetch existing metadata to ensure session exists
    existing_data_results = _result_to_dict(
        await asyncio.to_thread(collection.get, ids=[session_id], include=cast(Any, ["metadatas"]))
    )

    if existing_data_results["ids"] and existing_data_results["metadatas"]:
        # Session exists, proceed with deletion
        await asyncio.to_thread(collection.delete, ids=[session_id])
        logger.info("Deleted metadata for MCP session: %s", session_id)
    else:
        logger.warning("Attempted to delete metadata for non-existent session: %s", session_id)
