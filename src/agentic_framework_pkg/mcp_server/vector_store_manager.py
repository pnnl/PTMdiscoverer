import os
from collections.abc import Mapping
from typing import Any, cast

import chromadb

from agentic_framework_pkg.logger_config import get_logger

logger = get_logger(__name__)

CHROMA_DATA_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma_vsm_store")


class VectorStoreManager:
    _instance: "VectorStoreManager | None" = None
    client: chromadb.ClientAPI

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            logger.info(f"Initializing ChromaDB client with path: {CHROMA_DATA_PATH}")
            os.makedirs(CHROMA_DATA_PATH, exist_ok=True)
            cls._instance.client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
            logger.info("ChromaDB client initialized.")
        return cls._instance

    def get_or_create_collection(self, name: str, metadata: dict[str, Any] | None = None):
        try:
            collection = self.client.get_or_create_collection(name=name, metadata=metadata)
            logger.info(f"Accessed or created collection: {name}")
            return collection
        except Exception as e:
            logger.error(f"Error getting or creating collection '{name}': {e}", exc_info=True)
            raise

    def add_documents(
        self,
        collection_name: str,
        documents: list[str],
        embeddings: list[list[float]] | None = None,
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
    ):
        if not documents:
            logger.warning(f"No documents provided to add to collection '{collection_name}'.")
            return
        try:
            collection = self.get_or_create_collection(collection_name)
            if embeddings:
                collection.add(embeddings=embeddings, documents=documents, metadatas=metadatas, ids=ids)
            else:
                collection.add(documents=documents, metadatas=metadatas, ids=ids)
            logger.info(f"Added {len(documents)} documents to collection '{collection_name}'.")
        except Exception as e:
            logger.error(f"Error adding documents to collection '{collection_name}': {e}", exc_info=True)
            raise

    def query_collection(
        self,
        collection_name: str,
        query_texts: list[str] | None = None,
        query_embeddings: list[list[float]] | None = None,
        n_results: int = 5,
        where: dict | None = None,
        where_document: dict | None = None,
    ) -> dict[str, Any]:
        if not query_texts and not query_embeddings:
            raise ValueError("Either query_texts or query_embeddings must be provided.")
        try:
            collection = self.get_or_create_collection(collection_name)
            results = collection.query(
                query_texts=query_texts,
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where,
                where_document=where_document,
            )
            query_inputs = query_texts if query_texts is not None else (query_embeddings or [])
            results_dict = dict(cast(Mapping[str, Any], results))
            first_result_ids = results_dict.get("ids", [[]])
            logger.info(
                f"Queried collection '{collection_name}' with {len(query_inputs)} queries, got {len(first_result_ids[0])} results for the first query."
            )
            return results_dict
        except Exception as e:
            logger.error(f"Error querying collection '{collection_name}': {e}", exc_info=True)
            raise
