import json
import logging
import time
from typing import Any, Dict, List, Optional

try:
    from opensearchpy import OpenSearch, RequestsHttpConnection, helpers, NotFoundError
except ImportError:
    raise ImportError("OpenSearch requires extra dependencies. Install with `pip install opensearch-py`") from None

from pydantic import BaseModel

from mem0.configs.vector_stores.lindorm_search import LindormSearchConfig
from mem0.vector_stores.base import VectorStoreBase

logger = logging.getLogger(__name__)


class OutputData(BaseModel):
    id: str
    score: float
    payload: Dict


class LindormSearch(VectorStoreBase):
    def __init__(self, **kwargs):
        config = LindormSearchConfig(**kwargs)

        # Initialize OpenSearch client
        auth = config.http_auth
        if auth is None and config.user and config.password:
            auth = (config.user, config.password)

        self.client = OpenSearch(
            hosts=[{"host": config.host, "port": config.port}],
            http_auth=auth,
            use_ssl=config.use_ssl,
            verify_certs=config.verify_certs,
            connection_class=RequestsHttpConnection,
            pool_maxsize=20,
            timeout=30,
            retry_on_timeout=True,
            max_retries=3,
        )

        self.collection_name = config.collection_name
        self.embedding_model_dims = config.embedding_model_dims
        self.create_col(self.collection_name, self.embedding_model_dims, config.distance_method)

    def create_col(self, name: str, vector_size: int, distance: str) -> None:
        """Create a new collection (index in OpenSearch)."""
        if distance not in ["cosinesimil", "l2", "innerproduct"]:
            distance = "cosinesimil"
        index_settings = {
            "settings": {
                "index.knn": True,
                "knn_routing": True,
            },
            "mappings": {
                "_source": {
                    "excludes": ["vector_field"]
                },
                "properties": {
                    "vector_field": {
                        "type": "knn_vector",
                        "dimension": vector_size,
                        "method": {
                            "engine": "lvector",
                            "name": "flat",
                            "space_type": distance
                        },
                    },
                    "payload": {"type": "object"}
                }
            },
        }
        if self.client.indices.exists(index=name):
            logger.info(f"Index {name} already exists")
            return

        def _wait_for_index_ready(collection: str, max_retries: int = 180, retry_interval: float = 0.5) -> None:
            for _ in range(max_retries):
                try:
                    self.client.search(index=collection, body={"query": {"match_all": {}}})
                    logger.info(f"Index {collection} is ready")
                    return
                except Exception:
                    time.sleep(retry_interval)
            raise TimeoutError(f"Index {collection} creation timed out after {max_retries * retry_interval} seconds")

        try:
            logger.info(f"Creating index {name}...")
            self.client.indices.create(index=name, body=index_settings)
            _wait_for_index_ready(name)
        except Exception as e:
            logger.error(f"Failed to create index {name}: {str(e)}")
            raise

    def insert(
            self, vectors: List[List[float]], payloads: Optional[List[Dict]] = None, ids: Optional[List[str]] = None
    ) -> List[Dict]:
        """Insert vectors into the index using bulk API."""
        if not ids:
            ids = [str(i) for i in range(len(vectors))]

        if payloads is None:
            payloads = [{} for _ in range(len(vectors))]

        actions = []
        for vec, payload, id_ in zip(vectors, payloads, ids):
            action = {
                "_op_type": "index",
                "_index": self.collection_name,
                "_routing": payload.get("user_id", "general"),
                "_id": id_,
                "_source": {
                    "vector_field": vec,
                    "payload": payload
                }
            }
            actions.append(action)
        try:
            success, failed = helpers.bulk(self.client, actions, stats_only=True)
            results = [{"id": id_, "status": "success"} for id_ in ids[:success]]
            if failed:
                results.extend([{"id": id_, "status": "failed"} for id_ in ids[success:]])
            return results
        except Exception as e:
            logger.error(f"Bulk insert failed: {str(e)}")
            raise

    def search(
            self,
            query: str,
            vectors: List[float],
            limit: int = 5,
            filters: Optional[Dict] = None,
    ) -> List[OutputData]:
        knn_query = {
            "knn": {
                "vector_field": {
                    "vector": vectors,
                    "k": limit,
                }
            }
        }

        final_filter_clauses = []
        if query:
            hybrid_search_must_clauses = [{"match": {"payload.data": query}}]
            routing_value = filters.get("user_id", "general") if filters else "general"
            if routing_value:
                hybrid_search_must_clauses.append({"term": {"_routing": routing_value}})
            final_filter_clauses.append({"bool": {"must": hybrid_search_must_clauses}})

        scalar_filter_clauses = []
        if filters:
            for key in ["user_id", "run_id", "agent_id"]:
                value = filters.get(key)
                if value:
                    field_name = f"payload.{key}.keyword"
                    if isinstance(value, list):
                        scalar_filter_clauses.append({"terms": {field_name: value}})
                    else:
                        scalar_filter_clauses.append({"term": {field_name: value}})

        if scalar_filter_clauses:
            final_filter_clauses.append({"bool": {"filter": scalar_filter_clauses}})

        if final_filter_clauses:
            knn_query["knn"]["vector_field"]["filter"] = {
                "bool": {
                    "must": final_filter_clauses
                }
            }

        query_body = {
            "size": limit,
            "query": knn_query,
            "_source": {"excludes": ["vector_field"]},
            "ext": {
                "lvector": {
                    "hybrid_search_type": "filter_rrf",
                    "rrf_rank_constant": "60",
                    "rrf_knn_weight_factor": "0.5",
                    "filter_type": "efficient_filter"
                }
            }
        }

        try:
            routing = filters.get("user_id", "general") if filters else "general"
            response = self.client.search(
                index=self.collection_name,
                body=query_body,
                routing=routing,
            )
            return [
                OutputData(id=hit["_id"], score=hit["_score"], payload=hit["_source"].get("payload", {}))
                for hit in response["hits"]["hits"]
            ]
        except Exception as e:
            logger.error(f"Search failed. Query body: {json.dumps(query_body, indent=2)}")
            logger.error(f"Exception: {str(e)}")
            return []

    def delete(self, vector_id: str) -> None:
        """Delete a vector by custom ID."""
        # Delete using the actual document ID
        record = self.get(vector_id)
        if record is None:
            return
        routing = record.payload.get("user_id", "general")
        self.client.delete(index=self.collection_name, id=vector_id, routing=routing)

    def update(self, vector_id: str, vector: Optional[List[float]] = None, payload: Optional[Dict] = None) -> None:
        """Update a vector and its payload using the vector_id directly."""
        doc = {}
        if vector is not None:
            doc["vector_field"] = vector
        if payload is not None:
            doc["payload"] = payload

        if not doc:
            return None
        try:
            self.client.update(
                index=self.collection_name,
                id=vector_id,
                body={"doc": doc},
                retry_on_conflict=3
            )
            return
        except NotFoundError:
            raise

    def get(self, vector_id: str) -> Optional[OutputData]:
        """Retrieve a vector by ID."""
        try:
            # Check if index exists
            if not self.client.indices.exists(index=self.collection_name):
                logger.info(f"Index {self.collection_name} does not exist.")
                return None

            # Directly get the document by ID
            response = self.client.get(index=self.collection_name, id=vector_id)

            source = response['_source']
            return OutputData(
                id=vector_id,
                score=1.0,
                payload=source.get('payload', {})
            )

        except NotFoundError:
            logger.info(f"Vector with ID {vector_id} not found in index {self.collection_name}.")
            return None
        except Exception as e:
            logger.error(f"Error retrieving vector {vector_id} from index {self.collection_name}: {str(e)}")
            return None

    def list_cols(self) -> List[str]:
        """List all collections (indices)."""
        return list(self.client.indices.get_alias().keys())

    def delete_col(self) -> None:
        """Delete a collection (index)."""
        self.client.indices.delete(index=self.collection_name)

    def col_info(self) -> Any:
        """Get information about a collection (index)."""
        return self.client.indices.get(index=self.collection_name)

    def list(self, filters: Optional[Dict] = None, limit: Optional[int] = None) -> List[OutputData]:
        """
        List all memories with optional filters.

        :param filters: Optional dictionary of filters
        :param limit: Optional limit on the number of results
        :return: List of OutputData objects
        """
        try:
            query = self._build_list_query(filters)
            body = {
                "query": query,
                "_source": {"excludes": ["vector_field"]},
                "sort": [{"_score": {"order": "desc"}}]
            }

            if limit:
                body["size"] = limit

            response = self.client.search(index=self.collection_name, body=body)
            hits = response["hits"]["hits"]

            results = [
                    OutputData(
                        id=hit["_id"],
                        score=hit["_score"],
                        payload=hit["_source"].get("payload", {})
                    )
                    for hit in hits
            ]
            return results
        except Exception as e:
            logger.error(f"Error in list operation: {str(e)}")
            return []

    def _build_list_query(self, filters: Optional[Dict]) -> Dict:
        """Build the query for list operation based on filters."""
        if not filters:
            return {"match_all": {}}

        must_clauses = []
        for key, value in filters.items():
            if key in ["user_id", "run_id", "agent_id"]:
                if isinstance(value, list):
                    must_clauses.append({"terms": {f"payload.{key}.keyword": value}})
                else:
                    must_clauses.append({"term": {f"payload.{key}.keyword": value}})
        return {"bool": {"must": must_clauses}} if must_clauses else {"match_all": {}}

    def reset(self):
        """Reset the index by deleting all existing data."""
        logger.warning(f"Resetting index {self.collection_name}...")
        if self.client.indices.exists(index=self.collection_name):
            # Delete all documents in the index using delete_by_query
            delete_query = {"query": {"match_all": {}}}
            try:
                response = self.client.delete_by_query(
                    index=self.collection_name,
                    body=delete_query,
                    refresh=True
                )
                self.refresh()
                logger.info(f"Deleted {response.get('deleted', 0)} documents from index {self.collection_name}")
            except Exception as e:
                logger.error(f"Failed to delete documents from index {self.collection_name}: {str(e)}")
                raise
        else:
            logger.info(f"Index {self.collection_name} does not exist, nothing to reset")

    def refresh(self):
        """Refresh the index to make recent operations visible for search."""
        try:
            self.client.indices.refresh(index=self.collection_name)
            logger.info(f"Index {self.collection_name} refreshed successfully.")
        except Exception as e:
            logger.error(f"Failed to refresh index {self.collection_name}: {str(e)}")
            raise
