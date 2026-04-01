"""Production BM25 retriever backed by Elasticsearch."""

from typing import Dict, List, Any, Optional
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from pipeline.retrieval.base import BaseRetriever, register_retriever
from pipeline.utils import RetrievalResult


@register_retriever("bm25_elastic")
class BM25ElasticRetriever(BaseRetriever):
    """Production-grade BM25 retriever backed by Elasticsearch.

    Uses Elasticsearch's native BM25 ranking (the default scoring algorithm).
    Documents are indexed once and persisted in ES — subsequent runs skip
    indexing entirely if the index already contains the expected number of docs.

    Text content and IDs are stored as ES document fields so the retriever
    is fully self-contained after indexing.

    Config keys (under retrieval.text.elasticsearch):
        url:        ES endpoint (default: http://localhost:9200)
        index:      Index name (default: bm25_text)
        username:   Optional basic auth username
        password:   Optional basic auth password
        ca_cert:    Optional path to CA certificate for HTTPS

    Usage:
        Start Elasticsearch locally:
            docker run -d -p 9200:9200 -e "discovery.type=single-node" \\
                -e "xpack.security.enabled=false" elasticsearch:8.13.0
    """

    BATCH_SIZE = 500  # Bulk index in batches

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        text_cfg = config["retrieval"]["text"]
        es_cfg = text_cfg.get("elasticsearch", {})

        dataset_name = config.get("dataset", {}).get("name", "default")
        self.index_name = f"{es_cfg.get('index', 'bm25_text')}_{dataset_name}"
        url = es_cfg.get("url", "http://localhost:9200")
        username = es_cfg.get("username", "")
        password = es_cfg.get("password", "")
        ca_cert = es_cfg.get("ca_cert", None)

        client_kwargs: Dict[str, Any] = {"hosts": [url]}
        if username and password:
            client_kwargs["basic_auth"] = (username, password)
        if ca_cert:
            client_kwargs["ca_certs"] = ca_cert

        self.es = Elasticsearch(**client_kwargs)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _index_exists(self, expected_count: int) -> bool:
        """Return True if index exists with the expected number of documents."""
        try:
            if not self.es.indices.exists(index=self.index_name):
                return False
            self.es.indices.refresh(index=self.index_name)
            count = self.es.count(index=self.index_name)["count"]
            return count == expected_count
        except Exception:
            return False

    def _create_index(self) -> None:
        """Drop (if exists) and recreate the ES index with BM25 settings."""
        self.es.indices.delete(index=self.index_name, ignore_unavailable=True)

        self.es.indices.create(
            index=self.index_name,
            settings={
                "similarity": {
                    "default": {"type": "BM25", "k1": 1.2, "b": 0.75}
                },
                "number_of_shards": 1,
                "number_of_replicas": 0,
            },
            mappings={
                "properties": {
                    "text": {"type": "text", "similarity": "default"},
                    "doc_id": {"type": "keyword"},
                }
            },
        )

    # ------------------------------------------------------------------
    # BaseRetriever interface
    # ------------------------------------------------------------------

    def index(self, corpus: List[str], corpus_ids: Optional[List[str]] = None) -> None:
        """Index corpus into Elasticsearch. Skips if already indexed."""
        ids = corpus_ids or [f"chunk_{i}" for i in range(len(corpus))]

        if self._index_exists(len(corpus)):
            print(f"  [BM25Elastic] Index '{self.index_name}' already contains "
                  f"{len(corpus)} docs. Skipping indexing.")
            return

        print(f"  [BM25Elastic] Indexing {len(corpus)} chunks into '{self.index_name}'...")
        self._create_index()

        # Bulk index in batches
        for start in range(0, len(corpus), self.BATCH_SIZE):
            batch_texts = corpus[start: start + self.BATCH_SIZE]
            batch_ids   = ids[start: start + self.BATCH_SIZE]

            actions = [
                {
                    "_index": self.index_name,
                    "_id": doc_id,
                    "_source": {"text": text, "doc_id": doc_id},
                }
                for doc_id, text in zip(batch_ids, batch_texts)
            ]
            bulk(self.es, actions)

        self.es.indices.refresh(index=self.index_name)
        print(f"  [BM25Elastic] Indexed {len(corpus)} chunks into '{self.index_name}'.")

    def retrieve(self, query: str, top_k: int = 5, query_image: Optional[Any] = None) -> RetrievalResult:
        """Retrieve top-k chunks using BM25 scoring from Elasticsearch. query_image is ignored."""
        response = self.es.search(
            index=self.index_name,
            query={"match": {"text": query}},
            size=top_k,
        )

        hits = response["hits"]["hits"]

        return RetrievalResult(
            text_chunks=[hit["_source"]["text"] for hit in hits],
            text_scores=[hit["_score"] for hit in hits],
            text_ids=[hit["_source"]["doc_id"] for hit in hits],
            images=[],
            image_scores=[],
            image_ids=[],
            metadata={"method": "bm25_elastic", "index": self.index_name, "query": query},
        )
