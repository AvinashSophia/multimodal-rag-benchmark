"""Production BM25 retriever backed by EKS-hosted Elasticsearch.

AWS variant of bm25_elastic.py — reads from retrieval.text.elasticsearch_aws
so local and EKS endpoints can coexist in the same codebase. All retrieval
logic is identical to BM25ElasticRetriever.

EKS service: elasticsearch  →  localhost:9200 (via port-forward)
Security is disabled on the cluster (internal use).
"""

from typing import Dict, List, Any, Optional
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from pipeline.retrieval.base import BaseRetriever, register_retriever
from pipeline.utils import RetrievalResult


@register_retriever("bm25_elastic_aws")
class BM25ElasticAWSRetriever(BaseRetriever):
    """Production BM25 retriever backed by EKS-hosted Elasticsearch.

    Identical to BM25ElasticRetriever but reads connection config from
    retrieval.text.elasticsearch_aws, allowing both local and EKS endpoints
    to be configured independently without conflict.

    Config keys (under retrieval.text.elasticsearch_aws):
        url:        ES endpoint   (default: http://localhost:9200)
        index:      Index name    (default: bm25_text)
        username:   Optional basic auth username
        password:   Optional basic auth password
        ca_cert:    Optional path to CA certificate for HTTPS

    EKS note: security is disabled on the cluster — username/password/ca_cert
    are not required for internal cluster use.
    """

    BATCH_SIZE = 500
    _SENTINEL_ID = "__bm25_index_metadata__"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        text_cfg = config["retrieval"]["text"]
        es_cfg = text_cfg.get("elasticsearch_aws", {})

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

    def is_indexed(self) -> bool:
        """Return True if the Elasticsearch index has a valid metadata sentinel.

        Returns True when:
          - sentinel exists and corpus_count > 0  (normal full index), OR
          - sentinel exists and not_applicable=True (dataset has no text corpus)
        """
        try:
            if not self.es.indices.exists(index=self.index_name):
                return False
            doc = self.es.get(index=self.index_name, id=self._SENTINEL_ID)
            if doc["_source"].get("not_applicable"):
                return True
            return (doc["_source"].get("corpus_count") or 0) > 0
        except Exception:
            return False

    def mark_not_applicable(self) -> None:
        """Mark this retriever as not applicable for the current dataset."""
        self._create_index()
        self.es.index(
            index=self.index_name,
            id=self._SENTINEL_ID,
            document={"type": "metadata", "corpus_count": 0, "not_applicable": True},
        )
        self.es.indices.refresh(index=self.index_name)
        print(f"  [BM25ElasticAWS] Marked '{self.index_name}' as not applicable for this dataset.")

    def _index_exists(self, expected_count: int) -> bool:
        """Return True if index sentinel exists with the expected corpus_count."""
        try:
            if not self.es.indices.exists(index=self.index_name):
                return False
            doc = self.es.get(index=self.index_name, id=self._SENTINEL_ID)
            return doc["_source"].get("corpus_count") == expected_count
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
                    "text":         {"type": "text",    "similarity": "default"},
                    "doc_id":       {"type": "keyword"},
                    "type":         {"type": "keyword"},
                    "corpus_count": {"type": "integer"},
                }
            },
        )

    # ------------------------------------------------------------------
    # BaseRetriever interface
    # ------------------------------------------------------------------

    def index(self, corpus: List[str], corpus_ids: Optional[List[str]] = None) -> None:
        """Index corpus into EKS Elasticsearch. Skips if already indexed."""
        ids = corpus_ids or [f"chunk_{i}" for i in range(len(corpus))]

        if self._index_exists(len(corpus)):
            print(f"  [BM25ElasticAWS] Index '{self.index_name}' already contains "
                  f"{len(corpus)} docs. Skipping indexing.")
            return

        print(f"  [BM25ElasticAWS] Indexing {len(corpus)} chunks into '{self.index_name}'...")
        self._create_index()

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

        # Write metadata sentinel — marks a successfully completed full-corpus index.
        self.es.index(
            index=self.index_name,
            id=self._SENTINEL_ID,
            document={"type": "metadata", "corpus_count": len(corpus)},
        )
        self.es.indices.refresh(index=self.index_name)

        print(f"  [BM25ElasticAWS] Indexed {len(corpus)} chunks into '{self.index_name}'.")

    def retrieve(self, query: str, top_k: int = 5, query_image: Optional[Any] = None) -> RetrievalResult:
        """Retrieve top-k chunks using BM25 scoring from EKS Elasticsearch. query_image is ignored."""
        response = self.es.search(
            index=self.index_name,
            query={
                "bool": {
                    "must": {"match": {"text": query}},
                    "must_not": {"term": {"type": "metadata"}},
                }
            },
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
            metadata={"method": "bm25_elastic_aws", "index": self.index_name, "query": query},
        )

    def storage_info(self) -> Dict[str, Any]:
        try:
            stats = self.es.indices.stats(index=self.index_name)
            idx = stats["indices"][self.index_name]
            docs = idx["total"]["docs"]["count"]
            size_mb = round(idx["total"]["store"]["size_in_bytes"] / (1024 * 1024), 2)
            return {"type": "elasticsearch", "index": self.index_name, "documents": docs, "size_mb": size_mb}
        except Exception:
            return {"type": "elasticsearch", "index": self.index_name}
