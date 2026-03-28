"""
retriever.py — Dual-Mode Retrieval Module (FAISS + TF-IDF)

Provides semantic search via FAISS embeddings and keyword-based search
via TF-IDF, with a unified interface for switching between them.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseRetriever(ABC):
    """Common interface for all retriever implementations."""

    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """Index a list of LangChain Documents."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """Return the top-k documents most relevant to *query*."""

    @abstractmethod
    def is_ready(self) -> bool:
        """True when at least one document has been indexed."""


# ---------------------------------------------------------------------------
# FAISS (semantic / dense)
# ---------------------------------------------------------------------------

class FAISSRetriever(BaseRetriever):
    """
    Dense retriever backed by OpenAI embeddings + FAISS.

    Parameters
    ----------
    embedding_model : str
        OpenAI embedding model name (default: text-embedding-3-small).
    """

    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.vector_store: Optional[FAISS] = None
        self._doc_count = 0

    def add_documents(self, documents: List[Document]) -> None:
        if not documents:
            raise ValueError("Cannot index an empty document list")

        logger.info("Building FAISS index from %d chunks …", len(documents))
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        self._doc_count = len(documents)
        logger.info("FAISS index ready (%d vectors)", self._doc_count)

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        if not self.is_ready():
            raise RuntimeError("FAISS index has not been built yet")

        top_k = min(top_k, self._doc_count)
        results = self.vector_store.similarity_search(query, k=top_k)
        logger.debug("FAISS returned %d results for query: %.60s…", len(results), query)
        return results

    def is_ready(self) -> bool:
        return self.vector_store is not None


# ---------------------------------------------------------------------------
# TF-IDF (keyword / sparse)
# ---------------------------------------------------------------------------

class TFIDFRetriever(BaseRetriever):
    """
    Sparse retriever using scikit-learn TF-IDF + cosine similarity.

    Good baseline and works without any external API calls.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=10_000,
            ngram_range=(1, 2),
        )
        self.tfidf_matrix = None
        self.documents: List[Document] = []

    def add_documents(self, documents: List[Document]) -> None:
        if not documents:
            raise ValueError("Cannot index an empty document list")

        self.documents = documents
        texts = [doc.page_content for doc in documents]

        logger.info("Fitting TF-IDF on %d chunks …", len(texts))
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        logger.info("TF-IDF matrix: %s", self.tfidf_matrix.shape)

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        if not self.is_ready():
            raise RuntimeError("TF-IDF index has not been built yet")

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_k = min(top_k, len(self.documents))
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = [self.documents[i] for i in top_indices if scores[i] > 0]
        logger.debug("TF-IDF returned %d results for query: %.60s…", len(results), query)
        return results

    def is_ready(self) -> bool:
        return self.tfidf_matrix is not None


# ---------------------------------------------------------------------------
# Unified manager
# ---------------------------------------------------------------------------

class RetrieverManager:
    """
    Facade that holds both retrievers, indexes documents into both,
    and dispatches queries to whichever is currently selected.

    Parameters
    ----------
    default_method : str
        "faiss" or "tfidf" — the retriever used when no override is given.
    embedding_model : str
        Passed through to FAISSRetriever.
    """

    VALID_METHODS = {"faiss", "tfidf"}

    def __init__(
        self,
        default_method: str = "faiss",
        embedding_model: str = "text-embedding-3-small",
    ):
        if default_method not in self.VALID_METHODS:
            raise ValueError(f"method must be one of {self.VALID_METHODS}")

        self.default_method = default_method
        self.faiss_retriever = FAISSRetriever(embedding_model=embedding_model)
        self.tfidf_retriever = TFIDFRetriever()

    def index_documents(self, documents: List[Document]) -> dict:
        """
        Build both FAISS and TF-IDF indexes from the same chunks.

        Returns a summary dict with index stats.
        """
        errors = {}

        # Always build TF-IDF (no API cost)
        try:
            self.tfidf_retriever.add_documents(documents)
        except Exception as exc:
            logger.error("TF-IDF indexing failed: %s", exc)
            errors["tfidf"] = str(exc)

        # Build FAISS (requires OpenAI key)
        try:
            self.faiss_retriever.add_documents(documents)
        except Exception as exc:
            logger.warning("FAISS indexing failed (will fall back to TF-IDF): %s", exc)
            errors["faiss"] = str(exc)
            if self.default_method == "faiss":
                self.default_method = "tfidf"

        return {
            "total_chunks": len(documents),
            "faiss_ready": self.faiss_retriever.is_ready(),
            "tfidf_ready": self.tfidf_retriever.is_ready(),
            "active_method": self.default_method,
            "errors": errors or None,
        }

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        method: Optional[str] = None,
    ) -> List[Document]:
        """
        Retrieve relevant chunks for *query*.

        Parameters
        ----------
        method : str | None
            Override the default retriever for this call.
        """
        method = method or self.default_method

        if method == "faiss" and self.faiss_retriever.is_ready():
            return self.faiss_retriever.retrieve(query, top_k)
        elif method == "tfidf" and self.tfidf_retriever.is_ready():
            return self.tfidf_retriever.retrieve(query, top_k)
        else:
            # Fallback: try whichever is available
            if self.faiss_retriever.is_ready():
                return self.faiss_retriever.retrieve(query, top_k)
            if self.tfidf_retriever.is_ready():
                return self.tfidf_retriever.retrieve(query, top_k)
            raise RuntimeError("No retriever index is available — load a PDF first")

    def set_method(self, method: str) -> str:
        """Switch default retrieval method. Returns the new active method."""
        if method not in self.VALID_METHODS:
            raise ValueError(f"method must be one of {self.VALID_METHODS}")
        self.default_method = method
        return method
