"""RAG (Retrieval Augmented Generation) manager."""

from typing import Any, Dict, List, Optional
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


class RAGManager:
    """
    Manages RAG (Retrieval Augmented Generation) operations.

    Handles vector stores, embeddings, and document retrieval.
    """

    def __init__(self):
        self._collections: Dict[str, VectorStore] = {}
        self._embeddings: Optional[Embeddings] = None

    def register_collection(
        self,
        name: str,
        vector_store: VectorStore
    ) -> None:
        """
        Register a vector store collection.

        Args:
            name: Collection identifier
            vector_store: Vector store instance
        """
        self._collections[name] = vector_store

    def set_embeddings(self, embeddings: Embeddings) -> None:
        """
        Set the embedding model.

        Args:
            embeddings: Embeddings instance
        """
        self._embeddings = embeddings

    async def retrieve(
        self,
        collection: str,
        query: str,
        k: int = 5,
        similarity_threshold: Optional[float] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents from a collection.

        Args:
            collection: Collection name
            query: Search query
            k: Number of documents to retrieve
            similarity_threshold: Minimum similarity score
            **kwargs: Additional retrieval parameters

        Returns:
            List of retrieved documents with metadata

        Raises:
            ValueError: If collection not found
        """
        if collection not in self._collections:
            raise ValueError(f"Collection '{collection}' not found")

        vector_store = self._collections[collection]

        # Perform similarity search
        if similarity_threshold is not None:
            results = await vector_store.asimilarity_search_with_relevance_scores(
                query,
                k=k,
                **kwargs
            )
            # Filter by threshold
            results = [
                (doc, score) for doc, score in results
                if score >= similarity_threshold
            ]
        else:
            docs = await vector_store.asimilarity_search(query, k=k, **kwargs)
            results = [(doc, None) for doc in docs]

        # Format results
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'score': score
            })

        return formatted_results

    async def add_documents(
        self,
        collection: str,
        documents: List[Dict[str, Any]]
    ) -> None:
        """
        Add documents to a collection.

        Args:
            collection: Collection name
            documents: List of documents with 'content' and optional 'metadata'

        Raises:
            ValueError: If collection not found
        """
        if collection not in self._collections:
            raise ValueError(f"Collection '{collection}' not found")

        vector_store = self._collections[collection]

        # Format documents for vector store
        from langchain_core.documents import Document

        docs = [
            Document(
                page_content=doc['content'],
                metadata=doc.get('metadata', {})
            )
            for doc in documents
        ]

        await vector_store.aadd_documents(docs)

    def get_collection(self, name: str) -> Optional[VectorStore]:
        """Get a vector store collection by name."""
        return self._collections.get(name)

    def list_collections(self) -> List[str]:
        """List all registered collections."""
        return list(self._collections.keys())


class RAGConfig:
    """Configuration for RAG setup."""

    def __init__(
        self,
        provider: str = "chroma",
        embedding_model: str = "openai/text-embedding-3-small",
        collections: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        """
        Initialize RAG configuration.

        Args:
            provider: Vector store provider (chroma, pinecone, etc.)
            embedding_model: Embedding model identifier
            collections: Collection configurations
        """
        self.provider = provider
        self.embedding_model = embedding_model
        self.collections = collections or {}

    def setup(self) -> RAGManager:
        """
        Set up RAG manager based on configuration.

        Returns:
            Configured RAGManager instance
        """
        manager = RAGManager()

        # Set up embeddings
        embeddings = self._create_embeddings()
        manager.set_embeddings(embeddings)

        # Set up collections
        for name, config in self.collections.items():
            vector_store = self._create_vector_store(embeddings, config)
            manager.register_collection(name, vector_store)

        return manager

    def _create_embeddings(self) -> Embeddings:
        """Create embeddings instance based on configuration."""
        if self.embedding_model.startswith('openai/'):
            from langchain_openai import OpenAIEmbeddings
            model_name = self.embedding_model.replace('openai/', '')
            return OpenAIEmbeddings(model=model_name)

        elif self.embedding_model.startswith('anthropic/'):
            # Anthropic doesn't have native embeddings yet
            # Fall back to OpenAI
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings()

        else:
            raise ValueError(f"Unsupported embedding model: {self.embedding_model}")

    def _create_vector_store(
        self,
        embeddings: Embeddings,
        config: Dict[str, Any]
    ) -> VectorStore:
        """Create vector store instance based on provider."""
        if self.provider == "chroma":
            from langchain_chroma import Chroma
            return Chroma(
                collection_name=config.get('name', 'default'),
                embedding_function=embeddings,
                persist_directory=config.get('persist_directory', './chroma_db')
            )

        elif self.provider == "pinecone":
            from langchain_pinecone import PineconeVectorStore
            return PineconeVectorStore(
                index_name=config.get('index_name', 'default'),
                embedding=embeddings
            )

        else:
            raise ValueError(f"Unsupported vector store provider: {self.provider}")
