import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import os
from src.logging_config import setup_logger

logger = setup_logger(__name__)

class RAGEngine:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        logger.info(f"Loading RAG embedding model: {model_name}...")
        # Check for MPS (Apple Silicon GPU) availability for PyTorch/SentenceTransformers
        device = "cpu"
        try:
            import torch
            if torch.backends.mps.is_available():
                device = "mps"
                logger.info("RAG Engine using Apple Silicon GPU (MPS).")
            else:
                logger.info("RAG Engine using CPU (MPS not available).")
        except ImportError:
            logger.info("RAG Engine using CPU (torch not found/checked).")

        self.model = SentenceTransformer(model_name, device=device)
        self.chunks: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.embeddings: np.ndarray = None
        logger.info("RAG Engine initialized.")

    def add_document(self, text: str, filename: str, chunk_size=500, overlap=50):
        """
        Chunks the document and adds it to the vector store.
        """
        logger.info(f"Adding document: {filename}")
        new_chunks = self._chunk_text(text, chunk_size, overlap)
        
        if not new_chunks:
            return

        # Embed new chunks
        logger.info(f"Embedding {len(new_chunks)} chunks...")
        new_embeddings = self.model.encode(new_chunks)
        
        # Update storage
        self.chunks.extend(new_chunks)
        self.metadatas.extend([{"filename": filename, "chunk_id": i} for i in range(len(new_chunks))])
        
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
            
        logger.info(f"Total chunks in store: {len(self.chunks)}")

    def retrieve(self, query: str, k=5, threshold=0.3) -> List[str]:
        """
        Retrieves the top-k most relevant chunks for the query.
        Filters out chunks with cosine similarity < threshold.
        """
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        # Embed query
        query_embedding = self.model.encode([query])[0]
        
        # Calculate cosine similarity
        # (a . b) / (|a| * |b|)
        # sentence-transformers embeddings are typically normalized, but let's be safe
        scores = np.dot(self.embeddings, query_embedding)
        norm_doc = np.linalg.norm(self.embeddings, axis=1)
        norm_query = np.linalg.norm(query_embedding)
        
        cosine_scores = scores / (norm_doc * norm_query)
        
        # Get top-k indices
        top_k_indices = np.argsort(cosine_scores)[-k:][::-1]
        
        results = []
        for idx in top_k_indices:
            score = cosine_scores[idx]
            if score >= threshold:
                logger.debug(f"Chunk {idx} Score: {score:.4f} (Accepted)")
                results.append(self.chunks[idx])
            else:
                logger.debug(f"Chunk {idx} Score: {score:.4f} (Rejected < {threshold})")
            
        return results

    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Simple sliding window chunking by words.
        """
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks
