from typing import List, Dict, Any
import gc
from src.rag_engine import RAGEngine
from src.knowledge_graph import LocalGraphRAG

import threading
from src.logging_config import setup_logger

logger = setup_logger(__name__)

class ContextManager:
    def __init__(self, tokenizer, model=None, lock: threading.Lock = None, max_tokens=10000, system_prompt="You are a helpful AI assistant."):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.history: List[Dict[str, str]] = []
        
        # RAG Engine
        self.rag = RAGEngine()
        
        # Knowledge Graph
        self.graph_rag = None
        if model:
            self.graph_rag = LocalGraphRAG(model, tokenizer, lock=lock)
        
        # Global "Mental Model" Summary
        self.document_summary: str = ""

    def set_system_prompt(self, prompt: str):
        self.system_prompt = prompt

    def set_document_summary(self, summary: str):
        """
        Updates the global document summary (Mental Model).
        """
        self.document_summary = summary

    def add_file_context(self, text: str, filename: str = "Unknown File"):
        """
        Adds file content to the RAG engine and Knowledge Graph.
        """
        self.rag.add_document(text, filename)
        
        # Also extract knowledge graph (simplified: just first chunk for now to save time)
        # In production, we'd batch this.
        if self.graph_rag:
            logger.info(f"Extracting knowledge graph from {filename}...")
            # Take first 2000 chars as a sample for the graph
            self.graph_rag.extract_knowledge(text[:2000])

    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})

    def get_messages_for_inference(self) -> List[Dict[str, str]]:
        """
        Constructs the message list for the LLM using Hybrid Architecture.
        Structure:
        1. System Prompt (Always present)
        2. Global Document Summary (The "Mental Model")
        3. Retrieved Context (Dynamic RAG)
        4. Knowledge Graph Context (Relational)
        5. Recent History (Sliding window)
        """
        
        # 1. Start with System Prompt
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # 2. Add Global Document Summary (if available)
        if self.document_summary:
            messages.append({
                "role": "system",
                "content": f"=== CURRENT DOCUMENT MENTAL MODEL ===\nThe following is a high-level summary of the uploaded documents. Use this for global context:\n{self.document_summary}\n====================================="
            })
        
        # Get the last user message to use as a query
        last_user_msg = next((msg["content"] for msg in reversed(self.history) if msg["role"] == "user"), None)
        
        if last_user_msg:
            logger.debug(f"Retrieving context for query: {last_user_msg[:50]}...")
            
            # 3. Retrieve Context (Vector RAG)
            # Use threshold=0.3 to filter out irrelevant chunks
            retrieved_chunks = self.rag.retrieve(last_user_msg, k=5, threshold=0.3)
            
            if retrieved_chunks:
                logger.debug(f"Found {len(retrieved_chunks)} relevant chunks.")
                context_text = "\n\n---\n".join(retrieved_chunks)
                messages.append({
                    "role": "system", 
                    "content": f"=== RETRIEVED DOCUMENT CONTEXT ===\nThe following details were found in the uploaded documents and may be relevant:\n{context_text}\n=================================="
                })
            else:
                logger.debug("No relevant context found.")
                # Fallback: If query implies summarization and RAG failed, inject first few chunks
                if "summar" in last_user_msg.lower() and self.rag.chunks:
                    logger.info("RAG failed for summary query. Injecting first 3 chunks as fallback.")
                    fallback_text = "\n\n---\n".join(self.rag.chunks[:3])
                    messages.append({
                        "role": "system", 
                        "content": f"=== DOCUMENT BEGINNING (FALLBACK) ===\nThe user asked for a summary but RAG returned no specific chunks. Here is the beginning of the document:\n{fallback_text}\n====================================="
                    })

            # 4. Retrieve Context (Graph RAG)
            if self.graph_rag:
                graph_context = self.graph_rag.query_subgraph(last_user_msg)
                if "No related" not in graph_context:
                    logger.debug("Found Knowledge Graph connections.")
                    messages.append({
                        "role": "system",
                        "content": f"=== KNOWLEDGE GRAPH CONNECTIONS ===\n{graph_context}\n==================================="
                    })

        # 5. Add History (Sliding Window)
        # We work backwards from the most recent message
        logger.debug("Counting current tokens...")
        current_tokens = self._count_tokens(messages)
        logger.debug(f"Current tokens (System+RAG+Graph): {current_tokens}")
        
        history_tokens = 0
        included_history = []
        
        # Reserve some buffer for the response (e.g., 2048 tokens)
        available_tokens = self.max_tokens - current_tokens - 2048
        logger.debug(f"Available tokens for history: {available_tokens}")
                
        for msg in reversed(self.history):
            msg_tokens = self._count_tokens([msg])
            if history_tokens + msg_tokens <= available_tokens:
                included_history.insert(0, msg)
                history_tokens += msg_tokens
            else:
                break # Stop adding older messages
        
        return messages + included_history

    def _count_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        Approximate token count using the tokenizer.
        """
        count = 0
        for msg in messages:
            count += len(self.tokenizer.encode(msg["content"]))
        return count

    def get_token_usage_stats(self):
        messages = self.get_messages_for_inference()
        total = self._count_tokens(messages)
        return {
            "total": total,
            "limit": self.max_tokens,
            "percent": (total / self.max_tokens) * 100
        }
