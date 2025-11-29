from typing import List, Dict, Any
import gc
from src.rag_engine import RAGEngine

class ContextManager:
    def __init__(self, tokenizer, max_tokens=10000, system_prompt="You are a helpful AI assistant."):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.history: List[Dict[str, str]] = []
        
        # RAG Engine
        self.rag = RAGEngine()
        
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
        Adds file content to the RAG engine.
        """
        self.rag.add_document(text, filename)

    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})

    def get_messages_for_inference(self) -> List[Dict[str, str]]:
        """
        Constructs the message list for the LLM using Hybrid Architecture.
        Structure:
        1. System Prompt (Always present)
        2. Global Document Summary (The "Mental Model")
        3. Retrieved Context (Dynamic RAG)
        4. Recent History (Sliding window)
        """
        
        # 1. Start with System Prompt
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # 2. Add Global Document Summary (if available)
        if self.document_summary:
            messages.append({
                "role": "system",
                "content": f"=== CURRENT DOCUMENT MENTAL MODEL ===\nThe following is a high-level summary of the uploaded documents. Use this for global context:\n{self.document_summary}\n====================================="
            })
        
        # 3. Retrieve Context (RAG)
        # Get the last user message to use as a query
        last_user_msg = next((msg["content"] for msg in reversed(self.history) if msg["role"] == "user"), None)
        
        if last_user_msg:
            print(f"DEBUG: Retrieving context for query: {last_user_msg[:50]}...")
            retrieved_chunks = self.rag.retrieve(last_user_msg, k=5)
            
            if retrieved_chunks:
                print(f"DEBUG: Found {len(retrieved_chunks)} relevant chunks.")
                context_text = "\n\n---\n".join(retrieved_chunks)
                messages.append({
                    "role": "system", 
                    "content": f"=== RELEVANT DETAILS (RAG) ===\nUse the following specific details retrieved from the document to answer the request:\n{context_text}\n=============================="
                })
            else:
                print("DEBUG: No relevant context found.")

        # 3. Add History (Sliding Window)
        # We work backwards from the most recent message
        print("DEBUG: Counting current tokens...")
        current_tokens = self._count_tokens(messages)
        print(f"DEBUG: Current tokens (System+RAG): {current_tokens}")
        
        history_tokens = 0
        included_history = []

        # Reserve some buffer for the response (e.g., 2048 tokens)
        available_tokens = self.max_tokens - current_tokens - 2048
        print(f"DEBUG: Available tokens for history: {available_tokens}")

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
