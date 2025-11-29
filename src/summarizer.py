import mlx.core as mx
from mlx_lm import generate
from typing import List
import time

class RecursiveSummarizer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def summarize(self, chunks: List[str], filename: str) -> str:
        """
        Performs a Map-Reduce summarization of the document.
        """
        print(f"INFO: Summarizer - Starting recursive summarization for {filename}...")
        start_time = time.time()
        
        # 1. Map Phase: Summarize chunks (or groups of chunks)
        # To save time, we'll group chunks into larger blocks (e.g., 3 chunks at a time)
        # provided they fit in context.
        
        intermediate_summaries = []
        
        # Simple grouping: Combine 3 chunks (~1500 tokens)
        group_size = 3
        for i in range(0, len(chunks), group_size):
            group = chunks[i:i+group_size]
            group_text = "\n".join(group)
            
            prompt = f"""
[INST]
You are a helpful AI assistant.
Summarize the following text concisely, capturing the key points and main ideas.
Text:
{group_text}
[/INST]
Summary:
"""
            # Generate summary for this group
            # We use a lower max_tokens to keep it concise
            summary = generate(self.model, self.tokenizer, prompt=prompt, max_tokens=200, verbose=False)
            intermediate_summaries.append(summary)
            print(f"DEBUG: Summarized group {i//group_size + 1}/{(len(chunks)+group_size-1)//group_size}")

        # 2. Reduce Phase: Summarize the summaries
        if not intermediate_summaries:
            return "No content to summarize."
            
        combined_summaries = "\n".join(intermediate_summaries)
        
        final_prompt = f"""
[INST]
You are a helpful AI assistant.
Below are summaries of different parts of a document: "{filename}".
Create a comprehensive "Mental Model" or executive summary of the ENTIRE document based on these parts.
Structure it with:
- Main Topic/Goal
- Key Findings/Arguments
- Conclusion
Summaries:
{combined_summaries}
[/INST]
Final Mental Model:
"""
        print("INFO: Summarizer - Generating final Mental Model...")
        final_summary = generate(self.model, self.tokenizer, prompt=final_prompt, max_tokens=1000, verbose=False)
        
        duration = time.time() - start_time
        print(f"INFO: Summarizer - Completed in {duration:.2f}s")
        
        return final_summary
