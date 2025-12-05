# MLX infinite memory chat

A high-performance local chatbot for Apple Silicon. It uses a hybrid approach (RAG + Summarization) to let you chat with unlimited-size documents completely offline.

## 🚀 Key Features

*   **Local LLM**: Runs `Meta-Llama-3.1-8B-Instruct` locally using `mlx-lm` for high-performance inference on Apple Silicon.
*   **Hybrid Cognitive Architecture**:
    *   **Working Memory (Mental Model)**: Automatically reads and recursively summarizes uploaded documents (Map-Reduce) to build a global "Mental Model" injected into the system prompt.
    *   **Long-Term Memory (RAG)**: Uses `sentence-transformers` (MPS-accelerated) to index documents and retrieve specific details on demand.
*   **Background Processing**: Summarization happens in a background thread, allowing you to chat immediately while the "Mental Model" is being built.
*   **Memory-Aware**: Automatically checks available RAM (requires ~8GB min, 16GB recommended) to prevent system freezes.
*   **Sliding Window**: Manages conversation history to allow infinite chat sessions without hitting token limits.
*   **Multi-Modal**: Supports PDF and Python file ingestion.
*   **"System 2" Reasoning Planner**: An agentic loop that "thinks" before it speaks, creating a plan to answer complex queries.
*   **Local GraphRAG**: Extracts knowledge graphs from documents to find hidden relationships and connections.
*   **UI Model Selection**: Switch between downloaded models (e.g., 8B vs 70B) directly from the settings menu.
*   **MLX KV Context Caching**: Smart caching of the system prompt and mental model for instant prefill on subsequent turns.
*   **Professional Logging**: Centralized, structured logging for easier debugging and monitoring.

## 🛠️ Tech Stack

*   **Inference**: [MLX-LM](https://github.com/ml-explore/mlx-examples/tree/main/llms) (Apple Machine Learning framework)
*   **UI**: [Chainlit](https://github.com/Chainlit/chainlit)
*   **RAG**: `sentence-transformers`, `numpy`
*   **GraphRAG**: `networkx` (Graph Construction), `pyvis` (Visualization)
*   **PDF Processing**: `pypdf`

## 📦 Installation

1.  **Prerequisites**:
    *   Apple Silicon Mac (M1/M2/M3/M4)
    *   Python 3.11+
    *   Hugging Face Account (for Llama 3.1 access)

2.  **Clone & Install**:
    ```bash
    git clone https://github.com/rarenicks/mlx-infinite-memory-chat.git
    cd mlx-infinite-memory-chat
    make install
    ```

3.  **Model Setup**:
    *   Accept terms for `meta-llama/Meta-Llama-3.1-8B-Instruct` on Hugging Face.
    *   Login: `huggingface-cli login`
    *   Download:
        ```bash
        make download-model
        ```

## 🏃‍♂️ Usage

Run the application:
```bash
make run
```

1.  **Upload a File**: Drag & Drop a PDF (e.g., a research paper or book).
2.  **Immediate Chat**: Ask specific questions immediately (RAG is active instantly).
3.  **Wait for Mental Model**: Watch the terminal/UI for "Mental Model updated".
4.  **Deep Understanding**: Once updated, ask high-level questions like "What is the main argument?" or "Summarize the conclusion".

## 🧠 Architecture Details

### The "Lost in the Middle" Solution
Standard RAG loses the "big picture". Standard context stuffing hits token limits.
This project uses a **Dual-Stream** approach:
1.  **Stream A**: A recursive summarizer reads the *entire* document in chunks, summarizes them, and then summarizes the summaries. This "Global Summary" is always present in the context.
2.  **Stream B**: A vector store retrieves the top-5 most relevant chunks for every query.

This allows the model to "know" the book (Summary) and "quote" the book (RAG).

### "System 2" Reasoning
Before answering, the model generates a "Reasoning Plan". This helps avoid hallucinations by breaking down the user's request into logical steps.

### Local GraphRAG
The system builds a Knowledge Graph from your documents. If you ask about relationships (e.g., "How is X related to Y?"), it queries this graph to find connections that standard vector search might miss.

## 📄 License
MIT
