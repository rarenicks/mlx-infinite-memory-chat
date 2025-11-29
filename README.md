# MLX infinite memory chat

A high-performance local chatbot for Apple Silicon. It uses a hybrid approach (RAG + Summarization) to let you chat with unlimited-size documents completely offline.

## 🚀 Key Features

*   **Local LLM**: Runs `Meta-Llama-3.1-70B-Instruct` locally using `mlx-lm` for high-performance inference on Apple Silicon.
*   **Hybrid Cognitive Architecture**:
    *   **Working Memory (Mental Model)**: Automatically reads and recursively summarizes uploaded documents (Map-Reduce) to build a global "Mental Model" injected into the system prompt.
    *   **Long-Term Memory (RAG)**: Uses `sentence-transformers` (MPS-accelerated) to index documents and retrieve specific details on demand.
*   **Background Processing**: Summarization happens in a background thread, allowing you to chat immediately while the "Mental Model" is being built.
*   **Memory-Aware**: Automatically checks available RAM (requires ~32GB min, 64GB recommended) to prevent system freezes.
*   **Sliding Window**: Manages conversation history to allow infinite chat sessions without hitting token limits.
*   **Multi-Modal**: Supports PDF and Python file ingestion.

## 🛠️ Tech Stack

*   **Inference**: [MLX-LM](https://github.com/ml-explore/mlx-examples/tree/main/llms) (Apple Machine Learning framework)
*   **UI**: [Chainlit](https://github.com/Chainlit/chainlit)
*   **RAG**: `sentence-transformers`, `numpy`
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
    *   Accept terms for `meta-llama/Meta-Llama-3.1-70B-Instruct` on Hugging Face.
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

## 📄 License
MIT
