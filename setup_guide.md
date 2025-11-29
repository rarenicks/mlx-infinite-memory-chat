# Setup Guide: Unlimited Context Local Chatbot

## 1. Environment Setup

Ensure you have Python 3.10+ installed. It is recommended to use a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Model Download

This application is optimized for `mlx-community/Llama-3.1-70B-Instruct-4bit`. This is a large model (~40GB).

You can download it using the Hugging Face CLI (installed with transformers) or let `mlx_lm` handle it on first run (though pre-downloading is recommended for stability).

### Option A: Hugging Face CLI (Recommended)

> [!IMPORTANT]
> This model is gated. You must:
> 1. Accept the terms on the [Meta Llama 3.1 page](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct).
> 2. Create a [Hugging Face Access Token](https://huggingface.co/settings/tokens).
> 3. Run `huggingface-cli login` and paste your token.

```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli download mlx-community/Meta-Llama-3.1-70B-Instruct-4bit --local-dir models/Llama-3.1-70B-Instruct-4bit
```

### Option B: Automatic Download

If you do not manually download the model, the application will attempt to download it to the default Hugging Face cache directory (`~/.cache/huggingface/hub`) upon the first run.

## 3. Running the Application

```bash
chainlit run app.py
```
