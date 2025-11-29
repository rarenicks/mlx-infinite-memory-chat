import chainlit as cl
from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler
from src.model_loader import load_model, check_memory
from src.context_manager import ContextManager
from src.file_processor import process_file
from src.summarizer import RecursiveSummarizer
import os
import threading
import asyncio
import time
import gc

# Global state
model = None
tokenizer = None
context_manager = None
summarizer = None

@cl.on_chat_start
async def start():
    global model, tokenizer, context_manager

    # 1. Memory Check
    is_safe, msg = check_memory()
    if not is_safe:
        await cl.Message(content=f"⚠️ **Memory Warning**: {msg}").send()
    
    # 2. Load Model (if not already loaded)
    if model is None:
        msg = cl.Message(content="Loading model... (This may take a minute)")
        await msg.send()
        try:
            model, tokenizer = load_model()
            context_manager = ContextManager(tokenizer)
            summarizer = RecursiveSummarizer(model, tokenizer) # Initialize summarizer here
            msg.content = "✅ Model loaded successfully! You can now start chatting."
            await msg.update()
        except Exception as e:
            msg.content = f"❌ Error loading model: {e}"
            await msg.update()
            return

    # 3. Settings
    settings = await cl.ChatSettings(
        [
            cl.input_widget.TextInput(id="system_prompt", label="System Prompt", initial="You are a helpful AI assistant."),
            cl.input_widget.Slider(id="temperature", label="Temperature", initial=0.7, min=0.0, max=1.0, step=0.1),
            cl.input_widget.Slider(id="max_tokens", label="Max Response Tokens", initial=2048, min=128, max=8192, step=128),
        ]
    ).send()
    
    # Initialize session state for settings
    cl.user_session.set("settings", settings)
    # Apply initial system prompt
    if context_manager:
        context_manager.set_system_prompt(settings["system_prompt"])

@cl.on_settings_update
async def setup_agent(settings):
    cl.user_session.set("settings", settings)
    if context_manager:
        context_manager.set_system_prompt(settings["system_prompt"])
    await cl.Message(content="✅ Settings updated.").send()

@cl.on_message
async def main(message: cl.Message):
    global model, tokenizer, context_manager, summarizer

    if not model:
        await cl.Message(content="Model not loaded. Please restart the app.").send()
        return

    # 1. Handle File Uploads
    if message.elements:
        await cl.Message(content="INFO: Processing uploaded files...").send()
        for element in message.elements:
            if element.path:
                print(f"INFO: Processing {element.name}...")
                content = process_file(element.path)
                
                # A. Immediate RAG Indexing
                context_manager.add_file_context(content, filename=element.name)
                print(f"INFO: Added {element.name} to RAG context.")
                await cl.Message(content=f"📄 Indexed {element.name} for RAG retrieval.").send()
                
                # B. Background Summarization (The "Novel" Part)
                def run_summarization(text, fname):
                    print(f"INFO: Starting background summarization for {fname}...")
                    # Get chunks from RAG engine (it already chunked them)
                    # For simplicity, we'll just re-chunk or access rag.chunks if we exposed them.
                    # Let's just use the raw text and let summarizer handle it (or use rag chunks if accessible).
                    # Accessing rag chunks is cleaner.
                    # chunks = context_manager.rag.chunks[-len(context_manager.rag._chunk_text(text, 500, 50)):] # Hacky way to get last added chunks
                    # Better: Just re-chunk in summarizer or pass text. 
                    # Let's pass the text and let summarizer chunk it for summarization (maybe different size).
                    summary = summarizer.summarize(context_manager.rag._chunk_text(text, 1000, 100), fname)
                    context_manager.set_document_summary(summary)
                    print(f"INFO: Mental Model updated for {fname}.")
                    
                # Start thread
                threading.Thread(target=run_summarization, args=(content, element.name), daemon=True).start()
                await cl.Message(content=f"🧠 Started background analysis to build 'Mental Model' for {element.name}... (You can chat while this runs)").send()

    # 2. Add User Message to History
    context_manager.add_message("user", message.content)
    
    # 3. Prepare Messages
    print("INFO: Preparing context and calculating tokens...")
    messages = context_manager.get_messages_for_inference()
    
    # Estimate prefill time
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_tokens = len(tokenizer.encode(prompt))
    estimated_time = prompt_tokens / 1000
    
    print(f"INFO: Context ready. Total tokens: {prompt_tokens:,}. Est. wait: {estimated_time:.1f}s")
    
    # 4. Generate Response
    settings = cl.user_session.get("settings")
    if not settings:
        settings = {
            "system_prompt": "You are a helpful AI assistant.",
            "temperature": 0.7,
            "max_tokens": 2048
        }
    temperature = settings["temperature"]
    max_tokens = settings["max_tokens"]

    msg = cl.Message(content="")
    await msg.send()

    # Create sampler
    sampler = make_sampler(temp=temperature)

    response_content = ""
    
    # Streaming generation
    from mlx_lm import stream_generate
    import mlx.core as mx
    
    # Verify device
    print(f"INFO: MLX Device: {mx.default_device()}")
    
    start_time = time.time()
    
    if prompt_tokens > 10000:
        print(f"INFO: Large prompt detected ({prompt_tokens:,} tokens). This may take ~{estimated_time:.0f}s.")

    # Queue for passing tokens from thread to async loop
    token_queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    
    def generate_worker():
        """Runs the blocking generation in a separate thread."""
        print("DEBUG: Generation worker thread started.")
        
        token_count = 0
        first_token_time = 0
        decode_start = 0
        
        try:
            # We need to track time *inside* the loop to separate prefill from decode
            start_gen = time.time()
            
            for response in stream_generate(
                model, 
                tokenizer, 
                prompt=prompt, 
                max_tokens=max_tokens, 
                sampler=sampler
            ):
                current_time = time.time()
                
                if token_count == 0:
                    # First token received (Prefill done)
                    first_token_time = current_time
                    decode_start = current_time
                    prefill_duration = first_token_time - start_gen
                    print(f"INFO: Prefill complete in {prefill_duration:.2f}s")
                
                # Put token in queue in a thread-safe way
                loop.call_soon_threadsafe(token_queue.put_nowait, response.text)
                token_count += 1
            
            decode_duration = time.time() - decode_start
            # We generated (token_count - 1) tokens during decode_duration (since 1st token is prefill end)
            # But usually stream_generate yields the first token *after* prefill.
            # So all tokens are part of the "streaming" phase, but the *wait* for the first one is prefill.
            
            # Let's calculate TPS based on the time from 1st token to last token
            if token_count > 1 and decode_duration > 0:
                tps = (token_count - 1) / decode_duration
                print(f"INFO: Generation complete. Decode Speed: {tps:.2f} tokens/sec (excluding prefill).")
            else:
                print(f"INFO: Generation complete. Total tokens: {token_count}")
            
            # Signal completion
            loop.call_soon_threadsafe(token_queue.put_nowait, None)
            
        except Exception as e:
            # Signal error
            print(f"ERROR in generation worker: {e}")
            loop.call_soon_threadsafe(token_queue.put_nowait, e)
        finally:
            # Force cleanup in the thread
            gc.collect()
            print("DEBUG: Generation worker cleanup done.")

    # Start generation in a background thread
    print("DEBUG: Starting generation thread...")
    threading.Thread(target=generate_worker, daemon=True).start()

    # Consume tokens from the queue asynchronously
    first_token_received = False
    try:
        while True:
            # Wait for next token
            token = await token_queue.get()
            
            if not first_token_received and token is not None:
                first_token_time = time.time() - start_time
                print(f"INFO: First token received after {first_token_time:.2f}s")
                first_token_received = True
            
            # Check for completion sentinel
            if token is None:
                break
            
            # Check for error
            if isinstance(token, Exception):
                raise token
            
            # Stream token to UI
            response_content += token
            await msg.stream_token(token)
            
    except Exception as e:
        print(f"ERROR during generation: {e}")
        await cl.Message(content=f"❌ Error generating response: {e}").send()

    # 5. Update History
    context_manager.add_message("assistant", response_content)
    
    # 6. Show Token Usage
    stats = context_manager.get_token_usage_stats()
    await msg.update()
    
    # Display token usage as a toast notification
    if stats['percent'] > 80:
        await cl.Message(content=f"⚠️ **Warning**: Context usage at {stats['percent']:.1f}% ({stats['total']}/{stats['limit']})").send()
    else:
        print(f"Token Usage: {stats['total']}/{stats['limit']} ({stats['percent']:.1f}%)")
