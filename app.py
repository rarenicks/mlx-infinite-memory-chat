import chainlit as cl
from mlx_lm import generate, stream_generate
import mlx.core as mx
from mlx_lm.sample_utils import make_sampler
from src.model_loader import load_model, check_memory, list_available_models
from src.context_manager import ContextManager
from src.file_processor import process_file
from src.summarizer import RecursiveSummarizer
import os
import threading
import asyncio
import time
import gc
from src.logging_config import setup_logger

logger = setup_logger(__name__)

# Global state
model = None
tokenizer = None
context_manager = None
summarizer = None
model_lock = threading.Lock()

@cl.on_chat_start
async def start():
    global model, tokenizer, context_manager, summarizer, model_lock
    
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
            # Initialize Context Manager
            # Pass model for GraphRAG
            context_manager = ContextManager(tokenizer, model=model, lock=model_lock)
            summarizer = RecursiveSummarizer(model, tokenizer, model_lock) # Pass lock
            msg.content = "✅ Model loaded successfully! You can now start chatting."
            await msg.update()
        except Exception as e:
            msg.content = f"❌ Error loading model: {e}"
            await msg.update()
            return

    # 3. Settings
    available_models = list_available_models()
    # Default to the first model or a specific one if available
    initial_model = available_models[0] if available_models else "Llama-3.1-8B-Instruct-4bit"
    
    settings = await cl.ChatSettings(
        [
            cl.input_widget.Select(
                id="model",
                label="Model",
                values=available_models,
                initial_index=0 if available_models else 0
            ),
            cl.input_widget.TextInput(id="system_prompt", label="System Prompt", initial="You are a highly capable AI assistant running locally on Apple Silicon. You have access to a Knowledge Graph and RAG engine to answer questions based on uploaded documents. You can also plan complex tasks using a reasoning engine."),
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
    global model, tokenizer, context_manager, summarizer, model_lock
    
    # Check if model changed
    current_settings = cl.user_session.get("settings")
    if current_settings and current_settings.get("model") != settings["model"]:
        new_model_name = settings["model"]
        await cl.Message(content=f"🔄 Switching model to **{new_model_name}**...").send()
        
        # Unload current model
        with model_lock:
            model = None
            tokenizer = None
            gc.collect()
        
        # Load new model
        try:
            model_path = os.path.join("models", new_model_name)
            model, tokenizer = load_model(model_path)
            
            # Re-initialize components with new model
            context_manager = ContextManager(tokenizer, model=model, lock=model_lock)
            summarizer = RecursiveSummarizer(model, tokenizer, model_lock)
            
            # Restore system prompt
            context_manager.set_system_prompt(settings["system_prompt"])
            
            await cl.Message(content=f"✅ Switched to **{new_model_name}** successfully!").send()
        except Exception as e:
            await cl.Message(content=f"❌ Error switching model: {e}").send()
            return

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
                logger.info(f"Processing {element.name}...")
                content = process_file(element.path)
                
                # A. Immediate RAG Indexing
                context_manager.add_file_context(content, filename=element.name)
                logger.info(f"Added {element.name} to RAG context.")
                await cl.Message(content=f"📄 Indexed {element.name} for RAG retrieval.").send()
                
                # B. Background Summarization (The "Novel" Part)
                def run_summarization(text, fname):
                    logger.info(f"Starting background summarization for {fname}...")
                    # Get chunks from RAG engine (it already chunked them)
                    # For simplicity, we'll just re-chunk or access rag.chunks if we exposed them.
                    # Let's just use the raw text and let summarizer handle it (or use rag chunks if accessible).
                    # Accessing rag chunks is cleaner.
                    # chunks = context_manager.rag.chunks[-len(context_manager.rag._chunk_text(text, 500, 50)):] # Hacky way to get last added chunks
                    # Better: Just re-chunk in summarizer or pass text. 
                    # Let's pass the text and let summarizer chunk it for summarization (maybe different size).
                    summary = summarizer.summarize(context_manager.rag._chunk_text(text, 1000, 100), fname)
                    context_manager.set_document_summary(summary)
                    logger.info(f"Mental Model updated for {fname}.")
                    
                # Start thread
                threading.Thread(target=run_summarization, args=(content, element.name), daemon=True).start()
                await cl.Message(content=f"🧠 Started background analysis to build 'Mental Model' for {element.name}... (You can chat while this runs)").send()

    # 2. Add User Message to History
    context_manager.add_message("user", message.content)
    
    # --- UPGRADE 2: "System 2" Reasoning Planner ---
    # We want to "Think" before we "Speak".
    
    # 1. Define the Reasoning Prompt
    reasoning_system_prompt = "You are a logic engine. Analyze the user's request. Break down the steps needed to answer this accurately based on the provided Reference Material. Output a concise plan."
    
    # Create a temporary context for reasoning (System + User)
    # We don't want the full RAG context here necessarily, or maybe we do? 
    # Let's use the full context so it knows what it has access to.
    # But we need to inject the specific instruction.
    
    # Actually, let's just append the instruction to the messages for the reasoning step
    reasoning_messages = context_manager.get_messages_for_inference()
    # Replace the last user message with the reasoning prompt + user message
    last_user_msg = reasoning_messages[-1]['content']
    reasoning_messages[-1]['content'] = f"{last_user_msg}\n\n[INSTRUCTION: {reasoning_system_prompt}]"
    
    reasoning_step = cl.Step(name="🧠 Reasoning", type="tool")
    await reasoning_step.send()
    
    reasoning_plan = ""
    
    # Generate Reasoning
    # We use a separate lock acquisition here if needed, but since it's sequential in this async function, 
    # we just need to make sure we don't conflict with background summarizer.
    
    logger.info("Starting Reasoning Step...")
    
    # We'll use a simple generate call for reasoning (no streaming needed for the step content necessarily, but looks cool)
    # Let's stream it to the step.
    
    try:
        # Prepare prompt for reasoning
        reasoning_prompt = tokenizer.apply_chat_template(reasoning_messages, tokenize=False, add_generation_prompt=True)
        
        # Stream reasoning
        with model_lock:
            for response in stream_generate(model, tokenizer, prompt=reasoning_prompt, max_tokens=512, sampler=make_sampler(0.7)):
                reasoning_plan += response.text
                await reasoning_step.stream_token(response.text)
                
        await reasoning_step.update()
        
        # --- GRAPH VISUALIZATION ---
        # If the plan mentions "relationships" or "connections", show the graph
        if context_manager.graph_rag and ("relation" in reasoning_plan.lower() or "connect" in reasoning_plan.lower()):
            html_path = "graph.html"
            context_manager.graph_rag.visualize(html_path)
            if os.path.exists(html_path):
                # Read HTML content
                with open(html_path, "r") as f:
                    html_content = f.read()
                # Display as Chainlit Element
                await cl.Message(content="🕸️ **Knowledge Graph Visualization**", elements=[
                    cl.Html(content=html_content, name="knowledge_graph", display="inline")
                ]).send()
        
    except Exception as e:
        logger.error(f"ERROR in Reasoning: {e}")
        reasoning_plan = "Error generating plan. Proceeding with direct answer."
        await reasoning_step.stream_token(f"\nError: {e}")
        await reasoning_step.update()

    # Append the plan to the context for the final answer
    # We treat the plan as an "assistant" thought, or just part of the context.
    # Let's add it as an assistant message, then the user says "Proceed".
    # Or simpler: Just inject it into the final prompt.
    
    # Let's add it to the history temporarily or just modify the prompt?
    # Modifying the prompt is safer to not pollute the visible chat history with "thoughts".
    
    # --- UPGRADE 1: MLX KV Context Caching ---
    
    # 1. Get Static Context (System Prompt + RAG/Docs)
    # We need to separate the "Static" part (System + Docs) from the "Dynamic" part (Chat History).
    # For simplicity in this v1, we will cache the *entire* conversation up to the last turn.
    # But the user asked for "Static Context". 
    
    # Let's implement a robust KV Cache strategy:
    # Cache = KV Cache of (System Prompt + All RAG Context).
    # This is "Static" as long as no new files are added.
    
    # Check if we have a valid cache
    kv_cache = cl.user_session.get("kv_cache")
    cached_text = cl.user_session.get("cached_text")
    
    # Construct the text we WANT to cache (System + RAG)
    # We need a way to get JUST the system + RAG messages.
    # ContextManager doesn't explicitly separate them easily, but we know they are at the start.
    # Let's assume: System Prompt + Document Summary + RAG Chunks are the "Static" prefix.
    
    # Hack: Let's rebuild the prompt.
    # The prompt is: <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}\n\n{mental_model}\n\n{rag_context}<|eot_id|>...
    
    # We will cache the tokenized output of this prefix.
    
    current_system_msg = context_manager.system_prompt
    if context_manager.document_summary:
        current_system_msg += f"\n\n=== MENTAL MODEL ===\n{context_manager.document_summary}"
    
    # RAG context is dynamic per query? 
    # WAIT. RAG retrieves *different* chunks for every query.
    # If RAG changes every turn, we CANNOT cache it as "Static".
    # The user asked to cache "System Prompt + Uploaded Document Context".
    # If we are using RAG, the "Uploaded Document Context" is the *retrieved* chunks.
    # If we use the *entire* document, that's different.
    
    # Interpretation: The user might assume we are stuffing the whole doc, OR they want to cache the System Prompt + Mental Model.
    # Since RAG is dynamic, we can only cache the System Prompt + Mental Model.
    # UNLESS: We cache the *prefix* of the prompt which is just the System Prompt + Mental Model.
    
    # Let's cache the System Prompt + Mental Model.
    static_text = current_system_msg
    
    # If cache is missing or text changed, rebuild it.
    if kv_cache is None or cached_text != static_text:
        logger.info("KV Cache Miss. Rebuilding static cache...")
        # Tokenize just the system message part
        # We need to manually format it as a system message for Llama 3
        static_tokens = tokenizer.encode(f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{static_text}<|eot_id|>")
        
        # Prefill
        from mlx_lm.models.cache import make_prompt_cache
        
        kv_cache = make_prompt_cache(model)
        
        # We need to process these tokens to populate the cache
        # We can use `model(input_ids, cache=kv_cache)`
        input_ids = mx.array(static_tokens)[None, :]
        
        with model_lock:
            model(input_ids, cache=kv_cache)
            
        # Store in session
        cl.user_session.set("kv_cache", kv_cache)
        cl.user_session.set("cached_text", static_text)
        logger.info(f"KV Cache Rebuilt. ({len(static_tokens)} tokens)")
    else:
        logger.info("KV Cache Hit! Reusing static context.")

    # 2. Generate Final Response using Cache
    # Now we need to generate the REST of the prompt (History + User Message + Plan)
    # And append it to the cached context.
    
    # We need to be careful. `stream_generate` usually takes a prompt.
    # If we pass `kv_cache`, it assumes the prompt *continues* from where the cache left off.
    # So we must NOT include the System Message in the prompt we pass to `stream_generate`.
    
    # Construct dynamic prompt (History + User)
    # We need to strip the system message from the messages list because it's already in the cache.
    # Use get_messages_for_inference() instead of non-existent .messages attribute
    full_messages = context_manager.get_messages_for_inference()
    
    # Filter out the static parts (System Prompt + Mental Model) that are already in KV Cache
    # We know RAG chunks (also role=system) are dynamic, so we KEEP them.
    # The static parts are the first 1 or 2 messages.
    
    dynamic_messages = []
    for m in full_messages:
        # If it's the main system prompt or mental model, skip it (it's cached)
        # We identify them by content match or just assumption.
        # Safer: Check if content is part of static_text
        if m['role'] == 'system' and (m['content'] == context_manager.system_prompt or "=== CURRENT DOCUMENT MENTAL MODEL ===" in m['content']):
            continue
        dynamic_messages.append(m)
    
    # Add the plan to the context
    dynamic_messages.append({"role": "assistant", "content": f"Plan:\n{reasoning_plan}\n\nProceeding with answer."})
    dynamic_messages.append({"role": "user", "content": "Please provide the final answer based on the plan."})
    
    # Apply template but EXCLUDE the system prompt (since we manually cached it)
    # Llama 3 template usually starts with system. We need to trick it.
    # We will just format the conversation history manually or use the tokenizer without system.
    
    # Better approach for Llama 3:
    # The cache contains: <|begin_of_text|><|start_header_id|>system...<|eot_id|>
    # The next tokens should start with <|start_header_id|>user... (or whatever the next message is)
    
    dynamic_prompt = ""
    for m in dynamic_messages:
        dynamic_prompt += f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n{m['content']}<|eot_id|>"
    
    dynamic_prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    
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
    
    # Verify device
    logger.info(f"MLX Device: {mx.default_device()}")
    
    start_time = time.time()
    
    # Queue for passing tokens from thread to async loop
    token_queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    
    def generate_worker():
        """Runs the blocking generation in a separate thread."""
        logger.debug("Generation worker thread started.")
        
        token_count = 0
        first_token_time = 0
        decode_start = 0
        
        try:
            # We need to track time *inside* the loop to separate prefill from decode
            start_gen = time.time()
            
            # Acquire lock to prevent Metal concurrency crashes
            with model_lock:
                # IMPORTANT: We must pass the KV Cache here
                # And the prompt is ONLY the dynamic part
                for response in stream_generate(
                    model, 
                    tokenizer, 
                    prompt=dynamic_prompt, 
                    max_tokens=max_tokens, 
                    sampler=sampler,
                    prompt_cache=kv_cache # Use the cache!
                ):
                    current_time = time.time()
                    
                    if token_count == 0:
                        # First token received (Prefill done)
                        first_token_time = current_time
                        decode_start = current_time
                        prefill_duration = first_token_time - start_gen
                        logger.info(f"Prefill complete in {prefill_duration:.2f}s")
                    
                    # Put token in queue in a thread-safe way
                    loop.call_soon_threadsafe(token_queue.put_nowait, response.text)
                    token_count += 1
            
            decode_duration = time.time() - decode_start
            
            if token_count > 1 and decode_duration > 0:
                tps = (token_count - 1) / decode_duration
                logger.info(f"Generation complete. Decode Speed: {tps:.2f} tokens/sec (excluding prefill).")
            else:
                logger.info(f"Generation complete. Total tokens: {token_count}")
            
            # Signal completion
            loop.call_soon_threadsafe(token_queue.put_nowait, None)
            
        except Exception as e:
            # Signal error
            logger.error(f"ERROR in generation worker: {e}")
            loop.call_soon_threadsafe(token_queue.put_nowait, e)
        finally:
            # Force cleanup in the thread
            gc.collect()
            logger.debug("Generation worker cleanup done.")

    # Start generation in a background thread
    logger.debug("Starting generation thread...")
    threading.Thread(target=generate_worker, daemon=True).start()

    # Consume tokens from the queue asynchronously
    first_token_received = False
    try:
        while True:
            # Wait for next token
            token = await token_queue.get()
            
            if not first_token_received and token is not None:
                first_token_time = time.time() - start_time
                logger.info(f"First token received after {first_token_time:.2f}s")
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
        logger.info(f"Token Usage: {stats['total']}/{stats['limit']} ({stats['percent']:.1f}%)")
