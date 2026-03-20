import asyncio, time, uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from core.models import (
    ChatCompletionRequest, ChatCompletionResponse,
    ModelList, ModelCard
)
from engine.model_runner import runner
from core.config import load_config
import json, logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

cfg = load_config()
app = FastAPI(title="MLX-Serve", version="0.1.0", description="OpenAI-compatible inference server for Apple Silicon")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

AVAILABLE_MODELS = [
    "mlx-community/Qwen1.5-0.5B-Chat",
    "mlx-community/Qwen2-1.5B-Instruct-4bit",
    "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
]

@app.on_event("startup")
async def startup():
    model_name = cfg.model.default
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, runner.load, model_name)
    logger.info(f"Server ready — model: {model_name}")

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": runner.model_name,
        "loaded": runner.is_loaded(),
        "load_time": runner.load_time,
    }

@app.get("/v1/models", response_model=ModelList)
async def list_models():
    return ModelList(data=[
        ModelCard(id=m, created=int(time.time()))
        for m in AVAILABLE_MODELS
    ])

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if not runner.is_loaded():
        raise HTTPException(503, "Model not loaded yet")

    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    prompt = runner.format_prompt(messages)
    prompt_tokens = runner.count_tokens(prompt)

    if req.stream:
        return EventSourceResponse(_stream_response(req, prompt, prompt_tokens))

    loop = asyncio.get_event_loop()
    response_text, elapsed = await loop.run_in_executor(
        None, runner.run, prompt, req.max_tokens, req.temperature
    )
    completion_tokens = runner.count_tokens(response_text)
    logger.info(f"Generated {completion_tokens} tokens in {elapsed}s ({round(completion_tokens/elapsed, 1)} tok/s)")

    return ChatCompletionResponse.build(
        model=req.model,
        content=response_text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )

async def _stream_response(req, prompt, prompt_tokens):
    """SSE streaming — yields tokens as they generate."""
    req_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    loop = asyncio.get_event_loop()

    # For now yield full response as single chunk (streaming will be improved in Week 3-4)
    response_text, elapsed = await loop.run_in_executor(
        None, runner.run, prompt, req.max_tokens, req.temperature
    )
    chunk = {
        "id": req_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": response_text}, "finish_reason": "stop"}]
    }
    yield {"data": json.dumps(chunk)}
    yield {"data": "[DONE]"}
