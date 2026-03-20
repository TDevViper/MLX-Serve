import asyncio, time, uuid, json, logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from sse_starlette.sse import EventSourceResponse
from core.models import (
    ChatCompletionRequest, ChatCompletionResponse,
    ModelList, ModelCard
)
from core.metrics import metrics
from core.stats import stats
from engine.model_runner import runner
from engine.kv_cache import kv_cache
from engine.memory_monitor import MemoryMonitor
from engine.embedder import embedder
from core.config import load_config
from scheduler.queue import RequestQueue, InferenceRequest
from scheduler.worker import InferenceWorker
from scheduler.preemptor import Preemptor
from pydantic import BaseModel
from typing import Optional, List, Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

cfg    = load_config()
app    = FastAPI(
    title="MLX-Serve",
    version="0.7.0",
    description="OpenAI-compatible LLM inference server for Apple Silicon",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

AVAILABLE_MODELS = [
    "mlx-community/Qwen1.5-0.5B-Chat",
    "mlx-community/Qwen2-1.5B-Instruct-4bit",
    "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
]
EMBEDDING_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
]

queue       = RequestQueue(max_concurrent=cfg.server.max_concurrent_requests)
worker      = InferenceWorker(queue)
mem_monitor = MemoryMonitor(kv_cache, poll_interval=1.0)
preemptor   = Preemptor(kv_cache)

@app.on_event("startup")
async def startup():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, runner.load, cfg.model.default)
    await loop.run_in_executor(None, embedder.load, "sentence-transformers/all-MiniLM-L6-v2")
    await worker.start()
    await mem_monitor.start()
    preemptor.attach(queue, mem_monitor)
    logger.info(f"Server ready — model: {cfg.model.default}")

@app.on_event("shutdown")
async def shutdown():
    await worker.stop()
    await mem_monitor.stop()

# ── Health & metrics ──────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status":    "ok",
        "model":     runner.model_name,
        "loaded":    runner.is_loaded(),
        "load_time": runner.load_time,
        "queue":     queue.stats(),
        "metrics":   metrics.summary(),
        "memory":    mem_monitor.current(),
        "embedder":  {"model": embedder.model_name, "dim": embedder.dim()},
    }

@app.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics():
    return metrics.prometheus_with_kv(
        queue_stats=queue.stats(),
        kv_stats=kv_cache.stats(),
        mem_stats=mem_monitor.stats(),
    )

# ── OpenAI-compatible endpoints ───────────────────────────────────────────────
@app.get("/v1/models", response_model=ModelList)
async def list_models():
    all_models = AVAILABLE_MODELS + EMBEDDING_MODELS
    return ModelList(data=[
        ModelCard(id=m, created=int(time.time())) for m in all_models
    ])

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if not runner.is_loaded():
        raise HTTPException(503, "Model not loaded yet")
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    prompt   = runner.format_prompt(messages)
    inf_req  = InferenceRequest(
        request_id=uuid.uuid4().hex, prompt=prompt,
        max_tokens=req.max_tokens, temperature=req.temperature, stream=req.stream,
    )
    await queue.submit(inf_req)
    if req.stream:
        return EventSourceResponse(_stream_tokens(inf_req, req.model))
    while inf_req.status.value in ("waiting", "running"):
        await asyncio.sleep(0.05)
    if inf_req.error:
        raise HTTPException(500, inf_req.error)
    return ChatCompletionResponse.build(
        model=req.model, content=inf_req.result,
        prompt_tokens=inf_req.prompt_tokens,
        completion_tokens=inf_req.completion_tokens,
    )

class CompletionRequest(BaseModel):
    model: str = "mlx-community/Qwen1.5-0.5B-Chat"
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    stream: bool = False

@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    if not runner.is_loaded():
        raise HTTPException(503, "Model not loaded yet")
    inf_req = InferenceRequest(
        request_id=uuid.uuid4().hex, prompt=req.prompt,
        max_tokens=req.max_tokens, temperature=req.temperature, stream=req.stream,
    )
    await queue.submit(inf_req)
    if req.stream:
        return EventSourceResponse(_stream_completion(inf_req, req.model))
    while inf_req.status.value in ("waiting", "running"):
        await asyncio.sleep(0.05)
    if inf_req.error:
        raise HTTPException(500, inf_req.error)
    return {
        "id":      f"cmpl-{inf_req.request_id[:8]}",
        "object":  "text_completion",
        "created": int(time.time()),
        "model":   req.model,
        "choices": [{"text": inf_req.result, "index": 0, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens":     inf_req.prompt_tokens,
            "completion_tokens": inf_req.completion_tokens,
            "total_tokens":      inf_req.prompt_tokens + inf_req.completion_tokens,
        }
    }

class EmbeddingRequest(BaseModel):
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    input: Union[str, List[str]]
    encoding_format: str = "float"

@app.post("/v1/embeddings")
async def create_embeddings(req: EmbeddingRequest):
    if not embedder.is_loaded():
        raise HTTPException(503, "Embedding model not loaded yet")
    texts = [req.input] if isinstance(req.input, str) else req.input
    loop = asyncio.get_event_loop()
    try:
        vectors = await loop.run_in_executor(None, embedder.embed, texts)
    except Exception as e:
        raise HTTPException(500, str(e))
    return {
        "object": "list",
        "model":  req.model,
        "data": [
            {
                "object":    "embedding",
                "index":     i,
                "embedding": vec,
            }
            for i, vec in enumerate(vectors)
        ],
        "usage": {
            "prompt_tokens": sum(len(t.split()) for t in texts),
            "total_tokens":  sum(len(t.split()) for t in texts),
        }
    }

# ── Stats endpoints ───────────────────────────────────────────────────────────
@app.get("/v1/kv_cache")
async def kv_cache_stats():
    s = kv_cache.stats()
    s["memory_efficiency"] = f"{round((1 - s['utilization']) * 100, 1)}% free"
    return s

@app.get("/v1/memory")
async def memory_stats():
    return mem_monitor.stats()

@app.get("/v1/stats")
async def model_stats():
    return {
        "models":  stats.all(),
        "summary": metrics.summary(),
        "queue":   queue.stats(),
    }

# ── Streaming helpers ─────────────────────────────────────────────────────────
async def _stream_tokens(inf_req, model_name):
    req_id  = f"chatcmpl-{inf_req.request_id[:8]}"
    created = int(time.time())
    while inf_req.token_queue is None:
        await asyncio.sleep(0.01)
    while True:
        token = await inf_req.token_queue.get()
        if token is None:
            yield {"data": json.dumps({"id": req_id, "object": "chat.completion.chunk",
                "created": created, "model": model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})}
            yield {"data": "[DONE]"}
            break
        yield {"data": json.dumps({"id": req_id, "object": "chat.completion.chunk",
            "created": created, "model": model_name,
            "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}]})}

async def _stream_completion(inf_req, model_name):
    req_id  = f"cmpl-{inf_req.request_id[:8]}"
    created = int(time.time())
    while inf_req.token_queue is None:
        await asyncio.sleep(0.01)
    while True:
        token = await inf_req.token_queue.get()
        if token is None:
            yield {"data": json.dumps({"id": req_id, "object": "text_completion",
                "created": created, "model": model_name,
                "choices": [{"text": "", "index": 0, "finish_reason": "stop"}]})}
            yield {"data": "[DONE]"}
            break
        yield {"data": json.dumps({"id": req_id, "object": "text_completion",
            "created": created, "model": model_name,
            "choices": [{"text": token, "index": 0, "finish_reason": None}]})}
