import asyncio, time, uuid, json, logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from core.models import (
    ChatCompletionRequest, ChatCompletionResponse,
    ModelList, ModelCard
)
from engine.model_runner import runner
from core.config import load_config
from scheduler.queue import RequestQueue, InferenceRequest
from scheduler.worker import InferenceWorker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

cfg = load_config()
app = FastAPI(title="MLX-Serve", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

AVAILABLE_MODELS = [
    "mlx-community/Qwen1.5-0.5B-Chat",
    "mlx-community/Qwen2-1.5B-Instruct-4bit",
    "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
]

queue   = RequestQueue(max_concurrent=cfg.server.max_concurrent_requests)
worker  = InferenceWorker(queue)

@app.on_event("startup")
async def startup():
    await asyncio.get_event_loop().run_in_executor(None, runner.load, cfg.model.default)
    await worker.start()
    logger.info(f"Server ready — model: {cfg.model.default}")

@app.on_event("shutdown")
async def shutdown():
    await worker.stop()

@app.get("/health")
async def health():
    return {
        "status":    "ok",
        "model":     runner.model_name,
        "loaded":    runner.is_loaded(),
        "load_time": runner.load_time,
        "queue":     queue.stats(),
    }

@app.get("/v1/models", response_model=ModelList)
async def list_models():
    return ModelList(data=[
        ModelCard(id=m, created=int(time.time())) for m in AVAILABLE_MODELS
    ])

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if not runner.is_loaded():
        raise HTTPException(503, "Model not loaded yet")

    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    prompt   = runner.format_prompt(messages)

    inf_req = InferenceRequest(
        request_id  = uuid.uuid4().hex,
        prompt      = prompt,
        max_tokens  = req.max_tokens,
        temperature = req.temperature,
        stream      = req.stream,
    )
    await queue.submit(inf_req)

    if req.stream:
        return EventSourceResponse(_stream_tokens(inf_req, req.model))

    # Wait for result
    while inf_req.status.value in ("waiting", "running"):
        await asyncio.sleep(0.05)

    if inf_req.error:
        raise HTTPException(500, inf_req.error)

    return ChatCompletionResponse.build(
        model              = req.model,
        content            = inf_req.result,
        prompt_tokens      = inf_req.prompt_tokens,
        completion_tokens  = inf_req.completion_tokens,
    )

async def _stream_tokens(inf_req: InferenceRequest, model_name: str):
    req_id = f"chatcmpl-{inf_req.request_id[:8]}"
    created = int(time.time())

    # Wait until request starts processing
    while inf_req.token_queue is None:
        await asyncio.sleep(0.01)

    while True:
        token = await inf_req.token_queue.get()
        if token is None:
            # Send final done chunk
            done_chunk = {
                "id": req_id, "object": "chat.completion.chunk",
                "created": created, "model": model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
            }
            yield {"data": json.dumps(done_chunk)}
            yield {"data": "[DONE]"}
            break

        chunk = {
            "id": req_id, "object": "chat.completion.chunk",
            "created": created, "model": model_name,
            "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}]
        }
        yield {"data": json.dumps(chunk)}
