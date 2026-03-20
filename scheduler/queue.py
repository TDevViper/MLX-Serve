import asyncio
import uuid
import time
from dataclasses import dataclass, field
from typing import Optional, AsyncGenerator
from enum import Enum

class RequestStatus(Enum):
    WAITING   = "waiting"
    RUNNING   = "running"
    DONE      = "done"
    FAILED    = "failed"

@dataclass
class InferenceRequest:
    request_id: str
    prompt: str
    max_tokens: int
    temperature: float
    stream: bool
    created_at: float = field(default_factory=time.time)
    status: RequestStatus = RequestStatus.WAITING
    result: Optional[str] = None
    error: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed: float = 0.0
    # For streaming — results go into this queue token by token
    token_queue: Optional[asyncio.Queue] = field(default=None, repr=False)

    def __post_init__(self):
        if self.stream:
            self.token_queue = asyncio.Queue()

class RequestQueue:
    def __init__(self, max_concurrent: int = 1):  # MLX GPU is not thread-safe — serial only
        self.max_concurrent = max_concurrent
        self._queue: asyncio.Queue = asyncio.Queue()
        self._active: dict[str, InferenceRequest] = {}
        self._done: dict[str, InferenceRequest] = {}
        self._lock = asyncio.Lock()

    async def submit(self, req: InferenceRequest):
        await self._queue.put(req)

    async def next(self) -> Optional[InferenceRequest]:
        if len(self._active) >= self.max_concurrent:
            return None
        try:
            req = self._queue.get_nowait()
            async with self._lock:
                self._active[req.request_id] = req
            return req
        except asyncio.QueueEmpty:
            return None

    async def complete(self, req: InferenceRequest):
        async with self._lock:
            self._active.pop(req.request_id, None)
            self._done[req.request_id] = req

    def stats(self) -> dict:
        return {
            "waiting":    self._queue.qsize(),
            "active":     len(self._active),
            "completed":  len(self._done),
            "capacity":   self.max_concurrent,
        }

    def get_result(self, request_id: str) -> Optional[InferenceRequest]:
        return self._done.get(request_id) or self._active.get(request_id)
