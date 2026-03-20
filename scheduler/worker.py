import asyncio
import logging
import time
from scheduler.queue import RequestQueue, RequestStatus, InferenceRequest
from engine.model_runner import runner
from engine.kv_cache import kv_cache
from core.metrics import metrics, RequestMetric
from core.stats import stats

logger = logging.getLogger(__name__)

class InferenceWorker:
    def __init__(self, queue: RequestQueue):
        self.queue = queue
        self.running = False
        self._task = None

    async def start(self):
        self.running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("Inference worker started")

    async def stop(self):
        self.running = False
        if self._task:
            self._task.cancel()

    async def _loop(self):
        while self.running:
            req = await self.queue.next()
            if req is None:
                await asyncio.sleep(0.01)
                continue
            asyncio.create_task(self._process(req))

    async def _process(self, req: InferenceRequest):
        req.status = RequestStatus.RUNNING
        loop = asyncio.get_event_loop()
        try:
            req.prompt_tokens = runner.count_tokens(req.prompt)

            # Allocate KV cache blocks for this sequence
            ok = kv_cache.init_sequence(req.request_id, req.prompt_tokens)
            if not ok:
                raise RuntimeError("KV cache OOM — no free blocks available")

            if req.stream:
                await self._process_streaming(req, loop)
            else:
                text, elapsed = await loop.run_in_executor(
                    None, runner.run, req.prompt, req.max_tokens, req.temperature
                )
                req.result = text
                req.elapsed = elapsed
                req.completion_tokens = runner.count_tokens(text)
                req.status = RequestStatus.DONE
                tps = round(req.completion_tokens / max(elapsed, 0.01), 1)
                logger.info(
                    f"[{req.request_id[:8]}] {req.completion_tokens} tokens "
                    f"in {elapsed}s ({tps} tok/s) | "
                    f"KV blocks: {len(kv_cache.get_block_table(req.request_id) or [])}"
                )
                metrics.record(RequestMetric(
                    request_id=req.request_id, model=runner.model_name,
                    prompt_tokens=req.prompt_tokens,
                    completion_tokens=req.completion_tokens,
                    elapsed=elapsed, tokens_per_sec=tps,
                ))
                stats.record(
                    model=runner.model_name,
                    prompt_tokens=req.prompt_tokens,
                    completion_tokens=req.completion_tokens,
                    latency=elapsed,
                )

        except Exception as e:
            req.error = str(e)
            req.status = RequestStatus.FAILED
            logger.error(f"[{req.request_id[:8]}] Failed: {e}")
            metrics.record(RequestMetric(
                request_id=req.request_id,
                model=runner.model_name or "unknown",
                prompt_tokens=req.prompt_tokens,
                completion_tokens=0, elapsed=0,
                tokens_per_sec=0, status="failed",
            ))
            stats.record(
                model=runner.model_name or "unknown",
                prompt_tokens=req.prompt_tokens,
                completion_tokens=0, latency=0, failed=True,
            )
            if req.stream and req.token_queue:
                await req.token_queue.put(None)
        finally:
            # Always free KV cache blocks when sequence is done
            kv_cache.free_sequence(req.request_id)
            await self.queue.complete(req)

    async def _process_streaming(self, req: InferenceRequest, loop):
        from mlx_lm.generate import stream_generate
        from mlx_lm.sample_utils import make_sampler
        sampler = make_sampler(req.temperature)
        t0 = time.time()
        full_text = []
        token_count = [0]

        def _stream():
            for response in stream_generate(
                runner.model, runner.tokenizer,
                prompt=req.prompt, max_tokens=req.max_tokens, sampler=sampler,
            ):
                full_text.append(response.text)
                token_count[0] += 1
                # Track each generated token in KV cache
                kv_cache.append_token(req.request_id)
                loop.call_soon_threadsafe(req.token_queue.put_nowait, response.text)
            loop.call_soon_threadsafe(req.token_queue.put_nowait, None)

        await loop.run_in_executor(None, _stream)
        req.result = "".join(full_text)
        req.elapsed = round(time.time() - t0, 2)
        req.completion_tokens = token_count[0]
        req.status = RequestStatus.DONE
        tps = round(req.completion_tokens / max(req.elapsed, 0.01), 1)
        logger.info(
            f"[{req.request_id[:8]}] streamed {req.completion_tokens} tokens "
            f"in {req.elapsed}s ({tps} tok/s) | "
            f"KV blocks: {len(kv_cache.get_block_table(req.request_id) or [])}"
        )
        metrics.record(RequestMetric(
            request_id=req.request_id, model=runner.model_name,
            prompt_tokens=req.prompt_tokens,
            completion_tokens=req.completion_tokens,
            elapsed=req.elapsed, tokens_per_sec=tps,
        ))
        stats.record(
            model=runner.model_name,
            prompt_tokens=req.prompt_tokens,
            completion_tokens=req.completion_tokens,
            latency=req.elapsed,
        )
