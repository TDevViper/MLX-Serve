import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
from typing import Optional, Tuple, Any
import logging, time

logger = logging.getLogger(__name__)


class ModelRunner:
    def __init__(self):
        self.model      = None
        self.tokenizer  = None
        self.model_name = None
        self.load_time  = None

    def load(self, model_name: str):
        logger.info(f"Loading model: {model_name}")
        t0 = time.time()
        self.model, self.tokenizer = load(model_name)
        self.model_name = model_name
        self.load_time  = round(time.time() - t0, 2)
        logger.info(f"Model loaded in {self.load_time}s")

    def is_loaded(self) -> bool:
        return self.model is not None

    def format_prompt(self, messages: list) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            parts = []
            for m in messages:
                role    = m["role"]
                content = m["content"]
                parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
            parts.append("<|im_start|>assistant\n")
            return "\n".join(parts)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def run(self, prompt: str, max_tokens: int = 512,
            temperature: float = 0.7) -> Tuple[str, float]:
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")
        t0 = time.time()
        response = generate(
            self.model, self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=make_sampler(temperature),
            verbose=False,
        )
        elapsed = round(time.time() - t0, 2)
        return response, elapsed

    def make_prompt_cache(self):
        """Create a fresh prompt cache object for prefix caching."""
        try:
            from mlx_lm.models.cache import make_prompt_cache
            return make_prompt_cache(self.model)
        except Exception as e:
            logger.warning(f"Prompt cache not available: {e}")
            return None

    def run_with_prefix_cache(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        prompt_cache: Any = None,
    ) -> Tuple[str, float, Any]:
        """
        Run generation with prefix caching.
        Returns (response_text, elapsed, updated_cache).
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")

        from mlx_lm.generate import stream_generate
        sampler = make_sampler(temperature)
        t0 = time.time()
        tokens = []

        try:
            kwargs = {"sampler": sampler, "max_tokens": max_tokens}
            if prompt_cache is not None:
                kwargs["prompt_cache"] = prompt_cache

            for response in stream_generate(
                self.model, self.tokenizer,
                prompt=prompt,
                **kwargs,
            ):
                tokens.append(response.text)

        except Exception as e:
            logger.warning(f"Prefix cache generation failed, falling back: {e}")
            response_text, elapsed = self.run(prompt, max_tokens, temperature)
            return response_text, elapsed, None

        elapsed = round(time.time() - t0, 2)
        return "".join(tokens), elapsed, prompt_cache

    def stream_with_cache(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ):
        """
        Generator that yields (token_text, is_done).
        Uses prefix cache internally for efficiency.
        """
        from mlx_lm.generate import stream_generate
        sampler = make_sampler(temperature)

        for response in stream_generate(
            self.model, self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
        ):
            yield response.text, False
        yield "", True


# Global singleton
runner = ModelRunner()
