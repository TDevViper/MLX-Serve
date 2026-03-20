import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
from typing import Optional
import logging, time

logger = logging.getLogger(__name__)

class ModelRunner:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = None
        self.load_time = None

    def load(self, model_name: str):
        logger.info(f"Loading model: {model_name}")
        t0 = time.time()
        self.model, self.tokenizer = load(model_name)
        self.model_name = model_name
        self.load_time = round(time.time() - t0, 2)
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
                if m["role"] == "system":
                    parts.append(f"<|im_start|>system\n{m['content']}<|im_end|>")
                elif m["role"] == "user":
                    parts.append(f"<|im_start|>user\n{m['content']}<|im_end|>")
                elif m["role"] == "assistant":
                    parts.append(f"<|im_start|>assistant\n{m['content']}<|im_end|>")
            parts.append("<|im_start|>assistant\n")
            return "\n".join(parts)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def run(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7):
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")
        t0 = time.time()
        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=make_sampler(temperature),
            verbose=False,
        )
        elapsed = round(time.time() - t0, 2)
        return response, elapsed

runner = ModelRunner()
