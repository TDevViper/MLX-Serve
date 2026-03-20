"""
Embedding engine using sentence-transformers.
Provides /v1/embeddings endpoint compatible with OpenAI API.
"""
import logging
import time
from typing import List, Optional

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self):
        self._model = None
        self._model_name = None
        self._load_time = None

    def load(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {model_name}")
        t0 = time.time()
        self._model = SentenceTransformer(model_name)
        self._model_name = model_name
        self._load_time = round(time.time() - t0, 2)
        dim = self._model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model loaded in {self._load_time}s — dim={dim}")

    def is_loaded(self) -> bool:
        return self._model is not None

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not self.is_loaded():
            raise RuntimeError("Embedding model not loaded")
        vectors = self._model.encode(texts, convert_to_numpy=True)
        return vectors.tolist()

    def dim(self) -> Optional[int]:
        if not self.is_loaded():
            return None
        return self._model.get_sentence_embedding_dimension()

    @property
    def model_name(self) -> Optional[str]:
        return self._model_name

# Global singleton
embedder = Embedder()
