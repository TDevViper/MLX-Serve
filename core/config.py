from pydantic import BaseModel
from typing import Optional
import yaml, os

class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    max_concurrent_requests: int = 10

class ModelConfig(BaseModel):
    default: str = "mlx-community/Qwen1.5-0.5B-Chat"
    max_tokens: int = 2048
    dtype: str = "float16"

class SchedulerConfig(BaseModel):
    max_batch_size: int = 8
    max_waiting_tokens: int = 20000

class KVCacheConfig(BaseModel):
    block_size: int = 16
    max_blocks: int = 512

class Config(BaseModel):
    server: ServerConfig = ServerConfig()
    model: ModelConfig = ModelConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    kv_cache: KVCacheConfig = KVCacheConfig()

def load_config(path: str = "configs/default.yaml") -> Config:
    if not os.path.exists(path):
        return Config()
    with open(path) as f:
        data = yaml.safe_load(f)
    return Config(**data)
