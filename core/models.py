from pydantic import BaseModel
from typing import Optional, List, Literal
import time, uuid

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "mlx-community/Qwen1.5-0.5B-Chat"
    messages: List[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 1.0
    stream: bool = False
    stop: Optional[List[str]] = None

class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage

    @classmethod
    def build(cls, model: str, content: str, prompt_tokens: int, completion_tokens: int):
        return cls(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=model,
            choices=[ChatCompletionChoice(
                message=ChatMessage(role="assistant", content=content)
            )],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )
        )

class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "mlx-serve"

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard]
