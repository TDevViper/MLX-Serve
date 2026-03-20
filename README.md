# MLX-Serve ⚡

An OpenAI-compatible LLM inference server built for Apple Silicon using MLX.

> The missing piece between training (FineTuneKit) and orchestration (ASTRA) — a production-grade serving layer for on-device LLMs.

## Features

- ✅ OpenAI-compatible API (`/v1/chat/completions`, `/v1/models`)
- ✅ MLX backend — runs natively on Apple Silicon Metal GPU
- ✅ Loads models in under 1 second
- 🔄 Request queue + concurrent handling (Week 2)
- 🔄 Token-by-token SSE streaming (Week 2)
- 🔄 KV cache manager (Week 5-7)
- 🔄 Continuous batching (Week 8-9)
- 🔄 Real-time dashboard — tokens/sec, queue depth, memory (Week 10-11)

## Quickstart
```bash
git clone https://github.com/TDevViper/MLX-Serve.git
cd MLX-Serve
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Usage
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen1.5-0.5B-Chat",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 200
  }'
```

## OpenAI SDK compatible
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
response = client.chat.completions.create(
    model="mlx-community/Qwen1.5-0.5B-Chat",
    messages=[{"role": "user", "content": "What is MLX?"}],
)
print(response.choices[0].message.content)
```

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- Python 3.10+

## Project Structure
```
mlx-serve/
├── api/          # FastAPI server + endpoints
├── core/         # Config + OpenAI-compatible request/response models
├── engine/       # Model runner (MLX inference)
├── scheduler/    # Request queue + batching (Week 2)
├── configs/      # YAML config
└── tests/        # Test suite
```

---
Built on Apple Silicon · Part of the FineTuneKit ecosystem
