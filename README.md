<div align="center">

<img src="https://capsule-render.vercel.app/api?type=venom&color=0:0a0a0f,40:ff9500,100:0a0a0f&height=220&section=header&text=MLX-Serve%20⚡&fontSize=72&fontColor=ff9500&animation=fadeIn&fontAlignY=52&desc=OpenAI-compatible%20LLM%20inference%20for%20Apple%20Silicon&descSize=18&descAlignY=72&descColor=aaaacc"/>

<img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=700&size=17&duration=2000&pause=800&color=FF9500&center=true&vCenter=true&width=780&lines=~90+tok%2Fs+flat+across+all+concurrency+levels;2.2x+faster+than+Ollama+on+Apple+Silicon;PagedAttention+%7C+Prefix+Caching+%7C+Continuous+Batching;OpenAI-compatible+%E2%80%94+drop-in+replacement;The+serving+layer+between+FineTuneKit+and+ASTRA" alt="Typing SVG" />

<br/>

[![License](https://img.shields.io/badge/license-MIT-ff9500?style=for-the-badge&labelColor=0a0a0f)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-ff9500?style=for-the-badge&logo=python&logoColor=white&labelColor=0a0a0f)](https://python.org)
[![Platform](https://img.shields.io/badge/Apple_Silicon-M1%2FM2%2FM3%2FM4-ff9500?style=for-the-badge&logo=apple&logoColor=white&labelColor=0a0a0f)](https://github.com/TDevViper/MLX-Serve)
[![OpenAI Compatible](https://img.shields.io/badge/OpenAI-Compatible-ff9500?style=for-the-badge&logo=openai&logoColor=white&labelColor=0a0a0f)](https://github.com/TDevViper/MLX-Serve)

</div>

---

> The missing piece between training ([**FineTuneKit**](https://github.com/TDevViper/FineTuneKit)) and orchestration ([**ASTRA**](https://github.com/TDevViper/ASTRA)) — a production-grade serving layer for on-device LLMs.

```
FineTuneKit  →  fine-tune a model on your data
MLX-Serve    →  serve it at production throughput    ← you are here
ASTRA        →  build an agent on top of it
```

---

## `> run_benchmark.py`

> Apple M4 · MLX-Serve vs Ollama · 128 output tokens · 3 rounds per concurrency level

```
Concurrency    MLX-Serve    Ollama    MLX-Serve lat    Ollama lat    Speedup
───────────────────────────────────────────────────────────────────────────
1 user         84.4 tok/s   37.8      1.52s            3.65s         2.2x ⚡
2 users        89.8 tok/s   43.5      2.18s            4.42s         2.1x ⚡
4 users        91.4 tok/s   43.7      3.51s            7.28s         2.1x ⚡
8 users        89.4 tok/s   39.5      6.38s            14.52s        2.3x ⚡
```

**MLX-Serve** sustains ~90 tok/s flat across all concurrency levels via continuous batching.
**Ollama** degrades under load — latency doubles from 1 → 8 users. MLX-Serve scales linearly.

```
Throughput (tok/s)
100 ┤
 90 ┤  ●━━━━━━●━━━━━━●━━━━━━●   MLX-Serve (~90 flat)
 80 ┤
 70 ┤
 60 ┤
 50 ┤
 40 ┤  ○━━━━━━○━━━━━━○━━━━━━○   Ollama (degrades)
 30 ┤
    └───────┬──────┬──────┬──────┬
           1u     2u     4u     8u     concurrency
```

---

## `> cat architecture.md`

```
                        ┌─────────────────────────────────┐
                        │         FastAPI Server           │
                        │  /v1/chat  /v1/completions  ...  │
                        └──────────────┬──────────────────┘
                                       │
                        ┌──────────────▼──────────────────┐
                        │         Request Queue            │
                        │   SSE Streaming · Concurrency    │
                        └──────────────┬──────────────────┘
                                       │
              ┌────────────────────────▼──────────────────────────┐
              │                 Continuous Batcher                  │
              │     Iteration-level scheduling · Multi-sequence     │
              └──────┬──────────────────────────────────┬─────────┘
                     │                                  │
        ┌────────────▼──────────┐          ┌────────────▼──────────┐
        │    Prefix Cache       │          │    KV Cache Manager    │
        │  9x speedup on shared │          │  PagedAttention-style  │
        │  prompt prefixes      │          │  LRU eviction blocks   │
        └────────────┬──────────┘          └────────────┬──────────┘
                     │                                  │
              ┌──────▼──────────────────────────────────▼─────────┐
              │               MLX Inference Engine                  │
              │          Apple Silicon Metal GPU · Native           │
              └──────────────────────────────────────────────────┘
                                       │
              ┌────────────────────────▼──────────────────────────┐
              │           Memory Monitor + Preemptor               │
              │       Pressure levels · Auto-preemption            │
              └────────────────────────────────────────────────────┘
```

---

## `> cat features.md`

| Feature | Detail |
|---------|--------|
| OpenAI-compatible API | `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/embeddings` |
| MLX backend | Runs natively on Apple Silicon Metal GPU |
| Sub-second load time | Model ready in under 1 second |
| Request queue + SSE streaming | Token-by-token streaming, concurrent handling |
| Prometheus metrics | `/metrics` — tokens/sec, latency, p99 |
| KV cache manager | PagedAttention-style block allocator with LRU eviction |
| Memory monitor + preemption | Pressure levels, auto-preemptor |
| Continuous batching | Iteration-level scheduling, multiple sequences per step |
| Prefix caching | **9x** iteration speedup on shared prompt prefixes |
| React dashboard | Live metrics, charts, queue depth, memory gauge |

---

## `> ./quickstart.sh`

```bash
git clone https://github.com/TDevViper/MLX-Serve.git
cd MLX-Serve

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python main.py
```

Drop-in OpenAI replacement — point any existing client at `http://localhost:8000`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="local")

response = client.chat.completions.create(
    model="mlx-community/Llama-3.2-3B-Instruct-4bit",
    messages=[{"role": "user", "content": "Explain KV cache eviction."}]
)
```

---

## `> GET /routes`

```
GET   /health
GET   /metrics
GET   /v1/models
POST  /v1/chat/completions
POST  /v1/chat/completions/batched
POST  /v1/completions
POST  /v1/embeddings
GET   /v1/stats
GET   /v1/kv_cache
GET   /v1/memory
GET   /v1/batcher
GET   /v1/prefix_cache
```

---

## `> cat requirements.txt`

```
Apple Silicon Mac (M1 / M2 / M3 / M4)
Python 3.10+
Node.js 18+   # dashboard only
```

---

<div align="center">

```
╔══════════════════════════════════════════════════════════════════╗
║  ~90 tok/s. Flat. Across every concurrency level.               ║
║  Not a benchmark artifact. That's the architecture working.      ║
╚══════════════════════════════════════════════════════════════════╝
```

[![GitHub](https://img.shields.io/badge/GitHub-MLX--Serve-ff9500?style=for-the-badge&logo=github&logoColor=white&labelColor=0a0a0f)](https://github.com/TDevViper/MLX-Serve)
[![FineTuneKit](https://img.shields.io/badge/→_FineTuneKit-fine--tune_your_model-555570?style=for-the-badge&labelColor=0a0a0f)](https://github.com/TDevViper/FineTuneKit)
[![ASTRA](https://img.shields.io/badge/→_ASTRA-orchestrate_on_top-555570?style=for-the-badge&labelColor=0a0a0f)](https://github.com/TDevViper/ASTRA)

<br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0a0a0f,100:ff9500&height=120&section=footer"/>

</div>
