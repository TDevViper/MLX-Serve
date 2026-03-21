<div align="center">

<img src="https://capsule-render.vercel.app/api?type=venom&color=0:0d0d0d,50:00f7ff,100:0d0d0d&height=220&section=header&text=ARNAV%20YADAV&fontSize=65&fontColor=00f7ff&animation=fadeIn&fontAlignY=55&desc=TDevViper%20%2F%2F%20AI%20Systems%20Engineer&descSize=20&descAlignY=75&descColor=ffffff"/>

<img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=700&size=20&duration=2000&pause=800&color=00F7FF&center=true&vCenter=true&width=750&lines=Building+systems+that+think;AI+%2B+Backend+%2B+Security+%2B+Correctness;FastAPI+%7C+Async+%7C+LLMs+%7C+eBPF;MLX-Serve%3A+2.2x+faster+than+Ollama+on+Apple+Silicon;From+fine-tune+%E2%86%92+serve+%E2%86%92+orchestrate;I+write+code+I+can+defend+in+any+room" alt="Typing SVG" />

<br/>

[![GitHub followers](https://img.shields.io/github/followers/TDevViper?style=for-the-badge&logo=github&color=00f7ff&labelColor=0d0d0d)](https://github.com/TDevViper)
[![Email](https://img.shields.io/badge/Email-arnavyadavop%40gmail.com-00f7ff?style=for-the-badge&logo=gmail&logoColor=white&labelColor=0d0d0d)](mailto:arnavyadavop@gmail.com)
![Profile views](https://komarev.com/ghpvc/?username=TDevViper&style=for-the-badge&color=00f7ff&labelColor=0d0d0d)

</div>

---

<img align="right" width="380" src="https://media.giphy.com/media/qgQUggAC3Pfv687qPC/giphy.gif"/>

### Who I am

I'm a CS student from India who builds things that are more than they need to be.

I've built a full local AI ecosystem from scratch — fine-tuning to serving to orchestration, all on-device, all production-grade:

**[FineTuneKit](https://github.com/TDevViper/FineTuneKit)** → fine-tune models on your data · **[MLX-Serve ⚡](https://github.com/TDevViper/MLX-Serve)** → serve them at 2.2× Ollama's throughput on Apple Silicon · **[ASTRA](https://github.com/TDevViper/Astra_Presonal_ai)** → build agents on top.

**ASTRA** went through a full Staff+ engineering audit — from 62/100 to 93/100 across five phases of critical fixes, security hardening, and architectural refactors. **MLX-Serve** sustains ~90 tok/s flat under concurrent load via continuous batching, KV cache management, and prefix caching — while Ollama's latency doubles at 8 users.

I care about systems correctness more than feature count. I care about *why* a design decision was made more than *that* it was made. I write code I can defend in a room full of people smarter than me.

What sets me apart: I don't stop at "it works." I ask — *will it hold under concurrent load? Does the security model survive adversarial input? Can someone else extend this without touching the core?* Those questions shaped every decision in every project.

Currently exploring: **computational neuroscience + VR + brain-signal analysis**

<br clear="right"/>

---

## Stack

<div align="center">

<img src="https://skillicons.dev/icons?i=python,fastapi,react,docker,linux,git,redis,sqlite&theme=dark" />
<br/>
<img src="https://skillicons.dev/icons?i=js,nodejs,rust,typescript,bash,vscode,github,figma&theme=dark" />

</div>

<br/>

```python
arnav = {
    "languages":   ["Python", "JavaScript", "SQL", "Rust (learning)"],
    "backend":     ["FastAPI", "async/await", "WebSocket", "SSE"],
    "ai_stack":    ["MLX", "Ollama", "ChromaDB", "FAISS", "LLMBackend abstraction"],
    "infra":       ["Docker Compose", "Redis", "SQLite", "OpenTelemetry"],
    "security":    ["HMAC tokens", "injection filters", "sandboxed execution"],
    "exploring":   ["eBPF", "EEG signal processing", "distributed systems"],
    "region":      "India 🇮🇳",
    "status":      "building + applying to interesting problems"
}
```

---

## ASTRA — Local Personal AI System

<div align="center">
<img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExcDd4NHBxazVhbWJ6aTI3Y3pxbWxwdDZxMTcxYjR6Y3ZpaGZuM2diZyZlcD12MV9pbnRlcm5hbEdpZiZjdD1n/L1R1tvI9svkIWwpVYr/giphy.gif" width="600"/>
</div>

<br/>

> A production-quality local AI backend. 100% on-device. No cloud. No data leaves your machine.

```
User Input → Sanitize → Cache → Intent → Tools → Memory
          → Web Search → ReAct Agent → LLM → Critic → Reply
```

| What it has | Why it matters |
|---|---|
| Modular `PipelineRegistry` | Add capabilities without touching `brain.py` |
| Pluggable `LLMBackend` ABC | Swap Ollama → OpenAI → vLLM in 4 method overrides |
| Per-request `RequestContext` | No shared state = no race conditions |
| `MemoryTransaction` | Atomic batch writes — no partial saves under failure |
| HMAC-signed tool tokens | Client `approved: true` is rejected at the server |
| 3-session feedback quality gate | One accidental thumbs-up can't poison the dataset |
| Async OTel observability | Non-blocking traces on every request, zero overhead |
| Injection filter layer | Adversarial prompt inputs are caught before LLM reach |

**[→ View ASTRA on GitHub](https://github.com/TDevViper/Astra_Presonal_ai)**

---

## MLX-Serve ⚡ — Apple Silicon Inference Server

> The missing piece between fine-tuning and orchestration — a production-grade serving layer for on-device LLMs.

OpenAI-compatible API. Runs natively on Apple Silicon Metal GPU. Built because Ollama wasn't fast enough and I wanted to understand *why*.

### Benchmark — Apple M4 · 128 output tokens · 3 rounds per concurrency level

| Concurrency | MLX-Serve tok/s | Ollama tok/s | MLX-Serve lat | Ollama lat | Speedup |
|---|---|---|---|---|---|
| 1 user | 84.4 | 37.8 | 1.52s | 3.65s | **2.2×** |
| 2 users | 89.8 | 43.5 | 2.18s | 4.42s | **2.1×** |
| 4 users | 91.4 | 43.7 | 3.51s | 7.28s | **2.1×** |
| 8 users | 89.4 | 39.5 | 6.38s | 14.52s | **2.3×** |

MLX-Serve holds ~90 tok/s flat across all concurrency levels. Ollama's latency doubles from 1→8 users. MLX-Serve scales linearly.

**Why it stays flat:** continuous batching schedules at iteration-level, not request-level. The KV cache manager uses a PagedAttention-style block allocator with LRU eviction. Prefix caching gives 9× speedup on shared prompt prefixes.

| Feature | Detail |
|---|---|
| OpenAI-compatible API | `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/embeddings` |
| Continuous batching | Iteration-level scheduling, multiple sequences per step |
| KV cache manager | PagedAttention-style block allocator with LRU eviction |
| Prefix caching | 9× iteration speedup on shared prompt prefixes |
| Memory monitor | Pressure levels + auto-preemption |
| Prometheus metrics | `/metrics`, tokens/sec, latency, p99 |
| Sub-second load time | Model ready in under 1 second |
| React dashboard | Live metrics, queue depth, memory gauge |

```
FineTuneKit  →  fine-tune a model on your data
MLX-Serve    →  serve it at production throughput   ← you are here
ASTRA        →  build an agent on top of it
```

**[→ View MLX-Serve on GitHub](https://github.com/TDevViper/MLX-Serve)**

---

## Engineering Audit Trail

This is the part most people skip. I didn't.

ASTRA went through a full Staff+ engineering audit — concurrency correctness, security surface, architectural gaps, performance ceilings. Then I fixed everything, in phases, with documented reasoning for every decision.

```
Phase 0    →  62/100   Initial build
Phase 1–3  →  72/100   FastAPI migration, OpenTelemetry, structured logging
CR fixes   →  78/100   Concurrent history bug, double LLM call, Flask removal
Security   →  83/100   Injection filter, signed tokens, session-scoped cache
Arch fixes →  88/100   Async observability, lifespan management, TruthGuard isolation
Phase C    →  93/100   Pipeline registry, LLM abstraction, parallel tools, quality gate
```

**The remaining 7 points:** PostgreSQL migration and JWT multi-user auth — infrastructure changes, not code changes. I know exactly what they require. I chose to ship instead of over-engineering for zero users. That's also an architectural decision.

---

## What the audit actually found

Most people treat a code review as a checklist. I treated it as a stress test. Here's what surfaced:

| Issue found | Root cause | Fix applied |
|---|---|---|
| Concurrent history corruption | Shared mutable list across async requests | Per-request `RequestContext` isolation |
| Double LLM invocation | Control flow ambiguity in pipeline | Explicit state machine with single exit |
| Client-trusted `approved` flag | No server-side token validation | HMAC-signed tool tokens |
| Partial memory saves on failure | Non-atomic write operations | `MemoryTransaction` with rollback |
| Blocking OTel traces | Synchronous span flush in hot path | Async span export, fire-and-forget |
| Dataset poisoning via single feedback | No quality gate on feedback loop | 3-session confirmation threshold |

---

## GitHub Stats

<div align="center">

<img src="https://github-readme-stats.vercel.app/api?username=TDevViper&show_icons=true&theme=tokyonight&hide_border=true&bg_color=0d0d0d&title_color=00f7ff&icon_color=00f7ff&text_color=ffffff&ring_color=00f7ff" height="160"/>
<img src="https://github-readme-streak-stats.herokuapp.com/?user=TDevViper&theme=tokyonight&hide_border=true&background=0d0d0d&ring=00f7ff&fire=00f7ff&currStreakLabel=00f7ff" height="160"/>

<br/>

<img src="https://github-readme-stats.vercel.app/api/top-langs/?username=TDevViper&layout=compact&theme=tokyonight&hide_border=true&bg_color=0d0d0d&title_color=00f7ff&text_color=ffffff" height="140"/>

</div>

---

## Contribution Graph

<div align="center">
<img src="https://github-readme-activity-graph.vercel.app/graph?username=TDevViper&theme=react-dark&bg_color=0d0d0d&color=00f7ff&line=00f7ff&point=ffffff&hide_border=true" width="100%"/>
</div>

---

## What I'm looking for

<div align="center">

| Interested in | Not interested in |
|---|---|
| Hard systems problems | Tutorials disguised as projects |
| Security + AI intersection | Another CRUD app |
| Neuroscience + data science | Surface-level ML wrappers |
| Code review + architecture | Shipping fast without understanding |
| Research with real engineering depth | Hype without substance |
| Correctness-first design | "It works on my machine" |

</div>

---

## Currently reading the internals of

```bash
$ ls ~/deep_dives/
eBPF_kernel_tracing/
EEG_signal_processing/
distributed_systems_patterns/
vLLM_continuous_batching/
MLX_metal_kernel_internals/
computational_neuroscience_intro/
VR_brain_signal_interface_research/
```

---

<div align="center">

<img src="https://quotes-github-readme.vercel.app/api?type=horizontal&theme=tokyonight&quote=If%20it%20doesn%27t%20survive%20a%20concurrent%20load%20test%2C%20it%20doesn%27t%20exist%20yet.&author=personal%20rule" width="700"/>

<br/><br/>

**Open to research collabs, interesting problems, and people who care about correctness.**

[![Email](https://img.shields.io/badge/email-arnavyadavop%40gmail.com-00f7ff?style=for-the-badge&logo=gmail&logoColor=white&labelColor=0d0d0d)](mailto:arnavyadavop@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-TDevViper-00f7ff?style=for-the-badge&logo=github&logoColor=white&labelColor=0d0d0d)](https://github.com/TDevViper)

<br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d0d0d,100:00f7ff&height=100&section=footer"/>

</div>
