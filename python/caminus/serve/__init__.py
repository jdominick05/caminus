"""K3 server front end (docs/agentic/spec-server.md, track P of docs/AGENTIC-PLAN.md).

CPU-only pieces: GGUF vocab reader, tokenizer and streaming detokenizer (tokenizer.py), the chat template rendered as
transformers renders it plus literal-tag marking and the per-model effort pin (template.py), request normalization
for OpenAI and Anthropic requests (normalize.py), the streaming reasoning / content / Qwen XML tool-call parser with
JSON-schema coercion (toolparse.py), its OpenAI delta and Anthropic content-block shapes (deltas.py), the host
reference sampler with Philox draws and speculative acceptance (sampler.py), and a CPU MockEngine with the K3 pass
interface (mock_engine.py).

Serving (step S4): the EngineSession interface the real engine implements in M6/M7, and MockSession over the
MockEngine (engine.py); the per-request generation loop with stop-string holdback and forced tool calls
(generate.py); the single engine worker thread with its FIFO queue and cancel flags (worker.py); and the stdlib HTTP
server with the OpenAI and Anthropic routes, SSE, usage, timings and the JSONL log (server.py). Launcher:
tools/k3_serve.py."""
