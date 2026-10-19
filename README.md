# JunkInfer

Fast local inference of large mixture-of-experts models on junk-priced GPUs.
First target: **Qwen3.6-35B-A3B and its fine-tunes (`qwen35moe`) on the AMD RX 470/570/580 8 GB (gfx803, Polaris)**.

Status: **usable MVP** on one RX 580 (OpenAI- and Anthropic-compatible server, tool calls, thinking, prompt
caching, full 262k context). Measured on the dev box (EXL3 4.0 bpw, q8_0 KV), October 2026:

| | short context | 100k | 245k |
|---|---|---|---|
| decode | ~45 t/s | ~34 t/s | ~31 t/s |
| prefill | ~450 t/s (7k prompt) | ~212 t/s (117k prompt) | ~95-115 t/s |

Above 96k context the decode attention is sparse with tail estimation (KL vs full attention ~0.001, below the
q8_0-KV bar); `JI_SPARSE=0` turns it off.

## Targets (dev box: Ryzen 5 2600, 32 GB DDR4-3000, RX 580 8 GB, PCIe 3.0 x16)

| | single RX 580 | dual RX 580 |
|---|---|---|
| decode, any depth up to 262k | >= 50 t/s | >= 80 t/s |
| prefill at zero context | as fast as physics allows (~600-700 t/s ceiling) | >= 1000 t/s |
| prefill at 262k context | as fast as physics allows | >= 500 t/s |
| weights quality | not below EXL3 3.5 bpw (measured KLD vs BF16) | same |
| KV-cache quality | not below q8_0 KV (measured KLD vs BF16) | same |

The release must run on one card. A second card is an optional speed-up.

## Using it

Requirements (see `engineering-SIREN/current/` for the details): an RX 470/570/580 8 GB with RADV (Mesa) Vulkan,
32 GB RAM, the GTT boot parameters `ttm.pages_limit=6291456 amdgpu.gttsize=24576` (experts are pinned in host
memory), the engine built from `engine/` (Vulkan), and the model as GGUF (`tools/exl3/exl3_to_gguf.py` converts the
community EXL3 4.0 bpw weights).

```
scripts/serve.sh                                     # 127.0.0.1:8080, thinking on, 262k context
JI_HOST=0.0.0.0 JI_API_KEY=secret scripts/serve.sh  # LAN, with a key
JI_THINKING=off scripts/serve.sh                     # thinking off by default (per request: enable_thinking)
```

Other options are listed at the top of `scripts/serve.sh`; `scripts/junkinfer.service` is a systemd unit.
Loading takes ~1.5 min. One request runs at a time (others queue); the host-RAM prompt cache keeps recent
conversations, so switching between a few chats or agent sessions does not re-read them from scratch.

Clients:

- **OpenAI-compatible** (SDKs, Open WebUI, coding agents such as Cline / Roo / OpenCode / Continue): base URL
  `http://HOST:8080/v1`, any API key unless `JI_API_KEY` is set, model `qwen3.6-35b-a3b`.
  Chat Completions (streaming, tools incl. parallel calls and `tool_choice`, `response_format` JSON / JSON schema,
  logprobs), Completions and Responses.
- **Anthropic-compatible** (Anthropic SDK, Claude-Code-style clients): `ANTHROPIC_BASE_URL=http://HOST:8080`,
  Messages API with tools, `tool_choice`, thinking blocks and streaming.
- Thinking: on by default; the reasoning is returned separately (`reasoning_content` / Anthropic thinking blocks).
  Turn it off per request with `"chat_template_kwargs": {"enable_thinking": false}`.

`scripts/api_suite.py [base_url] [--long]` (needs `pip install openai anthropic`) checks all of the above against a
running server: 45 checks, a few minutes; `--long` adds a ~117k-token needle question.

## Layout

```
engineering-SIREN/   the engineering record (start at engineering-SIREN/index.md)
engine/              the inference engine (llama.cpp fork + the JunkInfer decode engine, src/junkinfer)
scripts/             serve.sh, the API suite, benchmarks
ref/                 the one-time BF16 reference run (VENE, RTX 3090) and its tools
LICENSE              Apache-2.0
NOTICES.md           third-party code and ideas this project uses
```

## Quality gates

Every change that touches numerics is measured as KL(BF16 || JunkInfer) against fixed
reference files made once from the BF16 model (`ref/`). See `engineering-SIREN/current/quality-gates.md`.
