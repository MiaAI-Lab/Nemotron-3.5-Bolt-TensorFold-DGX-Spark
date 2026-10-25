<h1 align="center">NVIDIA Nemotron 3.5 Bolt // TensorFold // DGX Spark</h1>

```text
███╗   ██╗███████╗███╗   ███╗ ██████╗ ████████╗██████╗  ██████╗ ███╗   ██╗
████╗  ██║██╔════╝████╗ ████║██╔═══██╗╚══██╔══╝██╔══██╗██╔═══██╗████╗  ██║
██╔██╗ ██║█████╗  ██╔████╔██║██║   ██║   ██║   ██████╔╝██║   ██║██╔██╗ ██║
██║╚██╗██║██╔══╝  ██║╚██╔╝██║██║   ██║   ██║   ██╔══██╗██║   ██║██║╚██╗██║
██║ ╚████║███████╗██║ ╚═╝ ██║╚██████╔╝   ██║   ██║  ██║╚██████╔╝██║ ╚████║
╚═╝  ╚═══╝╚══════╝╚═╝     ╚═╝ ╚═════╝    ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝

██████╗    ███████╗  ██████╗  ██████╗ ██╗     ████████╗
╚════██╗   ██╔════╝  ██╔══██╗██╔═══██╗██║     ╚══██╔══╝
 █████╔╝   ███████╗  ██████╔╝██║   ██║██║        ██║
 ╚═══██╗   ╚════██║  ██╔══██╗██║   ██║██║        ██║
██████╔╝██╗███████║  ██████╔╝╚██████╔╝███████╗   ██║
╚═════╝ ╚═╝╚══════╝  ╚═════╝  ╚═════╝ ╚══════╝   ╚═╝
```

<p align="center">
  <sub>by <a href="https://x.com/MiaAI_lab">Mia'a AI Lab</a></sub>
  <br><br>
  <a href="https://github.com/sponsors/MiaAI-Lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Sponsor%20me%20on%20GitHub-181717?style=for-the-badge&logo=githubsponsors&logoColor=white" alt="Sponsor me on GitHub" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
  <a href="https://x.com/MiaAI_lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow Mia on X" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
</p>

Serve **[NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB)**
on one DGX Spark (GB10, 128 GB) with [TensorFold](https://github.com/ashhart/TensorFold)'s Zig engine,
`tensorfold-native`. The weights are the 24GB pack: IQ2_XXS experts, NVFP4 Mamba projections, Q8_0 attention
and latent projections, C-RADIO vision. Loading it with the default settings takes 51.1 GB of unified memory, so most of
the Spark stays free for other work. The server takes text, images and video (see
[Images and video](#images-and-video)), with up to 16 streams sharing one copy of the weights and a 262,144-token window each. The script layout matches
[Qwen3.8 Flash Next on one DGX Spark](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold):
`./start.sh` prepares the image, `scripts/config.sh` holds the settings, `patches/` is applied onto a pinned
TensorFold commit.

TensorFold's Nemotron family serves Lightning 30B-A3B and, since 1.1.0, Bolt itself (its own GGML kernels, without
the LiLiCorr drafter); this checkpoint is `nemotron_h_puzzle` (hidden 4096, 88
blocks, per-layer expert counts, IQ2_XXS experts). `patches/0002-register-bolt.patch` registers a Bolt backend,
`engine/bolt_native.zig` (with its libbolt binding, `engine/bolt_api.zig`), in the native CUDA host, as the family `nemotron_h_puzzle`. The backend is a TensorFold *lane backend*: the HTTP server, chat
template, thinking budget, round loop, stop handling and keyed sampler are TensorFold's own. The forward pass lives in
`libbolt.so` (`engine/*.cu`), loaded with `dlopen`, because the native kernels only cover Lightning's shapes.
`./start.sh` checks the snapshot against [`contract/bolt.json`](contract/bolt.json) and serves it.

## Speed

Decode measured on 2026-10-20 with TensorFold's [`tools/bench_concurrent.py`](https://github.com/ashhart/TensorFold/blob/76aafcf274e60a5e31a59ce145b2f71f7b366e9d/tools/bench_concurrent.py)
against the running server (this build's defaults, thinking off): N clients send the same
prompt at once, 256 tokens each, medians of three runs. "aggregate" is the server's tokens a second over all N, "each
stream" one client's decode rate, TTFT the slowest client's time to its first token. The prefill table is from an
earlier run with [sparkDash](https://github.com/MiaAI-Lab/sparkDash) (the prompt path has not changed since). The
1.0.12 build measured the same way on the same day is in [`CHANGELOG.md`](CHANGELOG.md), cell for cell.

Decode, greedy (`temperature 0`):

| clients | prose aggregate | prose each | code aggregate | code each | TTFT |
| ---: | ---: | ---: | ---: | ---: | ---: |
| ×1 | 73.7 tok/s | 73.7 tok/s | 137.3 tok/s | 137.3 tok/s | 50–70 ms |
| ×8 | 240.3 tok/s | 32.7 tok/s | 342.0 tok/s | 48.5 tok/s | 0.9–1.0 s |
| ×16 | 284.4 tok/s | 20.4 tok/s | 385.1 tok/s | 29.2 tok/s | 2.3–2.6 s |

Decode, sampled (`temperature 1.0`, `top_p 0.95`, seeded):

| clients | prose aggregate | prose each | code aggregate | code each | TTFT |
| ---: | ---: | ---: | ---: | ---: | ---: |
| ×1 | 70.6 tok/s | 70.6 tok/s | 133.4 tok/s | 133.4 tok/s | 50–70 ms |
| ×8 | 205.4 tok/s | 28.1 tok/s | 230.7 tok/s | 39.0 tok/s | 0.9–1.0 s |
| ×16 | 251.1 tok/s | 18.4 tok/s | 280.0 tok/s | 23.9 tok/s | 2.3–2.6 s |

Every cell's replies were token for token the same as the 1.0.12 build's, with `SSM_LAZY=0` and with `DRAFT_Q8=0`
(the tool's `token_sha256` over all three runs).

Prefill (2026-10-20) with a fresh random-word prompt and `max_tokens 1` (so the prompt cache cannot help), in
tokens a second of prompt: 2,310 to 2,410 at 2.7k tokens, 2,490 to 2,510 at 5.3k, 2,485 at 21k, 2,365 at 42k,
2,255 at 63k, 1,975 at 126k (the 1.0.12 build: 1,640, 1,690, 1,650, 1,520, and 987 at 128k). These are with
`NV4_FP4=1`, the prompt pass's two NVFP4 projections on the fp4 tensor cores with NVFP4 rows, which costs 0.8 %
perplexity on the harness in `engine/tools`; `NV4_FP4=0` keeps fp16 rows and the exact numbers at about 7 % less.
The long-context end comes from a prompt attention kernel that is 2.2 to 2.5× the old one with identical outputs.
See [`docs/speed-research.md`](docs/speed-research.md) for where the time goes and what is left to do.

Prefill, cold prompts of one request (this build, `max_tokens 1`):

| prompt | prefill | TTFT | tokens |
| ---: | ---: | ---: | ---: |
| 2.7k | 2,312 to 2,408 tok/s | 1.1 s | 2,656 to 2,663 |
| 5.3k | 2,493 to 2,508 tok/s | 2.1 s | 5,270 to 5,346 |
| 21k | 2,483 tok/s | 8.5 s | 21,014 to 21,046 |
| 42k | 2,365 tok/s | 17.8 s | 41,905 to 42,400 |
| 63k | 2,254 tok/s | 28.1 s | 63,214 |
| 126k | 1,973 tok/s | 64.1 s | 126,455 |

## How it is built

In the order it matters:

1. **Weights stay packed.** NVFP4 Mamba projections, Q8_0 attention and latent projections, IQ2_XXS experts. At
   batch one a token reads about 5 GB, not the 14.6 GB the fp16 expansion read. `bolt_mma.cu` repacks each matrix at
   load into 16-row tiles laid out so a lane fetches an `mma.m16n8k16` fragment with one 16-byte load; the dense
   GEMMs run on tensor cores; prompt blocks stage activations in shared memory, while decode and verify blocks sum K in one
   fixed order whatever the row count. NVFP4 block scales
   are exact in fp16, so they are multiplied into the weights as they are decoded and the mma accumulates straight
   through K. Scales are applied to the fp32 accumulators, so only the activation (fp16, or bf16 for IQ2) is rounded.
2. **A block of rows is one CUDA graph.** Embedding, 88 layers and the head replay as a graph; the shared expert and
   the latent projection run on side streams. A prompt goes through in 1,024-row chunks, a shared decode round in a handful of padded shapes.
3. **Rollback is cheap.** A verify window writes only its final Mamba state, to the spare of two copies, and the scan
   inputs of its rows; `keep(n)` flips a pointer, or replays `n` rows from the untouched copy for a partial accept. KV
   rows past the kept prefix are overwritten later. With `SSM_LAZY=1` there is one copy: the window writes no state at
   all, `keep(n)` only notes `n`, and the stream's next forward folds those `n` saved rows in before its own, so a
   partial accept costs no replay pass and a stream holds 0.28 GiB instead of 0.41 (the default; 3 to 8 % more tokens a
   second at 4 to 16 streams, the same replies; see the changelog).
4. **TensorFold drives the speculation.** The backend implements `verify` (one forward over every stream's pending
   token and held drafts, every row drawn with the stream's keyed sampler), `keep` and `draft`. LiLiCorr proposes
   seven tokens a stream and round, for all streams in one pass. TensorFold's cost rule chooses how many to verify
   from the window and shared-round costs the backend measures at load.
5. **Prompt cache.** TensorFold's prefix reuse is on (`PROMPT_CACHE_GIB`, default 8). The backend keeps a stream's Mamba
   state, attention rows and drafter context at the cache's marks and restores them when a prompt shares the prefix, so
   a chat turn or a repeated system prompt prefills only the new tail. Replies are identical with and without it.
6. **Routed experts** are grouped by expert on the device and run on int8 tensor cores (activations quantized per 32,
   the IQ2 grid expanded to int8 in registers); hot experts are split into work items of 16 tokens.
7. **Attention** is flash-style on bf16 tensor cores for prompts and for verify windows (one CTA per stream's window,
   keys split across CTAs). A window splits a stream's keys into chunks fixed by the stream's own length (128 keys,
   doubled until 32 chunks reach its end), never by the batch. With the GEMMs' fixed order (1), a stream's logits are
   bit-identical alone, among other streams (tested up to 16 streams of 8-row windows) and in a window of any width; the
   one-token step runs as a one-row window for the same reason, so drafted and plain decoding agree. The price is paid
   in the widest rounds: 16 streams of 8-row windows (128 rows) take about 25 % longer; 64-row rounds are unchanged.
8. **Memory follows use.** Every stream has the full 262,144-token window, but its attention rows are mapped from the
   GPU's virtual memory 4,096 tokens at a time and returned when it ends; only the 0.41 GiB of recurrent state per
   stream (0.28 with `SSM_LAZY=1`) is held from load.

The three bugs that made the old engine's drafter accept 0 of 7 are fixed: the shared expert read the *updated*
residual's norm instead of the block input's, Mamba's `dt` was clamped at 0.001 (the config's limit is 0), and the
drafter applied its grouped convolution before `o_proj` instead of after.

## At a glance

- Checkpoint: [`nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB)
  at `a326822af111290f2af5ac59ef4b5040347dabc4`. Hub `usedStorage` is 24,102,448,875 bytes (22.45 GiB), 90 safetensors
  shards. Routed and shared experts are IQ2_XXS (2.0625 bits, ggml, group 256). Mamba `in_proj` and `out_proj` are
  W4A16 NVFP4, group 16. Attention QKV/O, the latent projections and `lm_head` are Q8_0 (ggml, group 32).
- API model id: `Nemotron-3.5-Bolt` (`SERVED_NAME`), OpenAI-compatible on port 8888
- Context: **262,144 tokens** a stream, `max_position_embeddings` in `llm_config`. One sequence's bf16 attention cache
  at that window is 2.00 GiB (8 attention layers, 2 kv heads, head dim 128), mapped as it fills.
- Drafter: the checkpoint has no MTP head. NVIDIA's companion is
  [`NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr).
  This recipe serves with it (`DRAFTS=1`, about 1.3 GB, revision `4c6a5613`); `DRAFTS=0` serves one row at a time.
- `./start.sh` prepares, checks and launches `tensorfold-native`. `./stop.sh` stops it.

## The checkpoint

74B parameters, 9B active, top-9 experts, latent size 1024, shared expert width 5376, routed scaling 5.0, sigmoid
router, ReLU² on the MoE. Hidden size 4096, vocab 131072. Eighty-eight blocks:

`MEMEMEMA EMEMEMEM AEMEMEME MAEMEMEM EMEMAEME MEMEMEMA EMEMEMEM EMAEMEME MEMEMAEM EMEMEMAE MEMEMEME`

That is 40 Mamba, 40 MoE, 8 attention. Every MoE block routes 9 experts. The routed width and the expert count
change by layer:

| Expert width | MoE layers | Routed experts | MoE layers |
| ---: | ---: | ---: | ---: |
| 1536 | 1 | 240 | 12 |
| 2048 | 11 | 304 | 6 |
| 2304 | 15 | 368 | 9 |
| 2560 | 13 | 432 | 10 |
| | | 496 | 3 |

Mamba: 128 heads, head dim 64, 8 groups, state 128, conv kernel 4. The config stores the SSM cache as float32.
A sequence's Mamba state from those shapes is about 165 MiB (40 layers × 4 MiB of SSM state, plus the conv window).
The model card's serve command stores that cache in bf16. This recipe's arithmetic uses the config's float32.

Attention: 32 query heads, 2 kv heads, head dim 128. Bytes a token, bf16 K and V:

`8 layers × 2 kv heads × 128 × 2 bytes × 2 = 8192` (bf16). With `KV_DTYPE=fp8` it is 8 × 2 × (256 + 2 × 4) = 4224.

| Window | bf16 attention cache |
| ---: | ---: |
| 32,768 | 0.25 GiB |
| 131,072 | 1.00 GiB |
| 262,144 | 2.00 GiB (1.03 GiB with `KV_DTYPE=fp8`) |

Vision is C-RADIO v4-H (`model_type: radio`, 32 layers, hidden 1280, patch 16, any resolution from 1,024 to 13,312
patches) with a 2×2 pixel shuffle and an MLP into the language model's embedding space. libbolt runs it (see
[Images and video](#images-and-video)); `VISION=1` is the default and `VISION=0` serves text only.

Quantized tensors in `config.json`: IQ2_XXS 27,728 (experts, including the shared expert), W4A16_NVFP4 80 (Mamba
projections), Q8_0 113 (latent projections, attention projections, `lm_head`). `quant_method` is `modelopt`,
`quant_algo` is `MIXED_PRECISION`, producer modelopt 0.37.0.

Sampling from `generation_config.json` and the model card: temperature 1.0, top_p 0.95. Thinking is on
(`enable_thinking`). `medium_effort` is the card's medium reasoning flag. The model card's tool parser is
`qwen3_coder`. TensorFold's own tool parser is what `tensorfold-native` will use once it loads this checkpoint.

## Requirements

- A DGX Spark (or another GB10 with 128 GB unified memory). Sixteen streams need about 55 GiB `MemAvailable` at
  start in this recipe's sum: 23 GiB of weights, 3 GiB for the vision tower and its scratch, 8 GiB of
  recurrent state, 1 GiB of working attention cache, the 10 GiB floor, 2 GiB, the 8 GiB prompt cache. `KV_PIN=1` adds
  the whole attention window of every stream (16 GiB for 8 × 262,144 in bf16). `start.sh` stops below that (`MEM_NEED_GIB`).
- Docker with the NVIDIA container runtime, and your user in the `docker` group.
- Disk for a fresh machine: about 23 GiB for the checkpoint under `~/.cache/huggingface`, plus the image under
  Docker's root. `scripts/prepare.sh` checks both.
- Optional: the `hf` CLI and a Hugging Face token in `~/.cache/huggingface/token` or `HF_TOKEN`. The checkpoint
  is public.

## Quick start

```bash
git clone https://github.com/MiaAI-Lab/Nemotron-3.5-Bolt-TensorFold-DGX-Spark.git
cd Nemotron-3.5-Bolt-TensorFold-DGX-Spark
./start.sh
```

The first run builds TensorFold at `eab8582` (v1.1.4) with `patches/` and the engine in `engine/`, downloads the checkpoint
(23 GiB) and the LiLiCorr drafter (1.3 GB), checks them, starts the server and sends a smoke request. Later runs
reuse the image and the snapshots. Loading takes about 45 seconds with the files in the page cache.

```bash
./start.sh restart                 # stop, then start again
docker logs -f nemotron35-bolt-tf
DRY_RUN=1 ./start.sh               # print the docker command and change nothing
SERVE=0 ./start.sh                 # prepare and check only
DRAFTS=0 ./start.sh                # no drafter
python3 scripts/ckptcheck.py ~/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB/snapshots/a326822af111290f2af5ac59ef4b5040347dabc4
```

`./stop.sh` removes the container.

## What `start.sh` and `scripts/prepare.sh` do

1. **Setup:** `scripts/prepare.sh` when the image is not the one these settings name, or when any weight is missing from
   the Hugging Face cache (a checkpoint shard, the tokenizer, the drafter). It downloads what is absent, about 23 GiB
   for a fresh machine, and a broken download resumes on the next `./start.sh`. `PREPARE=0` skips it.
2. **Checkpoint:** `scripts/ckptcheck.py` against `contract/bolt.json` (layer schedule, quant group counts, vision
   tower, 90 shards).
3. **Checks:** the arguments (`tensorfold-native`'s own parser, in a throwaway container), the port, and free memory.
   A typo leaves a running server alone. `./start.sh restart` stops only after those checks pass.
4. **Stop here** when `SERVE=0`.
5. **Launch** (`SERVE=1`, the default): `tensorfold-native serve` with the settings from `scripts/config.sh`. If `/health`
   answers, a one-line smoke test runs and the script prints LIVE.

`FOREGROUND=1 ./start.sh` stays attached. Extra arguments go to `tensorfold-native serve` after the
defaults (`./start.sh restart --context 131072`). `DRY_RUN=1` prints the docker command and changes nothing.

**`scripts/prepare.sh`:**

1. Preflight: Docker, disk, `MemAvailable`.
2. The image `tensorfold-nemotron-bolt:zig-eab8582`: TensorFold at `TF_REF` with every `patches/*.patch` applied,
   Zig 0.17.0 on `nvcr.io/nvidia/pytorch:26.07-py3`, and `zig build -Dnvcc=… install native bolt video grammar`.
   The Dockerfile checks the installed server and libraries. It tries a prebuilt image from GitHub Container Registry first
   (`PULL=0` builds locally). No image is published for this revision, so the first run builds.
3. The checkpoint, at the pinned revision.
4. `scripts/ckptcheck.py` on the snapshot.

```bash
scripts/prepare.sh
scripts/prepare.sh --rebuild
PREPARE=1 ./start.sh restart
```

## Memory

**Loading the model with the default settings takes 51.1 GB of the Spark's unified memory** (16 streams, a 262,144-token
window each, bf16 KV, vision and drafter on). That leaves roughly 70 GB of the 121 GB for whatever else you run beside
it. The figure is the load; the attention cache of a stream grows only as its conversation does (8 KB a token, given
back when it ends) and the prompt cache (`PROMPT_CACHE_GIB`, default 8) fills with use. For a fixed footprint, `KV_PIN=1`
reserves every stream's whole window up front (see below), and `PARALLEL` or `CONTEXT` trade streams and window for memory.

Weights on disk at the pinned revision: 22.45 GiB. `CKPT_GIB` is 23.

Each of the `PARALLEL` streams holds 0.41 GiB from load: two copies of the 160 MiB fp32 SSM state (a verify window
reads one and writes the other; `SSM_LAZY=1` keeps one and the stream holds 0.28 GiB), the conv ring and the drafter's 21 MB context. Its attention rows are mapped as it
grows (8 KB a token) and given back when it ends, so sixteen streams at the full window would take 32 GiB only if all of
them filled it; `BOLT_KV_RESERVE_GIB` (default 3) is the free memory a growing stream leaves alone, and a forward that
would pass it fails instead of freezing the machine. The prompt cache (`PROMPT_CACHE_GIB`) comes on top.

`KV_PIN=1` maps the full window of every stream at load instead (`PARALLEL` × `CONTEXT` × 8 KB in bf16, half that with
`KV_DTYPE=fp8`), so the memory is certain and a start that cannot have it fails at once. Vision adds the tower's
1.7 GB, a 134 MB rows buffer and the tower's scratch (about 1 GiB at the largest picture).

`prepare.sh` and `start.sh` refuse to run when `MemAvailable` is under 10 GiB (`MEM_FLOOR_GIB`). Keep at least
10 GiB free under load. On this machine, exhausting unified memory freezes it. A smaller `PARALLEL` or `CONTEXT` is the
way to trade; `PARALLEL=1` holds 0.41 GiB of state.

## Configuration

Every setting lives in [`scripts/config.sh`](scripts/config.sh). Override it from the environment
(`PARALLEL=2 ./start.sh`) or from `.env` next to `start.sh` (`KEY=value`; the environment wins).

| Variable | Default | Meaning |
| --- | --- | --- |
| `SERVE` | `1` | `0` stops after the checkpoint check |
| `PARALLEL` | `16` | streams decoded together, 1 to 16 (also sizes libbolt's state, `BOLT_SEQS`); more requests queue |
| `CONTEXT` | `262144` | prompt + reply window of every stream, the config's maximum |
| `PROMPT_CACHE_GIB` | `8` | GiB of kept prompt states (about 0.2 GB for a 2k-token prefix, 0.5 GB at 32k); `0` turns the cache off |
| `DRAFTS` | `1` | `0` passes `--no-drafts` and does not load the LiLiCorr drafter. The checkpoint has no MTP head |
| `TEMPERATURE` / `TOP_P` | `1.0` / `0.95` | the checkpoint's `generation_config.json`. A request's own values win |
| `TOP_K` | empty | omitted. Set it to pass `--top-k` |
| `THINKING` | `1` | `--thinking`; `0` passes `--no-thinking` |
| `MAX_TOKENS` | `32768` | the reply budget when a request sets none |
| `KV_PIN` | `0` | `1` maps every stream's whole `CONTEXT` window at load and keeps it: `PARALLEL` × `CONTEXT` × 8 KB (bf16) or 4.1 KB (fp8), e.g. 8 × 262,144 = 16 GiB. The memory is then certain, and a start that cannot have it fails at once, naming `PARALLEL`, `CONTEXT` and `KV_DTYPE` |
| `KV_DTYPE` | `bf16` | `fp8` stores K and V as e4m3 with an fp32 scale per token and head: 4.1 KB a token instead of 8 KB, so twice the tokens in the same memory and half the prompt-cache bytes. Passed to libbolt as `BOLT_KV` (the native server itself refuses `--kv-dtype`). No speed cost; see below |
| `SSM_LAZY` | `1` | one copy of each stream's Mamba state instead of two: a verify window writes none, and the stream's next forward folds the kept rows in from the saved scan inputs, so a partial accept costs no replay pass and a stream holds 0.28 GiB from load instead of 0.41. Measured: the same replies token for token, 3 to 8 % more tokens a second at 4 to 16 streams, no change alone. `0` is the two-copy path (`BOLT_SSM_LAZY`) |
| `BOLT_*` | | any `BOLT_*` variable in the environment is passed into the container: libbolt's own switches, e.g. `BOLT_PACK=0` (fp32 GEMM rows), `BOLT_MOE_COOP=0`, `BOLT_ATTN_V2=0` (the first prompt attention kernel), `BOLT_PDL=0`, `BOLT_CAL=spread`, `BOLT_PROF=1`, `BOLT_STATS=1`; and the off-by-default experiments `BOLT_IQ2_WHOLE=1`, `BOLT_SHARED_DECODE=1`, `BOLT_DENSE_RING=1`, `BOLT_TILE=1` (see the changelog) |
| `NV4_FP4` | `1` | the prompt pass's `in_proj` and `out_proj` on the fp4 tensor cores with the rows quantized to NVFP4 (vLLM's NVFP4 path does the same): prefill +17 to 20 %, perplexity +0.8 % on the harness. `0` keeps fp16 rows (`BOLT_NV4_FP4`) |
| `DRAFT_Q8` | `1` | the LiLiCorr drafter's attention and kernel projections as Q8_0 instead of bf16, half the bytes a proposal reads. Every draft is checked by the target, so this changes what is proposed, never the reply; `0` keeps bf16 (`BOLT_DRAFT_Q8`). Measured: the same replies and the same speed within noise either way (a proposal takes 6.4 ms against 7.2); Q8_0 keeps 200 MB |
| `VISION` | `1` | loads TensorFold's RADIO tower and projector (1.7 GB) so requests can carry images and video: `--vision` and the four limits below. `0` serves text only and skips the weights |
| `VISION_MAX_IMAGES` / `VISION_MAX_VIDEOS` | `64` / `4` | pictures and clips one request may hold |
| `VISION_IMAGE_TOKENS` / `VISION_VIDEO_TOKENS` | `262144` / `131072` | the visual tokens a request's images (and its videos) share; each gets an equal part of its total |
| `VISION_URLS` | `0` | `1` also fetches public HTTPS image and video URLs (`--vision-urls`) |
| `SERVED_NAME` | `Nemotron-3.5-Bolt` | the model id in `/v1/models` |
| `PORT` / `HOST` | `8888` / `0.0.0.0` | where the API listens |
| `TENSORFOLD_MEMORY_RESERVE_GIB` | `10` | the floor `prepare.sh` and `start.sh` refuse to start below |
| `MODEL_REVISION` | `a326822a…` | the checkpoint commit this contract was read from |

### Images and video

Send pictures as base64 data URLs in the OpenAI content-part form (`image_url`, or `video_url` for video):

```bash
IMG=$(base64 -w0 photo.jpg)
curl -s localhost:8888/v1/chat/completions -H 'Content-Type: application/json' -d @- <<JSON
{"model": "Nemotron-3.5-Bolt", "max_tokens": 300,
 "messages": [{"role": "user", "content": [
   {"type": "text", "text": "What is in this picture?"},
   {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,$IMG"}}]}]}
JSON
```

A video is `{"type": "video_url", "video_url": {"url": "data:video/mp4;base64,…"}}`. Several pictures and videos fit in
one message and in earlier turns of a conversation.

- **Images**: PNG and JPEG (baseline, progressive, grayscale; EXIF orientation honoured, alpha composited over white),
  up to 10 MB, 8,192 px a side and 16 megapixels. CMYK JPEG, WebP, GIF and BMP are refused by name. The picture keeps its
  aspect ratio and is resized (antialiased bicubic, as the checkpoint's processor does) to 1,024 to 13,312 patches of
  16 px, so it costs 256 to 3,328 prompt tokens (a 512 px square about 260; 2560×1920, 3,300).
- **Video**: MP4 or WebM (H.264, HEVC, VP8, VP9, MPEG-4), up to 16 MB a clip, 20 MB and four clips a request, an hour long.
  Frames are sampled at 2 a second (1 a second above 720 rows), at most 256 (128 above 720 rows), taken in pairs, each
  pair about 256 tokens behind its "Frame i sampled at t seconds and frame j sampled at u seconds:" label as the checkpoint's
  processor writes it. The model card advises clips up to 2 minutes: a 6 s clip is 1,700 tokens, a 2-minute 720p clip
  36,500 (37 s to answer). Efficient Video Sampling (token pruning) is not implemented: every frame token is kept.
- A request holds up to 64 images and 4 videos in one chat. Tested: 50 images and 4 videos, 30,500 tokens, 29 s; 50 images of
  2560×1920 (165,600 tokens), 313 s, every one read correctly. Each prompt piece encodes only the images it crosses, so
  memory does not grow with the count. The whole conversation must fit `CONTEXT`.
- The prompt cache sees the pixels: the same picture again hits the cache, a different one never gets a stale hit. A
  follow-up turn that repeats 50 images and 4 videos (30,500 tokens) answers in 1.7 s.
- Data URLs, or HTTPS URLs with `VISION_URLS=1`. A picture the server cannot read is a 400 with the reason.
- The chat route, the Anthropic route (`image` blocks) and the Responses route (`input_image`) take images; text-only
  requests are unchanged.
- Memory: the tower and projector are 1.7 GB, plus the tower's scratch and a 134 MB rows buffer.

### Structured output and tool calls

`response_format` (`json_schema` or `json_object`), the `guided_*` fields and `tool_choice` `"required"` or a named
function work as in the OpenAI API; the reply follows the grammar even when sampled, and streams as usual:

```bash
curl -s localhost:8888/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Nemotron-3.5-Bolt", "max_tokens": 200, "chat_template_kwargs": {"enable_thinking": false},
  "messages": [{"role": "user", "content": "Give a person named Ann aged 30 with two tags."}],
  "response_format": {"type": "json_schema", "json_schema": {"name": "person", "strict": true, "schema": {
    "type": "object", "required": ["name", "age", "tags"], "additionalProperties": false,
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"},
                   "tags": {"type": "array", "items": {"type": "string"}}}}}}}'
```

`"tool_choice": "required"` makes the model call one of the `tools`; `{"type": "function", "function": {"name": "get_time"}}`
forces that one. The server compiles the grammar with `libtfgrammar` (xgrammar), which the image carries and loads at the
first structured request; there is nothing to configure. A grammar-constrained reply decodes about as fast as a plain one
(131 tok/s against 90 for the same JSON task unconstrained, since the grammar makes the drafts accurate).

### Thinking and sampling

Thinking is on. Per request, `temperature`, `top_p` and `seed` override the defaults (`temperature: 0` is greedy).
`"chat_template_kwargs": {"enable_thinking": false}` answers without thinking. The model card's medium effort is
`{"enable_thinking": true, "medium_effort": true}`. Reasoning and the answer share `max_tokens`.

Not served by this backend: logprobs, speculative trees, int8/int4 KV caches (`fp8` and `bf16` only), WebP, GIF and
BMP pictures.

## What the patches and the engine are

- `patches/0001-contract-controls-include.patch`: upstream's `91eab7d0`, "cuda contract: the softcap control includes
  the act_round helpers it calls". Backported for v1.1.4; drop it when `TF_REF` includes `91eab7d0`.
- `patches/0002-register-bolt.patch`: the family `nemotron_h_puzzle` (an omni config around a puzzle text model reads as
  that type in the server and the `models` CLI), its registration in the CUDA host beside upstream's own Bolt family,
  `zig build bolt` for libbolt, and `zig build
  compile-bolt-family`, which compiles the backend without a GPU. Video input (TensorFold #940) and the omni vocabulary
  size for structured output (#951), patches of this recipe before, are upstream's since 1.1.0. In v1.1.4, upstream
  also supplies the image-token cap of 262144 and passes `--parallel` to families that size their state by it.
- `patches/0003-w4a8-test-verdict.patch`: upstream's `f70f50bad`, "test(cuda): pass the current W4A8 exceptional-input
  verdict". It updates the CUDA test's prompt calls and checks the finite-input verdict without dropping its comparisons.
  Drop it when `TF_REF` includes `f70f50bad`.
- `engine/bolt_native.zig` and `engine/bolt_api.zig` (its libbolt binding; both copied into the family at build time) are
  the lane backend; `engine/*.cu` builds
  `libbolt.so` (`zig build bolt -Dbolt-src=engine`, linked with `-lcuda` for the virtual-memory calls). [`docs/engine.md`](docs/engine.md) has the
  interface and the memory layout.

## Checks

| Script | What it does |
| --- | --- |
| `scripts/ckptcheck.py` | a snapshot, or a `config.json`, against `contract/bolt.json` |
| `scripts/test_ckptcheck.py` | the checker accepts a config built from the contract and rejects a changed expert width |

```bash
python3 scripts/test_ckptcheck.py
python3 scripts/ckptcheck.py /path/to/snapshot
```

## Repository layout

```
start.sh      prepare, check, launch
stop.sh       stop a launch
scripts/      prepare.sh, config.sh, publish-image.sh, banner.sh, ckptcheck.py (+ its test)
patches/      upstream CUDA build fixes and Bolt registration, applied onto TensorFold at TF_REF
engine/       bolt_native.zig and bolt_api.zig (lane backend and its libbolt binding), the CUDA sources of libbolt.so
contract/     bolt.json, the pinned checkpoint's schedule and quant groups
docs/         engine.md, the backend and the engine; speed-research.md, the stage profiles and the plan
.github/      issue and pull request templates, GitHub Sponsors
CREDITS.md    who and what this builds on
CHANGELOG.md  upgrade notes
NOTICE        third-party notices
```

## License

MIT, see [`LICENSE`](LICENSE). The model weights, downloaded from Hugging Face and not part of this repository,
are under the [OpenMDW License Agreement, version 1.1](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB).

**Third-party software in the image.** The image `scripts/prepare.sh` builds is based on NVIDIA's PyTorch container
`nvcr.io/nvidia/pytorch:26.07-py3`, redistributed as a value-added runtime image. The NVIDIA software in it is
governed by the [NVIDIA Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/)
and the [Product-Specific Terms for NVIDIA AI Products](https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/),
which the container prints at every start. By pulling or running the image you accept them. The MIT license above
covers this repository's scripts only. The image also holds FFmpeg 9.0.2 (decoders only, built from source, LGPL) inside
`libtfvideo`, libjpeg-turbo 3.1.4.1 (BSD, IJG, zlib) and the IQ2_XXS tables from ggml (MIT); see
[`NOTICE`](NOTICE).

## Credits

Built on [TensorFold](https://github.com/ashhart/TensorFold) by Ash Hart ([ashhart](https://github.com/ashhart)) and
the [TensorFold contributors](https://github.com/ashhart/TensorFold/graphs/contributors), and on NVIDIA's
[Nemotron 3.5 Bolt 74B-A9B 24GB](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB) and its
[LiLiCorr drafter](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr).
The script layout follows MiaAI-Lab's
[Qwen3.8 Flash Next TensorFold recipe](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold).
The IQ2_XXS and Q8_0 formats and tables are [ggml](https://github.com/ggml-org/ggml)'s (MIT), pictures are decoded with
[libjpeg-turbo](https://libjpeg-turbo.org), video with [FFmpeg](https://ffmpeg.org), and the image is
built with [Zig](https://ziglang.org) on NVIDIA's PyTorch container.
The full list is in [`CREDITS.md`](CREDITS.md).
