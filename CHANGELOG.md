# Changelog

Newest first. This recipe targets one DGX Spark. The image is `tensorfold-nemotron-bolt:zig-eab8582`, built by
`scripts/prepare.sh` from TensorFold `eab8582` (v1.1.4) plus `patches/`.

## Unreleased

The Bolt backend serves the checkpoint. Measured on one GB10 (this build, 2026-10-20): decode alone 74 tok/s greedy prose
and 137 code with LiLiCorr (71 and 133 sampled; 4.3 to 36 without drafts), prefill 2,310 to 2,510 tok/s from 2.7k to
42k tokens, 2,255 at 63k and 1,975 at 126k (NV4_FP4=0: about 7 % less); sixteen streams at once 284 tok/s aggregate
greedy prose, 385 code, 251 and 280 sampled.

### TensorFold 1.1.4

- **Rebased onto TensorFold v1.1.4** (`eab8582`). `TF_REF` now pins the release commit.
- Dropped `0001-vision-image-tokens.patch`: upstream now accepts up to 262144 image tokens.
- Backported upstream's `91eab7d0`, "cuda contract: the softcap control includes the act_round helpers it calls", as
  `0001-contract-controls-include.patch`. Drop it when `TF_REF` includes `91eab7d0`; no test is skipped.
- Backported upstream's `f70f50bad`, "test(cuda): pass the current W4A8 exceptional-input verdict", as
  `0003-w4a8-test-verdict.patch`. The CUDA test now passes the device flag and checks the host verdict with the current
  prompt API, retaining its comparisons. Drop it when `TF_REF` includes `f70f50bad`.
- Rebased `0002-register-bolt.patch`: the build hook uses upstream's CUDA options, and the registry retains its new
  families beside libbolt. The `cuda_options.zig` hunk is gone because upstream already passes `lanes` to families.
  The libbolt build target, backend compile check, family export, hub entry and puzzle-model routing are retained.
- The Dockerfile reads xgrammar's pin from `zig/src/platform/grammar_library.zig`; upstream now installs libtfgrammar
  directly into `native/lib`, so the copy from the old `lib` location is gone.
- Measured against 1.1.0 (2026-10-25, one DGX Spark GB10, `THINKING=0`, port 8899): four fresh boots in the order
  A1 (1.1.0, `zig-76aafcf`), B1 (1.1.4, `zig-eab8582`), B2 (1.1.4), A2 (1.1.0), each `PREPARE=0 ./start.sh restart`
  with its own smoke test, then nine exact requests, cold prefill, and TensorFold 1.1.0's `bench_concurrent.py
  --levels 1,8 --reps 3 --tokens 256 --temperatures 0,1.0`. All four boots and benches exited 0, and
  `scripts/test_ckptcheck.py` passed on both trees. The nine exact replies (prose, code, math, schema, two forced tool
  calls, vision, thinking on, sampled at seed 7) are byte-identical across the four boots, and every decode run's
  token_sha256 matches. Decode is within noise: -1.7 % to +3.0 % aggregate. The two boots of one build differ by up to
  0.8 % (1.1.0) and 4.5 % (1.1.4, chat sampled at 8 streams).

  | prompt | temp | streams | A1 | B1 | B2 | A2 | 1.1.4 vs 1.1.0 |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | code | 0.0 | 1 | 138.9 | 138.4 | 139.7 | 139.3 | -0.0 % |
  | code | 0.0 | 8 | 354.0 | 356.8 | 357.5 | 356.9 | +0.5 % |
  | chat | 0.0 | 1 | 75.6 | 74.1 | 74.2 | 75.2 | -1.7 % |
  | chat | 0.0 | 8 | 255.2 | 254.7 | 258.9 | 256.0 | +0.5 % |
  | code | 1.0 | 1 | 135.5 | 134.5 | 135.9 | 135.5 | -0.2 % |
  | code | 1.0 | 8 | 236.6 | 245.9 | 243.3 | 238.3 | +3.0 % |
  | chat | 1.0 | 1 | 71.3 | 70.3 | 71.0 | 71.7 | -1.2 % |
  | chat | 1.0 | 8 | 211.7 | 204.6 | 213.8 | 210.8 | -1.0 % |

  Aggregate tok/s, median of 3 runs per boot. The last column compares the mean of B1 and B2 with the mean of A1 and A2.

  Cold prefill, `max_tokens 1`, client wall time: 3,845 tokens at 2,539 / 2,539 / 2,548 / 2,530 tok/s (A1 / B1 / B2 / A2,
  +0.4 %), and 3,900 tokens at 2,612 / 2,580 / 2,595 / 2,604 (-0.8 %). The 16,079- and 16,125-token prompts ran only on
  A1 (2,644 and 2,639 tok/s). On B1, B2 and A2 they returned HTTP 500: `bolt: no memory to grow sequence 0 past 4096
  tokens`. Those three boots reported 2.2, 3.6 and 6.5 GiB free after load, against 26.6 GiB for A1. A2 is the same
  1.1.0 image as A1 and fails the same way, so I attribute this to the machine's memory state at boot, not to 1.1.4.
  Other jobs ran on that machine during the comparison. 1.1.4 prefill beyond 4k tokens is therefore not measured here.
  The earlier results below remain from their recorded runs.

### Prompt attention 2.2 to 2.5× at long context, and the fp4 kernel at 154 TFLOPS

The prompt attention kernel ran at 29 TFLOPS whatever the context (instruction-bound: register staging, a 2-byte
V transpose, 4-byte fragment loads, two barriers a 32-key tile), and at 40k context its eight layers were about
40 % of a chunk. `attn_mma2_part_k` (`BOLT_ATTN_V2`, default on with the bf16 cache) keeps the same per-warp
math and MMA order, so its outputs are bit-identical, and stages K and V by `cp.async` into a two-stage ring of
64 keys with `ldmatrix` / `ldmatrix.trans` fragments and the query block as the fastest grid dimension: 51
TFLOPS at 8k context, 62 at 40k, 64 at 128k on `engine/tools/attn_bench.cu`. Prefill 2,360 → 2,485 tok/s at 21k
tokens (+5 %), 2,115 → 2,365 at 42k (+12 %), 1,923 → 2,254 at 63k (+17 %), 1,499 → 1,973 at 126k (+32 %); first
tokens, the 22.5k-token continuation and decode replies unchanged. `BOLT_ATTN_V2=0` keeps the first kernel.

The fp4 projection kernel went from 133/98 to 154/118 TFLOPS (`in_proj`/`out_proj`): token rows at a 48-byte
stride, B fragments through `ldmatrix`, the row scales stored step-major (`[k/64][T][4]`, padded to 4 rows so
a step's 128 words load as 16-byte `cp.async`), two steps of 64 k a barrier in a two-stage ring. Prefill +2 %
(2,469 at 5.3k, 2,360 at 21k, 2,115 at 42k before the attention change). Bit-identical to the shipped fp4 kernel
on 118 million outputs at seven row counts (`engine/tools/fp4_cmp.cu`) and the same perplexity (3.0657). Losers:
a 256-row CTA of 16 warps (142 TFLOPS), one CTA an SM (126). Also tried: nothing else in the attention kernel
pays without changing the rounding (a 64-key softmax step, lazy rescaling); those are left for a perplexity-gated
round.

### TensorFold 1.1.0

- **Rebased onto TensorFold v1.1.0** (`76aafcf2`, 2026-10-18). Upstream took in what two of the recipe's patches carried:
  video through the RADIO tower with the reserved video token shielded in user text (#940, this recipe's `patches/0003`,
  gone) and the omni checkpoint's `vocab_size` for structured output (#951, the `server/vocab.zig` part of `patches/0002`,
  gone). `patches/0002` is now the registration alone: the `nemotron_h_puzzle` family in the CUDA host's registry, the
  server's and the `models` CLI's reading of an omni config around a puzzle model as that type, `--parallel` handed to a
  family whose `Options` take `lanes`, `zig build bolt`, and `zig build compile-bolt-family` (the backend compiled against
  the lane core without a GPU, as upstream's `test-bolt-host` does for its own Bolt).
- **Upstream 1.1.0 serves Bolt itself** (#897, #938: the Nemotron family reads the 24 GB checkpoint's puzzle layers and
  GGML experts, shared rounds, images, structured output) but has no LiLiCorr drafter: the checkpoint has no MTP head, so
  that path decodes plain. This recipe keeps routing the checkpoint to libbolt and the drafter; the omni model type of
  the config is remapped before the registry is consulted, so upstream's family never opens it.
- The lane core's changes reach the backend unchanged: its vtable and `Model` facts gained optional fields only
  (prompt pieces between rounds, `draft_step_ms`, `join_tail`), all at their defaults here. Streamed replies leave with
  `TCP_NODELAY` (#929). The Zig version (0.17.0), the xgrammar pin (v0.2.8) and the FFmpeg build script are unchanged.
- Checked without a GPU, in a CUDA 13.1 container: `zig build compile-bolt-family`, `compile-checkpoint-consumers` and
  `zig test zig/src/families/nemotron/config.zig` pass; upstream's `test-bolt-host` passes 162 of 163 tests, the one
  failure (leaks in a prompt-cache store test) fails identically on an unmodified v1.1.0 checkout; the `models` CLI's
  model-type test passes; libbolt compiles for sm_121.
- On the Spark: `scripts/prepare.sh` built `tensorfold-nemotron-bolt:zig-76aafcf` (the Zig build of the server, libbolt,
  libtfvideo and libtfgrammar took 197 s), `./start.sh` served it on a second port beside the running 1.0.12 server
  (loaded in 33 s, smoke test OK), and the two were benchmarked back to back; see the next section.

### The prompt pass's NVFP4 projections on the fp4 tensor cores, and a perplexity harness

`in_proj` and `out_proj` of prompt chunks run on GB10's block-scaled fp4 MMA (`kind::mxf4nvf4`, m16n8k64, e4m3
scales a block of 16) with the rows quantized to NVFP4 first, as vLLM's NVFP4 path does; the weights' nibbles
and scales are used as stored, reordered into the MMA's fragment order at load (`tile_nv4w_k`). libbolt now
builds for `sm_121a`. Exact against a CPU reference on the same quantized inputs; 133 TFLOPS on `in_proj`
(2.15× the fp16 path), 98 on `out_proj`. Prefill +17 to 20 % (2,425 tok/s at 5.3k tokens, 2,314 at 21k, 2,078
at 42k). This is the first default that changes the model's numbers: on `engine/tools/ppl.cu` (libbolt's C ABI,
44.6k tokens of prose and code) perplexity 3.0657 against 3.0418, +0.8 %; the first token of a prompt was
unchanged at every size tried, later greedy tokens can differ. `NV4_FP4=0` (`BOLT_NV4_FP4=0`) keeps the fp16
rows. The harness also shows the int8 shared-expert paths at +0.02 % and `BOLT_PROMPT_MIN_ROWS` /
`BOLT_MOE_TILE_PAIRS` let it exercise the prompt-chunk kernels on 128-row blocks.

### A tensor-core prompt scan, parked

`scan_ssd_k` (`BOLT_SCAN_SSD=1`): the Mamba-2 chunked scan on tf32 MMAs, numerically right (rms relative
difference 4e-4) but 2.5× slower as written (one CTA an SM, latency-serialized chunks). Off; `scan_bench.cu`
compares the two.

### Prompt chunks of 2048 rows

`TMAX` is 2048 (the lane core's prompt pieces already were), so each expert sees twice the tokens a chunk and the
routed-expert kernel's items are fuller: prefill +3 to 5 % (2,025 tok/s at 5.3k tokens, 1,943 at 21k, 1,768 at
42k), outputs unchanged. The per-chunk buffers grow with it (about 1 GB).

### Three decode experiments behind switches, all off

`BOLT_DENSE_RING=1` (a per-warp `cp.async` ring for the NVFP4 and Q8_0 decode GEMMs: level, 0.92 to 1.02×, the
kernels are at the DRAM ceiling already; identical outputs), `BOLT_IQ2_WHOLE=1` (whole-entry gathers for the bf16
IQ2 path, 2× on the kernel) and `BOLT_SHARED_DECODE=1` (the shared expert of verify blocks on the int8
deep-prefetch kernel): the last two are faster kernels that measure within noise end to end because the shared
expert runs on a side stream under the routed experts, and both change greedy replies in 7 to 8 of 10 prompts
(fp32 reordering, int8 rows), so they stay off. `gemm_bench.cu` cycles enough weight copies to defeat the L2.

### Acceptance by category, and the lattice ablation

`~/bench/accept.py`-style measurement (400 greedy tokens a prompt): accepted tokens a round are 2.4 to 2.6 on
writing and roleplay, 2.9 to 3.1 on chat, 4.9 on code, 5.8 on math, 6.4 on summarizing a document, 7.1 to 7.3 on
JSON edits and rewrites (the window holds 8). The drafter port was compared with vLLM's `lilicorr.py` and matches
it; the lattice's contribution was measured by forcing rank 0 at every slot (`BOLT_DRAFT_TOP1=1`, plain DFlash):
3.47 against 3.72 accepted a round overall, 12 % less on math and 21 % less on code, NVIDIA's reported range. So
prose acceptance is what this drafter gives; `docs/speed-research.md` has the table. Every forward kernel now also
triggers its dependents at the end of its work (`griddepcontrol.launch_dependents`): no measurable change.

### Programmatic dependent launch

Every kernel of the forward (`bolt_kernels.cu`, `bolt_mma.cu`) is launched with the programmatic-stream-serialization
attribute and waits for its predecessor at entry (`griddepcontrol.wait`), so a launch and its scheduling overlap
the tail of the kernel before it. Measured: one stream greedy code 131 → 137 tok/s, sampled 130 → 133, prose
+1 to 2 %; eight and sixteen streams level; the eight-row window 34.2 → 33.4 ms. Replies identical. `BOLT_PDL=0`
launches plainly. No kernel triggers its dependents early yet (the implicit trigger at completion is used), which
is where the rest of this item's gain would come from.

### Decode: the routed experts with two kilobytes a warp in flight

`moe_coop_k` streams each warp's two weight rows through a per-warp three-stage `cp.async` ring (64 bytes a row a
step, two steps ahead) instead of a 32-byte load a row one step ahead; items, tiles and arithmetic are
`moe_mma_k`'s, so replies are identical (checked, 12 cells). The kernel was latency-bound on bytes in flight, not
on bandwidth: on the microbenchmark it goes from 137 to 226 GB/s at eight rows and from 125 to 196 at 48. End to
end: one stream 118 → 135 tok/s greedy code and 66 → 74 prose, eight streams +17 to 20 %, sixteen streams +17 %
(greedy code 327 → 383, prose 240 → 281). `BOLT_MOE_COOP=0` keeps the old kernel.

### Prompt pass: the shared expert on the same int8 kernel

The shared expert of a prompt chunk runs on `moe_tile_k` as one expert over all the chunk's rows (a second copy
of its two tensors in the slab layout, 450 MB): prefill +4 % more (1,816 → 1,890 tok/s at 21k tokens). Its rows
are int8 per 32 for prompt chunks now, as the routed experts' always were, instead of bf16; the first token's
hash is unchanged at every prompt size tried and a 200-token greedy continuation after a 22.5k-token document is
identical, but this is a numerics change in principle. `BOLT_SHARED_TILE=0` keeps the bf16 path.

### Prompt pass: the routed experts on int8 tensor cores

`moe_tile_k` serves the routed experts of prompt chunks (over 4096 pairs): a CTA of 8 warps owns one expert's
128 weight rows and 64 of its tokens (work items of 64 pairs), decodes the IQ2 tile to int8 in shared memory
with its chunk scales, streams the weight pieces and the int8 token rows through `cp.async` rings, feeds the
int8 tensor cores through `ldmatrix`, skips the MMAs of empty token tiles, and accumulates with the same
`(dA·s)·dd` chunk expression in the same order as `moe_mma_k`, so the outputs are identical (checked bit for bit
on the microbenchmark and by the first token's hash at 2.7k, 5.3k and 21k tokens). Measured: prefill +7 % at
every prompt size (1,697 → 1,814 tok/s at 21k), the routed-expert stage of a 1024-row chunk −30 %.
`BOLT_MOE_TILE=0` keeps the old kernel. The microbenchmarks live in `engine/tools/` (`moe_bench.cu`,
`gemm_bench.cu`, `mma_peak.cu`; the last one shows fp16 `mma.sync` peaks at 107 TFLOPS on GB10 and int8 at
214, which is why the dense prompt GEMMs, at about 60 % of 107, were left alone and the experts moved to int8).

### Prompt pass: packed rows, and two prompt-GEMM experiments that lost

The prompt chunks' dense GEMMs (`mma_big_k`) now read the rows packed once a launch too (the per-stream scratch
grew to 1024 rows of 8192, 48 MB): prefill +4 % (1,632 → 1,702 tok/s at 2.7k tokens, 1,637 → 1,696 at 21k), the
first token unchanged. Tried and taken out again, with the numbers in `docs/speed-research.md`: a 128-token-group
prompt kernel with the weight fragments decoded once a step (12 % slower: one row tile a warp doubles the shared
memory reads per MMA), and 32-pair expert items for prompt chunks (17 % slower). Prefill's serialized profile is
there as well; the routed experts take 31 % of a 1024-row chunk at 16 TFLOPS and the dense GEMMs 41 % at 41 to
57, and the write-up says what kernel shape it will take to move them.

### Routed experts: whole-entry gathers, and two experiments that lost

`moe_mma_k` now lets each lane decode whole IQ2_XXS grid entries (K permuted inside the int8 MMA, the activations
read at the same real k; a 32-term integer dot product is the same in any order, so replies are unchanged): four
8-byte table reads a chunk instead of eight 4-byte ones. Measured 0 to 4 % (one stream and prefill), level at
sixteen streams; `BOLT_MOE_HALF=1` keeps the old form. Two more variants are in the code behind switches because
their results are evidence: `BOLT_MOE_BF=1`, the routed experts on bf16 tensor cores with the chunk scale folded
into the weight fragments and no per-chunk epilogue, is *slower* (one-row window 29.3 against 26.1 ms, prefill
1,335 against 1,655 tok/s); `BOLT_RT1=1`, one row tile a CTA for the small dense GEMMs, is 5 % slower at sixteen
streams. `docs/speed-research.md` has the serialized stage profiles (`BOLT_PROF=2`) for one stream, sixteen and
prefill: the IQ2 expert GEMMs are half of every forward at about 100 GB/s, and the remaining hypothesis is the
DRAM access pattern (32-byte pieces of sixteen rows a step).

### Many-row GEMMs: packed activations

The verify blocks' dense GEMMs (`mma_body`: four warps split K, a CTA holds two 16-row weight tiles) had every
warp re-read its tokens' fp32 activations from L2 and convert them at every K step; at 64 rows that is about
580 MB of L2 traffic a layer for `in_proj` against 43 MB of weights, and the stage took 463 µs a layer at 48
rows against 199 at 7. The rows are now packed to halves once a launch (`pack_x_k`, the same rounding, a scratch
per stream) and the kernels read 2 bytes an element; the sums are the same in the same order, so replies are
unchanged (checked, 12 cells). Measured: `in_proj` 463 → 329 µs at 48 rows, the 128-row round 205 → 191 ms,
aggregate decode +3 to 6 % at 8 and 16 streams (sixteen streams greedy code 309 → 324 tok/s, prose 222 → 234),
one stream unchanged. `BOLT_PACK=0` keeps the fp32 path. Also: every `BOLT_*` variable in start.sh's environment
now reaches the container; `BOLT_SPIN` (spin instead of yield on the host wait) and `BOLT_MOE_BD` (MoE work item
size) exist and measured nothing; `BOLT_PROF` prints its stage report every `BOLT_PROF_EVERY` (100) blocks with
the rows a block and the MoE's distinct experts and items a layer. The findings and the plan are in
[`docs/speed-research.md`](docs/speed-research.md).

### Speed work, measured

Measured on 2026-10-19 on one DGX Spark with TensorFold's `tools/bench_concurrent.py` (N clients send the same prompt at
once, 256 tokens each, medians of three runs; thinking off; nothing else on the GPU). Aggregate tok/s by cell, the
1.0.12 build against this build with two Mamba state copies and with one (`SSM_LAZY=1`, now the default):

| prompt | sampling | clients | 1.0.12 | 1.1.0, two copies | 1.1.0, one copy |
| --- | --- | ---: | ---: | ---: | ---: |
| prose | greedy | 1 | 62.5 | 66.4 | 69.0 |
| prose | greedy | 4 | 136.3 | 138.8 | 145.3 |
| prose | greedy | 8 | 173.9 | 176.6 | 191.1 |
| prose | greedy | 16 | 230.7 | 212.4 | 221.9 |
| code | greedy | 1 | 118.1 | 112.3 | 118.8 |
| code | greedy | 4 | 192.3 | 227.4 | 237.4 |
| code | greedy | 8 | 274.8 | 266.6 | 279.8 |
| code | greedy | 16 | 315.3 | 299.5 | 306.4 |
| prose | sampled | 1 | 58.9 | 62.9 | 63.9 |
| prose | sampled | 4 | 120.0 | 116.2 | 121.7 |
| prose | sampled | 8 | 141.3 | 152.3 | 163.3 |
| prose | sampled | 16 | 186.0 | 185.9 | 199.2 |
| code | sampled | 1 | 113.1 | 107.1 | 110.1 |
| code | sampled | 4 | 133.5 | 143.8 | 147.1 |
| code | sampled | 8 | 176.8 | 174.0 | 178.6 |
| code | sampled | 16 | 208.2 | 211.4 | 218.4 |

Single-stream cells move by up to 5 % between runs of the same build, so the one-stream columns are equal within noise.
One copy against two is a 3 to 8 % gain at 4 to 16 streams in 11 of 12 cells. The 1.1.0 build with one copy is level
with 1.0.12 alone and at 16 streams and ahead at 4 and 8 (code greedy ×4: 192 against 237). `DRAFT_Q8=0` against `1`
(both with one copy) is the same within noise in every cell. Every cell's replies, greedy and seeded, were token for
token the same across all four builds and settings (`token_sha256`), so the one-copy state is exact.

- **One Mamba state copy a stream, `SSM_LAZY=1` (`BOLT_SSM_LAZY`), the default.** The two-copy design pays a replay
  pass on every partial accept: `keep(n)` reads the pre-window copy and writes the post-`n` state, 4 MiB each way for
  each of the 40 Mamba layers, then waits for it, stream by stream. With one copy a verify window writes no state; its
  rows' scan inputs are saved per stream (two slots of eight rows), `keep(n)` only records `n`, and the stream's next
  block folds those rows in at the start of its scan, writes the state once, and runs its own rows from it. Per stream
  and round that is one read and one write of the state instead of up to two of each, no replay launch and no wait: at
  sixteen streams about 270 MB less traffic a stream a round, roughly 19 ms of a 220 ms round by the engine's own
  bandwidth figures, 1 to 2 ms of a lone stream's 55. A stream holds 0.28 GiB from load instead of 0.41. Prompt chunks
  commit at their end as before; a snapshot taken with rows pending folds them first (`bolt_snap_save_seq`). The
  arithmetic order of the state is unchanged (the same rows in the same order), and the replies are bit-identical to
  the two-copy path (above). With 16 streams libbolt reports 22.9 GiB free after load against 22.0.
- **Rollbacks in one call** (libbolt C ABI 3, `bolt_keep_multi`): the backend hands every stream's cut to libbolt at
  once, and with two copies their replays go out together with one wait instead of one each.
- **The drafter's projections as Q8_0, `DRAFT_Q8=1` (`BOLT_DRAFT_Q8`), on by default.** LiLiCorr's attention and
  kernel projections (five layers, 43 M bf16 parameters each) are quantized to Q8_0 at load on the host, ggml's blocks,
  and read through the target's Q8_0 tensor-core GEMM: about 200 MB less to read a proposal, of 1.4 GB. The target
  checks every draft, so this moves the acceptance rate at most, never the reply; `DRAFT_Q8=0` keeps bf16. Measured:
  the same replies, a proposal of 6.4 ms against 7.2, no difference in tokens a second beyond noise.
- **The planner's costs.** The round loop prices a shared round as a table by total rows plus an overhead it learns per
  stream count. The table used to be measured with the rows spread over as many streams as possible (16 rows: 16
  streams of one row), so it carried sixteen streams' state traffic and over-priced the rows of a round of 2 to 15
  streams, which under-drafts them. It is now one stream's window costs followed by the cost of a row, measured as
  sixteen streams at 1 row each against 8 each; the state of each further stream lands in the learned overhead.
  `BOLT_CAL=spread` (start.sh passes it through) restores the old table. Measured at 8 and 16 streams (one copy, Q8
  drafter): the new table decodes prose at 8 streams 10 to 15 % faster (greedy 190 tok/s against 165, sampled 159 to
  163 against 149) and is level everywhere else (code ×16 greedy 313 against 315). A repeat of the default run put
  the run-to-run spread at 8 and 16 streams within 3 %. And a draft level no longer costs `propose / 3` ms: LiLiCorr proposes all
  seven in one pass whatever the depth, so the pass is a flat per-round cost the loop measures (`mtp_step_ms` 0).

### Structured output and forced tool calls

`response_format` (`json_schema`, `json_object`), the `guided_*` fields and `tool_choice` `"required"` or a named function
were refused with HTTP 400 ("not supported by this engine yet"). Four causes, four fixes:

- **No masked draws in the backend**, so the engine advertised no structured output. `verify` now redraws the rows of a
  masked window among the tokens their bits allow, `first_masked` redraws the first token under the first mask, and
  `held()` hands the grammar the drafts the drafter holds. libbolt gained `bolt_mask_rows` (C ABI version 2): a masked
  variant of the two-stage GPU top-k that recomputes only the masked rows. Its kernel test against a CPU reference covers
  masks of 5 tokens, about half the vocabulary and 1 token.
- **The server could not read an omni checkpoint's vocabulary size** (`vocab_size` is in `llm_config`, not `text_config`
  or the top level): `server/vocab.zig`, in patch 0002, with a test.
- **`libtfgrammar` (xgrammar v0.2.8) was not in the image.** The Dockerfile builds it with upstream's `zig build grammar`
  and installs it beside the server.
- The libbolt binding moved to `engine/bolt_api.zig` so `bolt_native.zig` stays under the 600-line limit upstream asks
  for; the Dockerfile copies both files.

Measured on a dev server: `json_schema`, `json_object`, `tool_choice: required` and a forced named tool all returned valid
output; 6 of 6 replies sampled at temperature 1 were valid; a streamed structured reply was valid JSON; with 4 structured
and 4 plain requests running together, all 4 structured replies were valid and all 4 plain replies equalled their solo
runs. A structured reply decoded at 130.9 tok/s alone against 90.2 for the same task unconstrained.

### Review fixes for video

- User text that spells the video token (`<SPECIAL_30>`) or `<video>` stays text: it no longer turns a text-only request
  into a refused one, or counts a second placeholder beside a real video (patch by cwschroeder, in `patches/0003`;
  upstream's since 1.1.0).
- The rows buffer of an Omni checkpoint holds both a request's image and video budgets, so a long video, or a video
  beside an image, no longer fails with `ImageRows`; Bolt's piecewise path was not affected.

### Concurrent replies equal solo ones

A stream's replies used to depend on the batch it ran in: `bench_concurrent.py --alone` found every prose ("chat") reply
at four streams different from its solo run (0 of 12 equal), and code replies equal. The cause was not the GEMM kernels
chosen by row count, as first written up; it was three batch-dependent summation orders, all fixed now:

- **Attention key chunks.** A verify window merged partial softmaxes over key chunks sized by the number of streams in the
  block (about 32 splits for one stream, 8 for four) and by the window's width. The chunk size is now `win_chunk(base)`
  (128 keys, doubled until 32 chunks reach the sequence's end and window), a function of the sequence's own length.
- **GEMM order.** Blocks of more than 32 rows used `mma_big_k`, which sums K in one sequence; smaller blocks used the
  K-split kernel's four interleaved partials. Decode and verify blocks now always use the K-split kernel
  (`mma_decode_rows`, 64 tokens a CTA); the prompt pass keeps `mma_big_k`.
- **The plain step.** The non-speculative one-token step used the CUDA-core attention kernel; it now runs as a one-row
  window (then kept), so drafted and plain decoding share one arithmetic.

Tests: perplexity of passages scored alone and in blocks of 2, 3, 4, 8 and 16 streams with 8-row windows (up to 128 rows
a block; also 1- and 5-row windows at four streams) is identical to five decimals; sixteen sequences match their solo
runs for 768 of 768 tokens, plain steps and 4-row windows alike; `tools/bench_concurrent.py --alone --serial` on the production image
(`--parallel 8`, greedy, 256 tokens, 3 repetitions):

| prompt | streams | aggregate | each stream | concurrent equals solo | drafted equals `"draft": false` |
| --- | ---: | ---: | ---: | ---: | ---: |
| code | 1 | 115.1 tok/s | 115.1 tok/s | 3 of 3 | 1 of 1 |
| code | 4 | 207.9 tok/s | 54.8 tok/s | 12 of 12 | 1 of 1 |
| code | 8 | 252.6 tok/s | 35.2 tok/s | 24 of 24 | 1 of 1 |
| chat | 1 | 60.7 tok/s | 60.7 tok/s | 3 of 3 | 1 of 1 |
| chat | 4 | 139.1 tok/s | 35.6 tok/s | 12 of 12 | 1 of 1 |
| chat | 8 | 174.7 tok/s | 23.6 tok/s | 24 of 24 | 1 of 1 |

Perplexity on the 1,499-token passage is unchanged (5.286 one token at a time). Cost: the K-split kernel decodes the
weights again for each 64-token group, so the widest shared rounds are slower, 16 streams of 8-row windows (128 rows)
220.3 ms against 176.9 ms (about 25 %); a round of 64 rows is 130.4 ms against 135.5 ms for 8 streams of 8-row windows
and 141.7 ms against 133.2 ms for 16 streams of 4-row windows. A 16-sequence round of one row each is unchanged
(90.3 ms).

### TensorFold 1.0.12, images and video

- **Rebased onto TensorFold 1.0.12.** Upstream now ships a RADIO tower and image input, so the recipe uses them: the
  tower, the image and video server code and the Anthropic and Responses media routes are upstream's, the engine's own
  ViT, stb_image decoder and ffmpeg subprocess are gone. Bolt is the family `nemotron_h_puzzle`, built with
  `zig build bolt`; libbolt takes image rows by device pointer (`bolt_prefill_seq_rows`) and checks an ABI version.
- **Video** through the same tower: tubelets of two frames through `video_embedder`, the processor's frame labels,
  `<video>` turned into one token before encoding, `--vision-max-videos` (patch 0003). Decoding is upstream's libtfvideo
  over FFmpeg 9.0.2, built in the image.
- **A request holds up to 64 images and 4 videos** (`VISION_MAX_IMAGES`, `VISION_MAX_VIDEOS`); each prompt piece encodes
  only the images it crosses. Tested: 50 images and 4 videos in one chat, and 50 images of 2560×1920 (165,600 tokens).
- Formats follow upstream's: PNG and JPEG (libjpeg-turbo 3.1.4.1, in the image), HTTPS URLs behind `VISION_URLS=1`;
  GIF, BMP and CMYK JPEG are no longer read. Text speed and outputs are unchanged (perplexity 5.286; greedy replies
  identical to the previous base).
- `--parallel` reaches libbolt through the backend (no `BOLT_SEQS` in `start.sh`).

### Sixteen streams, a faster prompt pass

- **Streams.** libbolt keeps `BOLT_SEQS` (= `--parallel`, default 16) independent sequences. A block is a list of
  segments; a shared decode round carries every stream's verify window in one forward, so the weights are read once
  for all of them. Recurrent state is two SSM copies and a conv ring per sequence; a window reads one copy and writes
  the other, and a partial accept replays from the untouched one. The drafter, its ingest and its proposals are
  batched across streams too (25 ms for sixteen streams against 9 for one). `PARALLEL=16` is the default.
- **Memory follows use.** The attention rows of every stream have the full 262,144-token window in the address
  space, but are mapped 4,096 tokens at a time through CUDA virtual memory and unmapped when the stream ends; 16
  streams cost 0.4 GiB each from load, not 2 GiB each. `libbolt.so` now links `-lcuda`.
- **Prompt pass 380 -> 1,450 tok/s.** Tensor-core flash attention (prompt and verify windows), routed experts on int8
  tensor cores (tokens staged with overlapping loads, several weight tiles a warp), a shared-memory-staged GEMM for
  blocks over 32 rows, a row-parallel conv and a thread-pair scan, 1,024 rows a chunk, and the block scales of the
  NVFP4, Q8_0 and IQ2 GEMMs folded into the decoded weights (NVFP4's are exact in fp16) so the mma runs through K
  with no per-block epilogue. Prefill used to fall to 190 tok/s at 50k tokens and now holds 1,290.
- **Backend.** `prefill_step` (pieces of 2048 rows, 512 while other streams decode), `LaneMemory` when every sequence is
  busy, snapshots per stream, `draft_streams` and shared-round costs measured at load.
- **Checks.** The unit tests cover attention including several windows in one launch; sixteen sequences decoded in one
  block match their solo runs, and a leak test compared a stream's logits across neighbours (it missed the batch-dependent
  summation orders fixed under "Concurrent replies equal solo ones"); the rollback check now compares
  against a window of exactly the kept rows.

- **fp8 KV cache.** `KV_DTYPE=fp8` stores K and V as e4m3 with a scale per token and head: 4,224 bytes a token instead
  of 8,192, in the cache, the lazily mapped pool and prompt-cache snapshots. No speed change (prefill 1,420 tok/s at
  16k, a 30k-context round 68 ms either way); perplexity 2.760 against 2.761 over 4,000 tokens, per-token log-prob
  drift 0.026 against a 0.018 noise floor, the 195k needle and tool calls pass. bf16 stays the default.

- **Pinned KV.** `KV_PIN=1` (`BOLT_KV_PIN`) maps every stream's whole window at load and never unmaps it, for a guaranteed
  `PARALLEL` x `CONTEXT`; a start that cannot fit it fails with a message instead of failing a request later.

### Tried and did not help

- A grouped dp4a expert kernel (pairs grouped by expert, CUDA cores): no faster than one pair a thread.
- Work items of 32 pairs for the tensor-core expert kernel (fewer, wider items for big chunks): 10 % slower than 16.
- Prompt chunks of 2048 and 4096 rows: 1,000 and 750 tok/s against 1,200 at 1,024 (512 and 768 are within 3 %).
- Activations quantized per 256 (llama.cpp's Q8_K) so the expert kernel could scale once a block: no faster, and
  perplexity moved 0.2 %.
- 4 CTAs per SM for the expert kernel (64 registers): the spills cost more than the occupancy bought.
- Eight-token groups in the old large-batch GEMM: slower; the real cost was re-reading fp32 activations through L2
  (94 % L2 throughput), which staging them in shared memory fixed.
- Splitting the scan over more warps per channel (2 or 4 channels a warp): 2 % at most; the thread-pair scan with
  shared-memory staging took conv+scan from 3.6 to 1.2 ms a layer.
- fp16 or bf16 SSM state would cut the sixteen-stream round by about 12 ms, but the recurrence accumulates the
  rounding and NVIDIA recommends fp32 for this model: not done.

### Earlier, in this release

- `engine/bolt_native.zig`: the lane backend (prefill, verify windows, rollback, held drafts, keyed draws) in place of
  the one-row stub. `--drafts` is the default; `DRAFTS=0` serves one row at a time.
- `libbolt.so` rewritten: packed weights read once by tensor-core GEMMs, one CUDA graph a block, grouped expert
  kernel, per-request reset, KV for the whole window (it stopped at 4,096 before).
- Fixes behind the 0/7 acceptance: the shared expert took the norm of the updated residual, Mamba's `dt` floor was
  0.001 (the model's is 0), and the drafter's attention convolution ran before `o_proj` instead of after.
- Prompt cache: libbolt copies and restores its state (`bolt_snap_*`), the backend resumes prompts from kept prefixes and
  cuts the prompt pass at the cache's marks. `PROMPT_CACHE_GIB=8` by default.
- `scripts/prepare.sh` downloads the LiLiCorr drafter; `start.sh` serves by default (`SERVE=0` stops after the check).

### Before that

The checkpoint contract and the load gate. No Spark boot, and no decode or prefill numbers.

- Pin `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB` at `a326822af111290f2af5ac59ef4b5040347dabc4`.
  `contract/bolt.json` records the 88-layer schedule, the three quant groups and the C-RADIO tower.
  `scripts/ckptcheck.py` checks a snapshot against it.
- `patches/0001-refuse-bolt-checkpoint.patch` on TensorFold's Nemotron `parse`: a Bolt omni or puzzle config
  returns `UnsupportedCheckpoint` and names the checkpoint. Lightning's `nemotron_h` path is unchanged.
- `./start.sh` prepares and checks. `SERVE=1` launches `tensorfold-native`. On this pin the process exits at load.
