# Where the decode time goes, and what could still move it

Research notes from 2026-10-19 on one DGX Spark (GB10). Numbers are this build's (TensorFold 1.1.0, one-copy Mamba
state, Q8_0 drafter) unless marked otherwise. Sources are linked where a claim is someone else's measurement.

## What the hardware gives

- GB10: 48 SMs, 24 MB L2, 128 KB L1/shared an SM (99 KB a block), `mma.sync` tensor cores (no wgmma, no TMEM, no
  usable thread-block clusters); INT8, FP8 and BF16 all run at about 215 T-op/s, only packed NVFP4 doubles that.
  Nominal memory bandwidth 273 GB/s; people measure 200 to 240 GB/s for streaming kernels
  ([forum triad numbers](https://forums.developer.nvidia.com/t/gb10-really-does-hit-1-pflop-nvfp4-2-4-sparse-measured-with-an-open-source-tool-to-reproduce-it/373618),
  [Modular's Mamba kernel at 240](https://github.com/modular/modular/pull/6633)).
- Programmatic dependent launch works (compute capability 12.1); llama.cpp measured +2 to 6 % token generation on a
  Spark from it ([#22522](https://github.com/ggml-org/llama.cpp/pull/22522)). The default host wait after a sync
  carries wake-up latency on this chip; llama.cpp got +19 % on a 120B MoE from `cudaDeviceScheduleSpin`
  ([#16585](https://github.com/ggml-org/llama.cpp/pull/16585)).

## What a forward reads (the bandwidth model)

Bytes one verify window reads, from the checkpoint's shapes:

| part | bytes a forward | note |
| --- | ---: | --- |
| Mamba `in_proj`, 40 layers, NVFP4 4096×18560 | 1.7 GB | the largest single item |
| Mamba `out_proj`, NVFP4 8192×4096 | 0.7 GB | |
| shared expert, IQ2_XXS 2×4096×5376 | 0.45 GB | |
| latent `fc1`/`fc2`, Q8_0 | 0.33 GB | |
| attention q/k/v/o, 8 layers, Q8_0 | 0.3 GB | |
| `lm_head`, Q8_0 131072×4096 | 0.54 GB | read for every window |
| routed experts: 9 a row a layer, 1.08 MB each (IQ2_XXS up+down) | 0.39 GB a row | distinct experts only |

So one row reads about 4.4 GB (19 to 21 ms at 210 to 240 GB/s) and an 8-row window about 7 GB (30 to 33 ms).
Measured: 26.5 ms for one row, 40 ms for eight. The single-stream forward runs at 75 to 80 % of the achievable
bandwidth; the rest is launch gaps (about 900 kernels a forward inside a CUDA graph) and kernels below peak. The
drafter's proposal reads about 1 GB in 6.4 ms (150 GB/s: small kernels for eight rows).

A round alone is proposal 6.4 ms + window 35 to 40 ms + host ≈ 45 ms; prose accepts 2.8 tokens a round (63 to
69 tok/s), code 5.5 (115 to 119 tok/s).

Sixteen streams: a 16-row round (one row each) costs 45 ms, a 128-row round 206 ms, 1.44 ms a row in between. The
rows of 16 streams touch most of the 512 experts a layer (1152 pairs over 512 experts: about 460 distinct, 20 GB of
expert weights a round), so the floor of a 128-row round is roughly 90 ms of experts plus 19 ms of dense weights:
about 110 ms. The measured 206 ms is about twice that. **This is the largest single gap in the engine**: the
many-row MoE path, not the single stream.

## The drafter

- NVIDIA's own LiLiCorr numbers for this checkpoint (Bolt technical report, table 5: vLLM, 7 drafts, 64 streams)
  are 4.68 accepted tokens a round greedy and 4.25 sampled over their evaluation mix; coding 5.69 / 5.18, writing
  3.70 / 3.36, roleplay 3.75 / 3.42. Our code prompt reaches 5.5, matching; our prose prompt 2.8 is below their
  writing category but it is a different prompt, so this is not evidence of a port bug. The port implements the
  whole method (DFlash backbone over five captured target layers, the candidate lattice with top-8 a slot, the
  features and the greedy walk), so the likely remaining differences are the target's IQ2_XXS quantization
  (NVIDIA measured with the same 24GB export, so probably none) and the lane core's depth policy. The round loop
  already shortens windows on prose (about 3 drafts a round proposed there, 7 on code), so a fixed depth sweep is
  moot.
- Tree verification with one Mamba state is possible (STree: packed tree scan with a tree mask), but reported
  gains on a hybrid 8B are 3 % greedy and 10 % sampled, and each branch past the first costs expert reads here. A
  two-way fork at slot 1 or 2 is the only shape likely to pay on 273 GB/s; expected value is small.
- N-gram or suffix drafting helps only when the reply copies the prompt (edits, JSON rewrites: up to 1.9×), and
  hurts elsewhere unless gated by misses; a gated hybrid (n-gram when a long prompt match exists, LiLiCorr
  otherwise) is the only form worth adding.
- No other NVIDIA draft head exists for Bolt (no MTP, no EAGLE-3); Lightning's heads are for Lightning.

## Options, ranked by expected gain for the work

1. **Many-row MoE GEMM at the bandwidth floor** (sixteen streams: 206 → ~120 ms a round, up to 1.7× aggregate).
   Today `moe_mma_k` stages up to 16 pairs a work item and walks weight tiles once an item, so an expert with many
   rows is read once per 16 of them and each tile's IQ2 dequant is repeated per item; llama.cpp's MMQ decodes a
   weight tile to int8 in shared memory once and reuses it for all columns of the expert
   ([mmq.cuh](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmq.cuh)). Plan: items of 32
   pairs first (`BOLT_MOE_BD=32`, one line, measured below), then a kernel that loops over all of an expert's
   tokens with the decoded tile resident and a two-stage `cp.async` prefetch of the next tile.
2. **Launch overlap for the single stream** (26 → ~23 ms a row, 5 to 10 %): programmatic dependent launch on every
   kernel of the forward (a `griddepcontrol.wait` at kernel entry and the launch attribute in the graph), plus
   spin waits on the host (`BOLT_SPIN`, measured below). Fusing the small kernels around each GEMM (norm into the
   GEMM prologue, `quant_rows` into the producer's epilogue, combine into `fc2`) removes about ten launches a
   layer; with PDL in place the gain is a few percent more.
3. **Proposal in a CUDA graph** (hoped 6.4 → ~5 ms a round, 3 %): the drafter's eager launches and sync become one
   graph launch a round. Done 2026-10-26 (`BOLT_DRAFT_GRAPH`, default on; see the changelog) and measured 2026-10-27:
   6.5 → 6.4 to 6.5 ms a proposal, 19.9 → 19.7 to 19.8 at sixteen streams, about 1 % more tokens a second alone and
   nothing at eight streams. The launches were not the cost; the proposal is bound by its
   kernels (LiLiCorr's five layers and the head over seven slots at one row each, memory-bound), so the next step for
   the drafter is fewer or fused kernels, or a smaller read (Q8_0 already halved it). Kept on: it costs nothing and
   removes about 130 launches a round from the host thread.
4. **`lm_head` in NVFP4** (0.54 → 0.27 GB a window, about 1 ms a round, 2 %): quality has to be checked; the
   output layer is the one place NVIDIA's own NVFP4 exports usually keep in higher precision.
5. **Acceptance**: measure accepted tokens a round on NVIDIA's categories to confirm the port matches their 4.7;
   if prose really is short, the lattice walk and the features are the place to look. A gated n-gram drafter for
   prompt-copying workloads.
6. Not worth it here: tree verification (small gain, expert reads per branch), a bigger draft model (reads more
   than it saves at 273 GB/s), FP4 tensor cores for the experts (IQ2_XXS is 2.06 bits; NVFP4 would double the
   bytes).

## Measured on 2026-10-19

Stage times from libbolt's own profiler (`BOLT_PROF=1`: eager launches with an event after every stage, so the
totals run about 10 % above graph mode; 100 blocks each). One stream, 6.6 rows a round, 40 ms a block:

| stage | ms a block | share | achieved | of |
| --- | ---: | ---: | ---: | --- |
| routed experts up + down (40 layers) | 15.8 | 39 % | ~150 GB/s | ~60 MB a layer (56 distinct experts) |
| Mamba `in_proj` | 8.0 | 20 % | 216 GB/s | 43 MB a layer |
| Mamba `out_proj` | 4.0 | 10 % | 170 GB/s | 17 MB |
| router `gate` (bf16 4096×512) | 2.8 | 7 % | 56 GB/s | 4 MB: latency, not bandwidth |
| `lm_head` | 2.4 | 6 % | 225 GB/s | 0.54 GB |
| attention q/k/v, o (8 layers) | 1.65 | 4 % | 160 to 190 GB/s | |
| route, group, norms, candidates | 1.9 | 5 % | | 150 small launches |

The drafter's proposal (`BOLT_DPROF=1`), 6.8 ms: the target's `lm_head` over the seven slots 2.4 ms (35 %), the
five layers' MLP and convs 2.5 ms, their attention 1.3 ms, the lattice head 0.4 ms. So `lm_head` is read twice a
round, 4.8 ms of 46.

Sixteen streams (code prompt, 8.5 streams and 44 rows a round on average, 123 ms a block):

| stage | ms a block | share | note |
| --- | ---: | ---: | --- |
| routed experts up + down | 53 | 43 % | |
| Mamba `in_proj` | 18.5 | 15 % | 463 µs a layer against 199 at 7 rows: 93 GB/s of weights |
| Mamba `out_proj` | 8.6 | 7 % | 216 µs against 100 |
| route + group | 6.9 | 6 % | 109 and 63 µs a layer, one CTA each |
| `lm_head` + candidates | 4.7 | 4 % | |
| gate | 2.7 | 2 % | |

Why the dense GEMMs slow down with rows: `mma_body` splits K over four warps and gives each CTA two 16-row weight
tiles; every warp loads its tokens' fp32 activations from L2 and packs them to halves at every K step. At 64 rows
that is 16 KB a step a warp, and `in_proj`'s 580 CTAs each re-read the whole 1 MB activation block: about 580 MB
of L2 traffic a layer against 43 MB of weights from DRAM. The prompt kernel stages activations in shared memory
but sums K in another order, and the recipe promises replies independent of the batch (`f5d1471`), so the fix
has to keep the K-split order: pack the rows to halves once a launch (same rounding) and give a CTA more row
tiles. The first half of that (`BOLT_PACK`, packed rows, 2 bytes an element and no per-step conversion) is
measured below.

Switches that did nothing (all within noise, 3 runs each): `BOLT_SPIN` (spin instead of yield on the host wait:
the engine syncs once a forward, unlike llama.cpp), `BOLT_MOE_BD=32` (32 pairs a MoE work item instead of 16).

Packed rows (`BOLT_PACK=1`, now the default; 3 runs each, aggregate tok/s, `BOLT_PACK=0` → `1`): eight streams
code 178 → 190 sampled and 273 → 285 greedy, prose 165 → 173 and 196 → 203; sixteen streams code 221 → 228 and
309 → 324, prose 203 → 213 and 222 → 234; one stream unchanged. `in_proj` at 48 rows went from 463 to 329 µs a
layer, the 128-row calibration round from 205 to 191 ms. Replies identical to the fp32 path in every cell.

The MoE statistic at sixteen streams (48 rows a round): 119 distinct experts a layer, 124 work items, so items
are not split and the kernel reads 128 MB a layer in 1.3 ms: 99 GB/s, against 150 GB/s with one stream. The
kernel is bound by instruction issue, not memory: it multiplies int8 weights by int8 activations and has to scale
and add every chunk's partial sum in fp32 (four FMAs and two shared loads per MMA), twice over with 16-pair
items. The dense IQ2 path folds the chunk scale into bf16 A fragments and lets the tensor core accumulate across
chunks with no epilogue; doing the same for the routed experts (bf16 activations in shared memory, scale folded
into the weight fragment) should bring the MoE stage near bandwidth: about 24 ms of a 118 ms sixteen-stream block
and 4 ms of a 40 ms single-stream window. Replies would change slightly (bf16 activations instead of int8 with
per-32 scales) and would need a quality check.

### Serialized stage profile (`BOLT_PROF=2`: the shared expert and `fc1` on the main stream, so each stage's time is its own)

Prefill, 1024-row chunks of a 10.5k-token prompt (606 ms a chunk, 1,650 tok/s):

| stage | ms a chunk | share | rate |
| --- | ---: | ---: | --- |
| routed experts up + down (int8 MMA, 203 distinct experts, 689 items a layer) | 187 | 31 % | 16 TFLOPS |
| Mamba `in_proj` (NVFP4) | 109 | 18 % | 57 TFLOPS |
| shared expert (dense IQ2, `mma_big_k`) | 87 | 14 % | 41 TFLOPS |
| Mamba `out_proj` | 51 | 8 % | 54 TFLOPS |
| Mamba conv + chunked scan | 43 | 7 % | |
| attention (8 layers, 10k context) | 30 | 5 % | |
| norms (`add_rms`, `gate_rms`) | 43 | 7 % | at bandwidth (100 MB each) |
| q/k/v, o, gate, route, group, `lm_head` | 56 | 9 % | route and group are 15 µs a layer: the 1 ms seen earlier was the shared expert's contention |

One stream (6.1 rows a window, 42.6 ms serial, 40 real): routed experts 14.2 ms (33 distinct experts, 36 MB a
layer in 354 µs: **100 GB/s**), shared expert 5.5 ms (11 MB in 137 µs: **81 GB/s**), `in_proj` 7.7 ms (216 GB/s),
`out_proj` 3.9, `lm_head` 2.4, conv + scan 2.0, attention 2.3, gate 0.9, the rest 2.7.

Sixteen streams (46 rows a block, 113 ms serial): routed experts 47 ms (112 distinct, 121 MB a layer in 1.17 ms:
103 GB/s), conv + scan 21.7 ms (16 states read and written: 128 MB a layer at 237 GB/s, the floor for fp32 state),
`in_proj` 12.4, shared expert 11.3 (39 GB/s: 2× slower per byte than with 6 rows, so instruction-bound),
`out_proj` 6.7, `lm_head` + candidates 4, the rest 9.

**So the IQ2_XXS tensor-core GEMMs are the lever in every regime**: 46 % of a single-stream window at under half
the bandwidth, 52 % of a sixteen-stream block, 45 % of a prefill chunk at a tenth of the tensor peak. The NVFP4
`in_proj` beside them runs at bandwidth with one stream. Two causes: (1) the routed kernel's int8 MMA needs a
scaled fp32 add per chunk per 8 tokens (instruction-bound; the bf16 kernel with folded scales below removes it);
(2) at small N the dense IQ2 kernel launches too few warps to hide latency (shared expert up: 336 row tiles, two a
CTA, 14 warps an SM), at many rows it is instruction-bound like the routed one. A proper MMQ-style kernel for both
(decode a weight tile to bf16 fragments once, all tokens of the CTA from shared memory, two-stage prefetch) is the
single most valuable piece of work left: it touches about half of every forward.

### What the IQ2 kernels are actually bound by

A bf16 version of the routed kernel (`BOLT_MOE_BF=1`: chunk scale folded into bf16 A fragments, bf16 activations,
no per-chunk epilogue) was **slower**: one-row window 29.3 against 26.1 ms, prefill 1,335 against 1,655 tok/s,
16 streams level. So the per-chunk scaled add is not the limit; the dequantization's table gathers are. Each lane
reads eight random 4-byte shared-memory words a chunk (grid halves and sign halves for two rows and two
sub-blocks); at the ~3-way bank conflicts random indices produce, that is about 24 shared-memory cycles for 64
weights (16.5 bytes) per warp, and shared memory is one unit an SM: 48 SMs × 2.6 GHz / 24 cycles × 16.5 bytes
≈ 90 GB/s, which is the measured rate. The remedy keeps the arithmetic exact: permute K inside each MMA so lane
t owns grid entry t of the chunk (real k 8t..8t+7 for its slots 4t..4t+3 and 16+4t..16+4t+3) and reads the
activations at the same real k; a 32-term int8 dot product is the same integer in any order, so replies do not
change, and the gathers halve to four 8-byte ones (`BOLT_MOE_HALF=1` keeps the old form). The sign table could
go too (the 7 sign bits and their parity expand to byte masks in a few ALU ops), leaving two gathers a chunk.
The dense IQ2 path (shared expert, `PIQ2::step`) gathers the same way and can take the same permutation, though
its bf16 MMA may then round differently (not bit-exact across builds).

Measured (3 runs each, aggregate tok/s): whole-entry gathers against half-entry, one stream 113 / 62 / 120 / 67
against 108 / 61 / 118 / 66 (code sampled, prose sampled, code greedy, prose greedy), sixteen streams level,
prefill 1,700 against 1,650 tok/s: **0 to 4 %**, kept because it is exact and never slower. So halving the
gathers did not move the kernel either. One tile a CTA for small-N dense GEMMs (`BOLT_RT1=1`) is 5 % slower at
sixteen streams (every CTA re-reads the activations) and level alone; off. Eight CTAs an expert item instead of
four (`BOLT_MOE_CTAS=8`): level. Replies identical in all of these (12 cells against the packed-rows build).

What is left as the explanation for the IQ2 kernels' ~100 GB/s is the DRAM access pattern: a warp reads sixteen
weight rows 32 bytes at a time per step (rows 2 KB apart), so each 128-byte line is fetched as four separate
sector reads over four steps, interleaved with fifteen other rows; LPDDR5X and the L2 do poorly with that. The
remedy is cooperative loads: the four lanes of a group fetch 64 or 128 contiguous bytes of a row per step and
exchange chunk words with shuffles. This should be settled with a standalone microbenchmark of `moe_mma_k`
(random routing, 1 / 8 / 48 / 1024 rows, GB/s per variant) rather than more full-server A/Bs, which take ten
minutes a point and resolve 3 %. That microbenchmark, then the cooperative-load variant, is the next step; the
same pattern applies to the dense IQ2 path and to the NVFP4 and Q8_0 tiles (`P::load`: 16 bytes a lane, one step
of prefetch), which may be why `out_proj` and `attn.o_proj` sit at 170 to 190 GB/s rather than 220.

### Prefill work (2026-10-19, afternoon)

The prompt-path dense GEMM (`mma_big_k`) streams a layer's weights once per 64-token group, so a 1024-row chunk
reads `in_proj`'s 43 MB sixteen times (690 MB a layer, 3.1 ms at 220 GB/s: the measured 2.73 ms), and every CTA
re-reads and converts its 64 tokens' fp32 rows for each pair of row tiles. Two steps, both keeping the sums:

- **Packed rows for the prompt path** (the scratch grew to 1024 rows): prefill 1,632 → 1,702 tok/s at 2.7k
  tokens, 1,680 → 1,745 at 5.3k, 1,637 → 1,696 at 21k (**+4 %**), the first token's hash unchanged.
- **Wide kernel** (`mma_wide_k`, `BOLT_WIDE`): 128-token groups, one row tile a warp, each weight fragment
  decoded once a step (`P::frag`) and applied to all 16 token tiles with the B fragments read tile by tile from
  shared memory, so the weights are streamed once per 128 tokens and the decode is amortized twice as far.
  146 to 180 registers, no spills. Measured **12 % slower** (1,465 against 1,655 tok/s at 21k, output identical):
  with one row tile a warp every B fragment read from shared memory feeds one MMA instead of two, and the kernel is
  shared-memory bound before it is DRAM bound. Splitting the policies' `step` into a fragment decode and the MMAs
  cost the existing prompt kernel 2.5 % by itself (the compiler no longer interleaves them), so both were taken
  out again; the lesson is that the next prompt GEMM has to be a different design, not a wider version of this
  one: eight warps in a 2×4 tile, weights and activations staged through shared memory with `cp.async`, 128×128
  output tiles (the MMQ shape), which keeps every fragment fed to several MMAs from both sides.

Items of 32 pairs for the routed experts at prefill (`BOLT_MOE_B=32`, four token tiles an item) were **17 %
slower** (1,440 against 1,700 tok/s at 21k), which says the routed kernel at prefill is bound by the per-token
work of each chunk (B fragments and scales from shared memory, the scaled add), not by the dequantization: more
tokens an item made it worse, the bf16 variant without the per-chunk add but with twice the MMAs was also worse.
A prefill-specific routed kernel would want bf16 MMAs (no per-chunk add) *and* wide token tiles, which the shared
memory budget only allows with K split in two passes; not attempted.

### Tensor-core ceilings, measured (2026-10-20)

`engine/tools/mma_peak.cu` runs bare `mma.sync` chains on all SMs:

| MMA | TFLOPS |
| --- | ---: |
| m16n8k16 f16 → f32 | 107 |
| m16n8k16 bf16 → f32 | 107 |
| m16n8k16 f16 → f16 | 107 |
| m16n8k32 s8 → s32 | 214 |

So fp16/bf16 `mma.sync` peaks at **107 TFLOPS** on GB10, half the int8 rate, whatever the accumulator (the 213
figure in the research summary was int8). The dense prompt GEMMs at 57 to 67 TFLOPS are therefore at 55 to 60 %
of their real ceiling, and a 2× on them needs int8 or fp8 operands, i.e. different numerics. The routed experts,
whose IQ2_XXS values are small integers and whose activations the engine already quantizes to int8 per 32, are
the one large GEMM that can use the 214 TOPS path exactly.

### Tile kernels against microbenchmarks

`engine/tools/gemm_bench.cu` (dense shapes, random weights in the tile layouts, outputs compared bit for bit) and
`engine/tools/moe_bench.cu` (512 experts, 1024 tokens × 9 with skewed routing). Both kernels keep the summation
order of the kernels they replace, and every output matched exactly.

- `mma_tile_k` (dense, fp16/bf16: 128×128 tile, 8 warps, weights dequantized once a step into shared memory,
  `ldmatrix` for both operands, three-stage `cp.async` ring for the rows, MMAs issued before the next step's
  dequant): 0.88 to 0.97× of `mma_big_k` on the NVFP4, Q8_0 and IQ2 shapes, 1.28× on the bf16 router. Both
  sit near 60 % of the 107 TFLOPS peak; the `cp.async`/`ldmatrix` structure buys nothing more on this part. Off
  (`BOLT_TILE=1` to try).
- `moe_tile_k` (routed experts for prompt chunks, int8: a CTA of 8 warps owns an expert's 128 rows × 64 of its
  tokens, items of 64 pairs, the IQ2 tile decoded to int8 in shared memory with its chunk scales, weight pieces
  and int8 token rows through three-stage `cp.async` rings, `ldmatrix`, the same `(dA·s)·dd` chunk epilogue):
  first version 1.09×, with the weight pieces moved from a two-step register prefetch into the `cp.async` ring
  1.25×, and skipping the MMAs of empty token tiles (an item averages 18 pairs at 1024 tokens, so most of a
  64-token tile is empty) **1.51×** on the microbenchmark, 20 to 21 TFLOPS of useful work. 128 registers, 48 KB
  of shared memory, two CTAs an SM. The remaining distance to the int8 ceiling is the low arithmetic intensity
  of routed experts at this chunk size (18 tokens an expert on average: a weight tile is used 18 times, so the
  kernel's floor is the weight bandwidth) and the per-item decode of 128 rows × K repeated for every 64-token
  item of a hot expert. End-to-end prefill with it is measured below.

End to end (`BOLT_MOE_TILE`, on by default): prefill 1,692 → 1,803 tok/s at 2.7k tokens, 1,736 → 1,855 at 5.3k,
1,697 → 1,814 at 21k (**+7 %**), the first token's hash unchanged at every size. Serialized profile of a 1024-row
chunk: routed experts up 2.09 → 1.48 ms a layer and down 2.58 → 1.79 (the stage −30 %), 285 work items a layer
instead of 689; the chunk 606 → 548 ms. The shared expert (dense IQ2, 2.35 ms a layer, now 17 % of the chunk)
could take the same int8 kernel as a single expert with all the chunk's tokens, but its activations would then
be int8 per 32 instead of bf16, which changes the prompt pass's numbers slightly; not done.

The shared expert on the same kernel (`BOLT_SHARED_TILE`, on by default): prefill 1,857 → 1,933 tok/s at 5.3k
tokens and 1,816 → 1,890 at 21k (**+4 %**). Its input rows are now int8 per 32 instead of bf16 for prompt chunks
(the routed experts have always had int8 rows), so this is the first change today whose numbers are not the
same by construction; measured, the first token's hash is unchanged at 2.7k, 5.3k and 21k tokens and a 200-token
greedy continuation after a 22.5k-token document is identical character for character. `BOLT_SHARED_TILE=0`
restores the bf16 path. With it the prompt chunk stands at about 530 ms against 606 at the start of the day
(1,900 against 1,650 tok/s); the dense NVFP4 projections (`in_proj`, `out_proj`: 27 %) are the largest piece left
and sit at 60 % of the fp16 tensor-core ceiling.

### Decode expert kernel: bytes in flight (2026-10-20)

`moe_bench.cu` at decode sizes (uniform routing, GB/s over the distinct experts' weights): the current kernel
reaches 190 GB/s with 9 items (one row), 174 with 50 (six rows), 137 with 67 (eight rows) and 125 with 301
(48 rows). Not the access pattern after all but Little's law: a warp keeps 128 bytes of weight loads outstanding
(two rows × 32 B, one step ahead), and at a loaded latency over a microsecond that caps an SM far below the
bandwidth; more items only add contention. `moe_coop_k` (`BOLT_MOE_COOP`, on by default for more than one row's
nine experts) keeps the items, tiles and arithmetic and streams each warp's two rows through a per-warp
three-stage `cp.async` ring, 64 bytes a row a step and two steps ahead (about 2 KB a warp in flight): 226 GB/s
at six and eight rows, 196 at 48, outputs identical. End to end (3 runs each, aggregate tok/s, old → new):

| cell | old | new |
| --- | ---: | ---: |
| 1 stream, code / prose greedy | 118 / 66 | **135 / 74** |
| 1 stream, code / prose sampled | 110 / 63 | 132 / 70 |
| 8 streams, code / prose greedy | 289 / 201 | 339 / 243 |
| 16 streams, code / prose greedy | 327 / 240 | 383 / 281 |
| 16 streams, code / prose sampled | 230 / 210 | 275 / 251 |

The one-row window went from 25.3 to 27.3 ms (the lone row keeps the old kernel; the difference is noise) and
the eight-row window from 39.5 to 34.1; the 128-row calibration round from 190 to 161 ms. Replies identical in
every cell. The same treatment would apply to the dense NVFP4 and Q8_0 decode GEMMs (`mma_body`: 16 bytes a lane,
one step ahead), which the profile has at 170 to 216 GB/s.

Programmatic dependent launch on top (`BOLT_PDL`, default on; every forward kernel waits at entry, no early
triggers): one stream greedy code 131 → 137 tok/s, sampled 130 → 133, prose +1 to 2 %, the rest level; the
eight-row window 34.2 → 33.4 ms. Exact. The day's decode position: **137 / 74 tok/s alone (code / prose), 385 /
284 at sixteen streams**, from 116 / 68 and 325 / 239 this morning, all with replies identical to the 1.0.12 build.
Items 1 and 2 are done; what is left for prose alone is the accepted tokens a round (2.8 against code's 5.5),
and for everything the dense NVFP4/Q8 decode GEMMs' own bytes in flight (same remedy as the expert kernel).

### Acceptance, measured by category (2026-10-20, `~/bench/accept.py`: 400 greedy tokens a prompt)

| category | accepted tokens a round | proposed a round | acceptance | tok/s alone |
| --- | ---: | ---: | ---: | ---: |
| writing (essay / story) | 2.45 / 2.58 | 4.3 / 4.9 | 34 / 32 % | 60 / 62 |
| roleplay | 2.42 | 5.0 | 29 % | 58 |
| chat (explain / advice) | 3.08 / 2.90 | 5.0 / 5.8 | 41 / 33 % | 74 / 68 |
| math | 5.80 | 6.8 | 70 % | 126 |
| code | 4.94 | 6.9 | 57 % | 109 |
| summarize a 3.4k document | 6.35 | 6.7 | 80 % | 83 |
| JSON edit / text rewrite | 7.27 / 7.14 | 6.9 / 6.7 | 91 / 91 % | 116 / 119 |
| all | 3.72 | | | |

Sampled (temperature 1, top-p 0.95) runs 5 to 15 % lower on prose and code. Against NVIDIA's table 5 (greedy
coding 5.69, writing 3.70, roleplay 3.75, mean 4.68): code and math match, prose is a third short, copy tasks
are at the eight-row ceiling (7.1 to 7.3 of a possible 8). The port was checked against vLLM's `lilicorr.py`
(features, node sum, slot-relative and same-slot biases, RMS norms, factor over [h, a, h·a], L2-normalized
cosine edges, the greedy walk from the anchor) and matches it. The decisive check was an ablation: the walk
forced to rank 0 at every slot (`BOLT_DRAFT_TOP1=1`, plain DFlash) gives 3.47 overall, 2 to 9 % less on prose,
12 % less on math, 21 % less on code, which is NVIDIA's reported gain for the lattice. So the gap to their prose
figure is their evaluation set and target precision, not the port. What this means for decode alone: prose is
bounded near 75 to 80 tok/s on this box with this drafter (2.5 tokens a round over a 33 ms round), code and
structured output sit at 110 to 150, and the way to more on prose is a better draft model, not this engine.

### Dense decode GEMMs: the same ring, a null result (2026-10-20)

`gemm_bench.cu` now cycles enough weight copies to defeat the 24 MB L2 (the first numbers it printed, up to
950 GB/s, were L2 hits). From DRAM, the K-split decode kernel reaches 238 to 245 GB/s on `in_proj` and `o_proj`
at one to eight rows, 218 to 235 on `out_proj` and `fc2`, 177 to 185 on `fc1` and the bf16 router (4 MB
matrices: 25 µs launches), and 140 to 215 at 48 rows. A per-warp three-stage `cp.async` ring for the weight
pieces (`BOLT_DENSE_RING=1`, bit-identical) measured 0.92 to 1.02×: these kernels are at the DRAM ceiling
already, unlike the expert kernel, so the ring buys nothing and stays off. The dense IQ2 path (the shared expert,
`PIQ2` in the same kernel) is the outlier: 86 to 100 GB/s at one to eight rows and 40 to 50 at 48, the slowest
kernel family left in decode; it moves to the int8 deep-prefetch expert kernel (`BOLT_SHARED_DECODE`, measured
below), which changes its rows from bf16 to int8 per 32 as the prompt path already does.

### The shared expert at decode: faster kernels, no round to show for it (2026-10-20)

Two faster paths for the shared expert's two GEMMs in verify blocks were built and measured:

- `BOLT_IQ2_WHOLE=1`: the bf16 IQ2 policy with whole grid entries (K permuted inside each MMA, one 16-byte table
  read for eight values instead of eight 4-byte ones, B fragments as one 8-byte load). On the microbenchmark
  1.8 to 2.0× at one to eight rows (97 → 185 GB/s) and 1.5× at 48; differences at the fp32 rounding level (max
  relative 1e-4 to 1e-3 on normal-sized outputs).
- `BOLT_SHARED_DECODE=1`: the int8 deep-prefetch expert kernel as a single expert over the block (the same
  path prefill uses), with its grid widened to every tile group; int8 rows per 32 instead of bf16.

End to end neither moves the window: the shared expert runs on a side stream under the routed experts' chain,
so its latency is hidden at every stream count; the eight-row window stays 33 to 35 ms and the cells move
within noise (code greedy alone 137 → 136 / 142, prose greedy 74 → 79 / 84 but with different replies, so the
acceptance path differs too). Both change greedy replies in 7 to 8 of 10 prompts (fp32 reordering in the first,
int8 rows in the second) for no reliable gain, so both are off; `docs` and the switches stay for a profile that
accepts changed numerics. The lesson for the critical path: at one stream it is the serial chain embed → norm →
`in_proj` → conv/scan → `out_proj` → gate/route/group → experts → `fc2` → `lm_head` on the main stream, all of
which are now at the DRAM ceiling or launch-bound, which is why the window sits at 25 ms for one row against a
19 to 21 ms byte floor and further exact gains there are small.

### Prefill, second round (2026-10-20): 2048-row chunks

The prompt chunk is 2048 rows now (`TMAX`; the lane core's prompt pieces were already 2048). Each expert sees
twice the tokens a chunk, so the routed-expert kernel's items are fuller: on the microbenchmark it does 27 to 33
TFLOPS of useful work at 2048 tokens against 19 to 22 at 1024. End to end: 1,913 to 1,946 tok/s at 2.7k tokens
(+2 %), 2,022 to 2,031 at 5.3k (+5 %), 1,943 at 21k (+3 %), 1,768 at 42k (+3 %); first-token hashes, the
22.5k-token continuation and decode replies unchanged. Serialized profile of a 1612-row average block: `in_proj`
20 %, shared expert 14 %, routed experts 22 %, `out_proj` 11 %, conv + scan 8 %, attention 7 %, norms 9 %.
Reordering the expert tile kernel to issue all of a step's MMAs before their scaled adds was slightly slower
(register pressure at the 128-register cap) and was dropped.

### The prompt scan (item 3): two attempts, both parked

`scan_ssd_k` (`BOLT_SCAN_SSD=1`, `engine/tools/scan_bench.cu`): Mamba-2's chunked form on m16n8k8 tf32 MMAs,
64-row chunks, the state fp32 in shared memory, n in two halves. Correct (y rms relative difference 3.8e-4, the
state 2.8e-4, both tf32-level) but **2.5× slower** than the sequential kernel: 86 KB of shared memory means one
CTA an SM, and each chunk's loads, eight barriers and a serial prefix sum leave the tensor cores idle. A paying
version needs the state in registers (32 a lane in the B-fragment layout), double-buffered `cp.async` loads, a
warp-parallel prefix and two CTAs an SM: a day's work for about 4 % of prefill, so it is parked as a reference.
A four-threads-a-channel variant of the sequential kernel was 2.4× slower (exact): that kernel is bound by its
per-step shared-memory broadcast reads of B and C, and doubling the warps a block doubles them.

### Item 5: the NVFP4 projections on the block-scaled fp4 tensor cores (2026-10-20)

GB10's `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64 ... e2m1.e2m1.f32.ue4m3` takes both
operands as e2m1 nibbles with an e4m3 scale a block of 16: exactly NVFP4. The weights keep their nibbles and
block scales (`tile_nv4w_k` only reorders them into the MMA's fragment order: a lane's 16-byte piece is its A
fragment as stored); the rows are quantized the same way at run time (`quant_nv4_rows_k`: scale = amax / 6 a
block, rounded to e4m3, values to the nearest e2m1), which is what vLLM's NVFP4 path does with these
checkpoints. `mma_nv4_fp4_k`: 128 rows × 128 tokens a CTA, 64 k a step, `cp.async` rings for the
weight pieces, their scales, the rows and their scales; no dequantization anywhere. The recipe's libbolt build
targets `sm_121a` now (the block-scaled MMA is refused on plain `sm_121`). Checked exact against a CPU reference
computed from the same quantized inputs (`gemm_bench.cu ... fp4`); 133 TFLOPS on `in_proj` (2.15× the fp16
path) and 98 on `out_proj` (1.68×). The difference to the fp16 path is 9.5 % rms, the fp4 rows themselves.

End to end (`BOLT_NV4_FP4=1`, prompt chunks only): prefill 1,913 → 2,330 tok/s at 2.7k tokens, 2,027 → 2,425
at 5.3k, 1,944 → 2,314 at 21k, 1,768 → 2,078 at 42k (**+17 to 20 %**). The first token's hash is unchanged at
every size, but a 200-token greedy continuation after a 22.5k-token document diverges after 76 characters, so
this is a real numerics change and the perplexity harness (`engine/tools/ppl.cu`, through libbolt's C ABI on a
22k-token prose and a 22k-token code sequence) decides it.

Perplexity (`ppl.cu`, 22,473 prose tokens from this repo's docs and 22,159 code tokens from `bolt_kernels.cu`,
blocks of 128 rows, run with the test server stopped because the lazy KV mapping refuses to grow under the
memory reserve):

| path | prose | code | all |
| --- | ---: | ---: | ---: |
| decode-path kernels (bf16 shared expert) | 5.6280 | 1.6293 | 3.0413 |
| prompt-path kernels (int8 shared expert, routed tile) | 5.6287 | 1.6296 | 3.0418 |
| prompt path + fp4 projections (`BOLT_NV4_FP4`) | 5.6968 | 1.6353 | 3.0657 |
| decode path + `BOLT_IQ2_WHOLE` + `BOLT_SHARED_DECODE` | 5.6287 | 1.6296 | 3.0418 |

So the int8 rows of the shared expert cost 0.02 % and the fp4 rows of the two projections 0.8 % (prose 1.2 %,
code 0.4 %), for +17 to 20 % prefill. `NV4_FP4=1` is the default with that number stated; `NV4_FP4=0` keeps the
fp16 rows. The two parked decode switches are numerically free by this measure but were not faster end to end,
so they stay off.

### Tuning the fp4 kernel: 133 → 154 TFLOPS

The shipped kernel sat at 133 TFLOPS on `in_proj` and 98 on `out_proj` of a measured 429 peak (`mma_peak.cu`,
the block-scaled MMA alone). Each step below was measured on `gemm_bench.cu ... fp4` (2048 rows, best of
five, exact against the CPU reference throughout):

| change | `in_proj` TFLOPS | `out_proj` TFLOPS |
| --- | ---: | ---: |
| as shipped (three-stage ring, 32-byte token rows) | 133 | 98 |
| token rows at a 48-byte stride (conflict-free B reads) | 137 | 101 |
| B fragments through `ldmatrix` | 136 | 106 |
| row scales step-major (`[k/64][T][4]`), 16-byte `cp.async` loads | 154 | 113 |
| two steps of 64 k a barrier, two-stage ring | 154 | 118 |
| 256-row CTA of 16 warps (one CTA an SM) | 142 | 111 |
| 128-row CTA, one CTA an SM (136 registers) | 126 | 102 |

The two losers are not kept. The kernel is at 126 registers with two CTAs an SM; what keeps it at 36 % of
peak is the shared-memory traffic a step (each warp re-reads 16 token rows and 2 weight tiles for 32 MMAs) and
the barrier a stage, the usual gap between `mma.sync` with manual staging and a `wgmma`/TMA pipeline, which
GB10 does not have. End to end +2 %: 2,305 to 2,388 tok/s at 2.7k tokens, 2,469 at 5.3k, 2,360 at 21k, 2,115 at
42k. The step-major scales first crashed prompt chunks whose row count was not a multiple of 4 (a 16-byte
`cp.async` from a 4-byte-aligned address; `gemm_bench` had only run 2048 rows), so the stride is padded to 4 rows
and the microbenchmarks now run odd row counts too. `engine/tools/fp4_cmp.cu` compares the tuned kernel with the
shipped one (kept as `old_fp4.inc`) on rows spanning 16 binades: bit-identical on 118 million outputs at 2048,
1440, 945, 601, 128, 7 and 1 rows, and the perplexity harness repeats 5.6968 / 1.6353 / 3.0657 exactly.

### Prompt attention at long context: 29 → 62 TFLOPS, identical outputs

`engine/tools/attn_bench.cu` runs the prompt attention (`attn_mma_part_k` + `attn_mma_comb_k`, splits 4) for a
2048-row chunk against a bf16 cache of 8k, 40k and 128k earlier keys and checks the variants bit-exact against the
shipped kernel and a few rows against a CPU softmax. The shipped kernel sits at **29 TFLOPS at every context
length** (of the 107 fp16 `mma.sync` peak), so it is instruction-bound, not memory-bound: it stages K and V
through registers with 2-byte stores for the V transpose, builds every B fragment from 4-byte shared-memory
loads, and takes two barriers a 32-key tile. At 40k context the eight attention layers cost 8 × 50 ms a chunk,
about 40 % of the chunk; at 128k, 8 × 153 ms, which is why prefill fell to 987 tok/s there.

`attn_mma2_part_k` (`BOLT_ATTN_V2`, default on, bf16 cache) keeps the per-warp math of the first kernel (one
query row's 16 heads against 32 keys a softmax step, the same MMA order, so the outputs are **bit-identical**)
and changes only how the operands arrive: K and V tiles come in by `cp.async` (zero-fill past the range) into a
two-stage ring of 64 keys, V stays `[key][dim]` and is transposed by `ldmatrix.trans`, K fragments come from
`ldmatrix`, one barrier pair a 64-key stage. 201 registers, one CTA an SM. The grid also puts the query block
first so the CTAs resident at once walk the same key range (that alone was 12 % at 40k).

| context | shipped | query block fastest | v2 64 keys × 2 stages | v2 32 × 3 | v2 32 × 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8k | 10.4 ms, 30 TFLOPS | 10.6 | **6.1 ms, 51** | 6.0 | 6.1 |
| 40k | 50.2 ms, 28 | 44.1 | **22.8 ms, 62** | 23.1 | 23.4 |
| 128k | 153 ms, 28 | 148 | **68.1 ms, 64** | 69.1 | 69.9 |

End to end (`BOLT_ATTN_V2=1`, now the default; the 42k to 126k rows measured against `BOLT_ATTN_V2=0` on the same
build, the rest against the fp4-tuned build before it): 2.7k tokens 2,312 to 2,408 tok/s (level), 5.3k 2,493 to
2,508 (+1 %), 21k 2,360 → 2,483 (+5 %), 42k 2,115 → 2,365 (+12 %), 63k 1,923 → 2,254 (+17 %), 126k 1,499 → 1,973
(+32 %). First-token hashes at every size, the 22.5k-token continuation and the one- and sixteen-stream decode
replies are unchanged (decode 137 code / 73 prose alone, 379 / 280 at sixteen streams).

(64 keys × 3 stages and 128 × 2 need more than GB10's 99 KB of shared memory a CTA.) Forcing two CTAs an SM
(128 registers) spills and is 15 % slower. What is left in it: the online softmax's rescale of 64 accumulators
and 18 `exp`s a 32-key step per warp are about as many issue slots as the 64 MMAs, with only eight warps an SM
to overlap them; a 64-key softmax step or FA3-style lazy rescaling would help but changes the rounding.
