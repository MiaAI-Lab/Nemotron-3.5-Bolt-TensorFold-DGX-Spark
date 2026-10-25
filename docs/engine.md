# The Bolt backend and engine

`./start.sh` builds TensorFold at `eab8582` (v1.1.4), applies upstream CUDA build fixes and Bolt registration in `patches/`, copies `engine/bolt_native.zig` and `engine/bolt_api.zig` into
`zig/src/families/nemotron/` and compiles `engine/*.cu` into `/opt/tensorfold/lib/libbolt.so`. `tensorfold-native`
opens the checkpoint through the `nemotron_h_puzzle` family that `patches/0002` registers, loads libbolt from its own `lib/` and serves.

## Why a separate library

TensorFold's native Nemotron kernels serve Lightning: hidden 2688, 128 experts, top-6, MLX 4-bit or ModelOpt NVFP4
weights. Bolt is a different shape and packing.

| | Lightning | Bolt 24GB |
| --- | --- | --- |
| Hidden / layers | 2688 | 4096, 88 (40 Mamba, 40 MoE, 8 attention) |
| Routed experts | 128, one width, top-6 | 240 to 496 a layer, widths 1536 to 2560, top-9 |
| MoE | shared expert | latent size 1024 around the routed experts, shared width 5376, sigmoid router, routed scale 5.0, ReLU² |
| Quantization | MLX affine 4-bit / NVFP4 | IQ2_XXS experts (ggml), NVFP4 Mamba projections, Q8_0 attention, latent projections and `lm_head` |
| Draft head | MTP in the checkpoint | none; the separate LiLiCorr checkpoint |

## The lane backend (`engine/bolt_native.zig`, `engine/bolt_api.zig`)

TensorFold's round loop (`core/lanes`) asks a backend for a handful of things. Bolt answers them over the C API of
libbolt (`bolt_api.zig` holds the C ABI binding and finds the library; `bolt_native.zig` is the backend). Every live stream owns one *sequence* of libbolt (a `Slot`, found from the stream's pointer); the lane host
asks for a stream only after a sequence is free, otherwise the backend returns `LaneMemory` and the job waits at the
head of the queue.

| Backend entry | What it does |
| --- | --- |
| `prefill_step` | takes a sequence, resumes it from a kept prompt state when the cache has one, then runs the prompt in pieces (2048 rows alone, 512 while other streams decode, cut at the cache's marks) so a cancel lands quickly and the other streams' rounds run between pieces |
| `first` / `queue` / `first_masked` | the keyed draw of one row at the given position; `first_masked` draws the first token again among the tokens a grammar's mask allows |
| `verify` | **one forward over every stream's window** (up to 16 windows, 128 rows): `pending`, the held drafts and any host drafts, each row drawn with its stream's `Sampling` at its own position (among the allowed tokens when the window carries a grammar's masks) |
| `keep` | roll each stream back to its accepted prefix (`bolt_keep_seq`; a repeat is a no-op) |
| `draft` | cuts each stream back to its kept rows first (a shared round drafts before it keeps), then LiLiCorr proposes seven tokens for every stream in **one pass over the drafter's weights**; the first `depth` are held for the next window |
| `release` | gives the sequence back and unmaps the attention memory it grew into |

Draws use `lanes.sampling.choose` (Gumbel over the top candidates, `top_k` / `top_p` / `min_p`, keyed by seed and
position), so a speculative run draws what a plain run would. libbolt returns the top 20 or 64 candidates a row
(1 when greedy); `top_k` above 64 is capped. Facts reported to the round loop: window 8, drafts 7, window costs and
shared-round costs measured at load (rows 2 to 128 over all sequences), the drafter's cost, `hidden_rows` and
`draft_streams` on when there is more than one sequence, acceptance prior 0.80 falling to 0.68. Logprobs and
trees are not served by this backend; the round loop is told so.

`--parallel N` sets how many streams run together; the backend hands N to libbolt as `BOLT_SEQS` (unless the environment
sets it), which sizes the recurrent state. Extra jobs beyond N queue.

## libbolt (`engine/`)

- `bolt_host.cu`: loading, the block forward, the C API (`bolt_abi` is its version, 3, which the backend checks at load;
  `bolt_keep_multi` cuts every stream of a round back in one call;
  `bolt_prefill_seq_rows` takes image rows by device pointer; `bolt_mask_rows` redraws rows under a grammar's masks), sequences, snapshots, calibration. A *block* is a list of
  segments, each a run of rows of one sequence (a prompt chunk is one segment; a shared decode round has one verify
  window per stream). `Ctl` carries what changes between replays: tokens, and per segment its first row, cache length,
  sequence and state slots. A block of up to 1024 rows (128 for a shared round) is one launch sequence, captured into
  a CUDA graph the second time a shape repeats; shared rounds are padded to a few row counts (multiples of 8, then 16)
  so a handful of graphs serve them. The shared expert and the latent projection run on side streams.
- `bolt_mma.cu`: tensor-core GEMMs for NVFP4, Q8_0, bf16 and IQ2_XXS from 16-row tiles (layouts in the file header);
  `mma_big_k` stages activations in shared memory for prompt blocks over 32 rows, while decode and verify blocks
  (`mma_decode_rows`) use the K-split kernel at any row count, 64 tokens a CTA. Prompt attention and verify-window
  attention on bf16 mma with online softmax (`attn_mma2_part_k` for prompt chunks with the bf16 cache: `cp.async`
  staging and `ldmatrix` fragments, `BOLT_ATTN_V2=0` for the first kernel; `attn_mma_part_k` for windows, one CTA a
  stream's window, and the fp8 cache). The prompt pass's NVFP4 projections run on the block-scaled fp4 MMA
  (`mma_nv4_fp4_k`, `BOLT_NV4_FP4`).
- `bolt_kernels.cu`: norms, the Mamba conv and scan (a row-parallel conv and a thread-pair scan for long blocks, a
  warp-per-channel scan for windows), the router, the routed experts on int8 tensor cores, candidate selection.
- `bolt_draft.cu`: the LiLiCorr drafter: five DFlash layers (grouped conv, sliding-window sink attention, NVFP4 MLP)
  over the six captured residuals, then the candidate lattice and its greedy walk. A call drafts for up to 16 streams;
  the kept rows of every stream are ingested in one pass too.

### Images and video

TensorFold's RADIO tower (`families/radio`) encodes pictures; libbolt only takes their rows. The backend reads each prompt
piece's images (`Request.images`: patches, grid, the placeholder row they start at) and, for the images the piece
crosses, runs the tower into a 16,384-row rows buffer and calls `bolt_prefill_seq_rows` with runs of
`{pos, count, device pointer}`. libbolt copies each run over the block's embedding rows before the layer graph and
reads placeholder tokens (`<image>`) as those rows; it synchronizes its stream before returning, so the next piece
reuses the buffer. An image a piece cuts is encoded again by the next piece, which keeps streams independent: no image
state lives between pieces, so several image prompts interleave. A video is the same: each pair of sampled frames is
one tower input whose patches hold both frames, through the checkpoint's `video_embedder` projection.

### Structured output

The server turns a grammar (`response_format`, `guided_*`, `tool_choice` `required` or a named function) into allowed-token
bits for each row of a stream's verify window: `rows x words` u32, 4,096 words for the 131,072-token vocabulary, bit `t` of
a row set when token `t` may be drawn there. `verify` collects the masked rows of every window in the block, runs the
forward as usual, then calls `bolt_mask_rows` once: a masked variant of the two-stage GPU top-k (`cand1m_k` reads a row's
mask and treats a disallowed token as minus infinity; the merge stage is the ordinary one) that recomputes only those rows'
candidates, and the keyed draw then picks among them (a row with fewer allowed tokens than candidates is drawn from the
allowed ones). The first token is drawn from the prompt's last row, which is still row 0 of the logits, so `first_masked`
runs `bolt_mask_rows` on that row under the first mask. The grammar also needs each drafted token to build the next row's
mask, so `held()` returns the drafts the drafter holds (they are host tokens already). Rows without a mask, and other
streams' rows in the same block, are untouched, and a round with no masks costs what it did; the kernel test compares the
masked top-k with a CPU reference for a mask of 5 tokens, about half the vocabulary and a single token.

### State of a sequence

- **SSM**: two copies of the 4 MiB-a-layer fp32 state. A block reads one and writes the other, so a verify window only
  writes its final state; `keep(n)` for a partial accept replays the first `n` rows from the pre-window copy
  (`k_scan_replay`, using the window's saved scan inputs). Rolling back to zero rows is a pointer flip.
  With `BOLT_SSM_LAZY=1` (`SSM_LAZY`) there is one copy: a verify window writes no state, the conv saves each row's scan
  inputs in one of two per-sequence slots of eight rows, `keep(n)` records `n` (`Seq.pending`), and the sequence's next
  block folds those rows in at the start of its scan (`Ctl.seg_replay`, `seg_hist`), writes the state once and runs its
  own rows from it; a prompt chunk commits its final state as before, and a snapshot with rows pending folds them first
  (`k_scan_fold`). The default: measured on a Spark, replies are the same token for token and 4 to 16 streams decode 3 to 8 %
  more tokens a second (one stream the same). `BOLT_SSM_LAZY=0` keeps two copies.
- **Prompt chunks' NVFP4 projections** (`NV4_FP4`, default 1): `in_proj` and `out_proj` run on the block-scaled fp4
  MMA (`kind::mxf4nvf4`, m16n8k64, e4m3 scales a block of 16) with the rows quantized to NVFP4 first (`quant_nv4_rows_k`);
  the weights are read as stored, reordered into the MMA's fragment order at load (`tile_nv4w_k`, a second copy). Verify
  blocks keep the fp16 rows. Perplexity +0.8 % on `engine/tools/ppl.cu` for +17 to 20 % prefill; libbolt builds for `sm_121a`.
- **Conv**: a ring of nine slots; a window writes the state after each row, `keep(n)` picks slot `n`.
- **Attention K/V**: bf16, 8 KB a token; with `BOLT_KV=fp8` (`KV_DTYPE=fp8`) e4m3 bytes plus an fp32 scale per token and
  head (amax / 448, set when the row is stored), 4.1 KB a token. The kernels dequantize while staging tiles, so the
  cache costs no speed. Upstream TensorFold has no Nemotron cache quantization (its int8/int4 `--kv-dtype` is Flash
  Next's) and the checkpoint ships no KV scales, hence the dynamic ones. Reserved at the full window for every sequence but mapped 4096 tokens at a
  time (CUDA virtual memory, 2 MiB granules) as the sequence grows, and unmapped down to one granule when it is
  released, so 16 streams can each have the 262,144-token window while memory follows use. If the driver cannot do
  that, or `BOLT_KV_EAGER=1`, everything is allocated up front. `BOLT_KV_RESERVE_GIB` (default 3) is the free memory
  a growing stream leaves alone; past it the forward fails instead of freezing the machine.
- **Drafter**: a 2048-entry K/V ring per layer and the anchor row; prompt rows older than 1,100 never reach it.

Memory a sequence holds from load: 0.41 GiB (two SSM copies over 40 layers, the conv ring) plus the drafter's 21 MB;
0.28 GiB with one copy (`SSM_LAZY=1`, which adds 27 MB of saved scan inputs).

### Why a round costs what it does

A shared round is bound by DRAM, not arithmetic. At 16 streams with one row each it reads, per round: every stream's
SSM state in and out (40 layers × 4 MiB × 2 × 16 = 5.4 GB), about 130 distinct experts a layer (6.9 GB), the dense
weights (2.8 GB). That is the 85 ms the profile shows (`BOLT_PROF=1`), and about 80 % of
the GB10's practical bandwidth. More rows per stream (draft depth) cost the extra experts they touch but no more state,
which is why TensorFold's cost rule asks for longer windows when streams are few. Measured shared rounds, ms by total
rows over 16 sequences, before the fixed summation order below: 16 → 61, 32 → 96, 64 → 141, 128 → 197. With it, 16
streams of 4-row windows (64 rows) take 141.7 ms and 16 streams of 8-row windows (128 rows) 220.3 ms; 8 streams of
8-row windows (64 rows) 130.4 ms against 135.5 before.

The round loop prices a shared round as a table by total rows plus an overhead it learns per stream count, so the
table libbolt reports (`bolt_shared_costs`) holds what rows cost, not streams: one stream's window costs for 2, 4 and
8 rows, then a line at the cost of a row, measured at load as all sequences with 1 row each against 8 each (the same
states, more rows). `BOLT_CAL=spread` reports the older table instead, measured with the rows spread over as many
sequences as possible, which carries every sequence's state traffic into the price of a row. The drafter's proposal is
a flat cost a round (seven drafts in one pass whatever the depth), reported as `mtp_step_ms` 0 and measured by the loop
as overhead.

## Checks

The engine is checked against references that live in the development harness, not in this repository: kernel,
tensor-core GEMM and attention unit tests (attention includes several windows in one launch), and engine-level gates:
perplexity 5.284 on the first 1,499 tokens of this README (5.288 through 8-row windows), a window cut to `n` rows equal
to a window of `n` rows bit for bit, a resumed snapshot equal to a whole prefill, the drafter accepting 4.3 of 7 tokens
on the model's own text, sixteen sequences decoded in one block each equal to its solo run (768 of 768 tokens, plain
steps and 4-row windows alike), and a sequence's logits bit-identical whatever its neighbours hold. `BOLT_PROF=1` times
each stage of a round.

**Rounding does not depend on the batch.** Three things used to make a stream round differently among other streams
than alone, and each is now fixed by the stream's own state alone:

1. *Attention key chunks.* A verify window's attention merges partial softmaxes over chunks of keys. The chunk size is
   `win_chunk(base)`: 128 keys, doubled until 32 chunks reach the sequence's end and its window, so it follows the
   sequence's length and nothing else (the launch merely has room for the most any sequence needs, and unused chunks
   contribute exact zeros). Before, the chunks followed the number of streams in the block (about 32 splits for one
   stream, 8 for four) and the window's width.
2. *GEMM summation order.* Blocks of more than 32 rows used `mma_big_k`, which walks K in one sequence, while the
   K-split kernel sums four interleaved partials; the two round differently. Decode and verify blocks now always use
   the K-split kernel (`mma_decode_rows`, 64 tokens a CTA, so the weights are decoded once per 64 tokens); only the
   prompt pass keeps `mma_big_k`.
3. *The plain step.* The non-speculative one-token step used the CUDA-core attention kernel. It now runs as a one-row
   window (then kept), so drafted and plain decoding share the same arithmetic.

Before this, a greedy reply followed a near tie the other way after some tokens whenever the batch differed (prose
prompts at four streams all differed from their solo runs). Now perplexity of passages scored alone and in a block of
2, 3, 4, 8 or 16 streams, with 8-row windows (up to 128 rows a block; also 1- and 5-row windows at four streams), is
identical to five decimals;
sixteen sequences match their solo runs for 768 of 768 tokens, plain steps and 4-row windows alike; and
`bench_concurrent.py --alone --serial` finds concurrent replies equal to solo ones and drafted equal to
`"draft": false`. The cost is in the widest rounds: 128 rows a round (16 streams of 8-row windows) take about 25 %
longer (176.9 → 220.3 ms), while 64-row rounds are unchanged or faster.

## Checking a snapshot

`scripts/ckptcheck.py` checks a snapshot against `contract/bolt.json`. A change that serves different shapes fails that
check on purpose.
