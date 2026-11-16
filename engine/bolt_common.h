// Shared declarations for the Bolt engine: packed weight views, the per-launch control block and the
// kernel launchers in bolt_kernels.cu. Everything here is C++ and private to libbolt.
#pragma once
#include <cuda_runtime.h>
#include <cstdint>

constexpr int TMAX = 2048;     // most rows one block forward carries (prompt chunks; the lane core's prompt pieces are 2048 rows)
constexpr int BATCH_MAX = 128; // most rows of a block that carries several sequences' verify windows
constexpr int LOGIT_ROWS = BATCH_MAX; // rows whose logits are kept (verify windows and the last prompt row)
constexpr int SEG_MAX = 16;    // most sequences in one block
// Rows one sequence's verify window carries at most, in any model: a model's own limit is its `win` (BOLT_WINDOW: 8, 16
// or 32, set at load). The drafter proposes seven tokens, so eight rows hold every drafted window; a wider one lets the
// lane core verify a longer copy of earlier text in one forward, for a conv ring (win + 1 slots) and saved scan inputs
// (two slots of win rows) that grow with it: 0.28 GiB a stream at 8, 0.33 at 16, 0.46 at 32.
constexpr int VERIFY_MAX = 32;
constexpr int DRAFT_ROWS = 8;  // rows of a drafted window: the pending token and the drafter's seven
constexpr int WIN_Q = 8;       // query rows one CTA of a window's attention carries
constexpr int QB_MAX = (BATCH_MAX + (WIN_Q - 1) * SEG_MAX) / WIN_Q;  // most query blocks a shared round's windows split into
constexpr int HIST_ROW = 10240;             // floats a saved scan-input row holds (x after the conv, B, C)

// Keys one CTA of a verify window's attention covers: 128, doubled until 32 CTAs reach the end of the sequence and a
// drafted window. It depends on the sequence's own length alone, so a stream's rounding is the same alone, among other
// streams, or in a wider window; the split of the keys is the only part of the forward whose order depended on the batch.
// (A window wider than DRAFT_ROWS that starts within its width of 32 chunks' end takes a 33rd chunk: split_log_of.)
__host__ __device__ inline int win_chunk(int base0) {
    int k = 128;
    while (k * 32 < base0 + DRAFT_ROWS) k *= 2;
    return k;
}
constexpr int HID = 4096;
constexpr int VOCAB = 131072;
constexpr int TOPN = 9;        // routed experts a token uses
constexpr int CAND = 64;       // sampling candidates kept per row

// Q8_0, planar: q [N][K] int8, d [N][K/32] half.
struct Q8W { const int8_t* q; const void* d; int n, k; };
// ModelOpt NVFP4: q [N][K/2] (low nibble first), sc [N][K/16] e4m3, one fp32 tensor scale.
struct NV4W { const uint8_t* q; const uint8_t* sc; float s2; int n, k; };
// ggml IQ2_XXS, planar: qs [N][K/256][64] bytes, d [N][K/256] half.
struct IQ2W { const uint8_t* qs; const void* d; int n, k; };
// Row-major bf16, used by the router and the drafter.
struct BF16W { const uint16_t* w; int n, k; };

// All experts of one layer share a slab: expert e starts at qs + e * qs_stride, d + e * d_stride (elements).
struct IQ2Slab { const uint8_t* qs; const void* d; size_t qs_stride, d_stride; };

// Tile-major weights for the tensor-core GEMMs: rows come in tiles of 16, laid out so that each lane of a warp
// reads the A fragments of m16n8k16 with one 16-byte load. n and k are the logical sizes; the tile count is
// ceil(n / 16) and missing rows are zero. See bolt_mma.cu for the exact byte layouts.
struct TNV4 { const uint4* q; const uint2* sc; float s2; int n, k; };   // NVFP4
// NVFP4 in the fragment order of the block-scaled fp4 MMA (m16n8k64, kind::mxf4nvf4): per 16-row tile and 64-k step,
// lane l holds 4 words (rows g, g+8 at k 8t.. and 32+8t..), and 16 scale words (4 e4m3 bytes a row, one a k-block of 16)
struct TNV4W { const uint4* q; const uint32_t* sf; float s2; int n, k; };
struct TQ8  { const uint4* q; const uint32_t* d; int n, k; };           // Q8_0
struct TBF  { const uint4* w; int n, k; };                              // bf16
struct TIQ2 { const uint4* q; const uint16_t* d; int n, k; };           // IQ2_XXS

// Device-visible values that change from launch to launch, so a captured graph can be replayed.
// A block is a list of segments: consecutive rows of one sequence (a prompt chunk is one segment, a shared verify
// round has one window per sequence). Row r belongs to the segment with seg_row0[i] <= r < seg_row0[i + 1].
struct Ctl {
    int tokens[TMAX];
    int nrows;                  // rows in this block
    int nseg;
    int seg_row0[SEG_MAX + 1];  // first row of each segment; [nseg] == nrows
    int seg_base[SEG_MAX];      // the sequence's cache length before the segment
    int seg_seq[SEG_MAX];       // which sequence's state the segment uses
    int seg_ssm[SEG_MAX];       // its SSM slot to read (0 or 1; the other one is written; always 0 with a lazy state)
    int seg_conv[SEG_MAX];      // its conv ring slot to read
    // Lazy SSM state (BOLT_SSM_LAZY=1): a sequence keeps one state slot and folds the kept rows of its last window into
    // it at the start of its next block's scan, from the scan inputs the conv saved per sequence (two slots of `win`
    // rows: seg_hist is the slot holding the rows to fold, the block's own rows go to the other one).
    int seg_replay[SEG_MAX];    // rows of the last window to fold in first (0 without a lazy state)
    int seg_hist[SEG_MAX];      // the saved-input slot they are in
    // A verify window's attention runs in query blocks of WIN_Q rows (one CTA each): block q covers the rows from
    // qb_off[q] of segment qb_seg[q], and seg_qb0[i] is segment i's first block. A window of up to WIN_Q rows is one block.
    int win;                    // rows a window carries at most in this model: its conv ring has win + 1 slots
    int nqb;
    int qb_seg[QB_MAX];
    int qb_off[QB_MAX];
    int seg_qb0[SEG_MAX];
};

struct Q8Multi {
    Q8W w[3];
    float* y[3];
    int ldy[3];
    int rows[3];  // rows of each matrix
};

// GEMV launchers. x is [T][ldx] fp32, y is [T][ldy] fp32. epi: 0 store, 1 store relu(v)^2.
void gemv_q8(cudaStream_t s, const Q8W& w, const float* x, int ldx, float* y, int ldy, int T, int epi = 0);
void gemv_q8_multi(cudaStream_t s, const Q8Multi& m, const float* x, int ldx, int T);
void gemv_nv4(cudaStream_t s, const NV4W& w, const float* x, int ldx, float* y, int ldy, int T);
void gemv_iq2(cudaStream_t s, const IQ2W& w, const float* x, int ldx, float* y, int ldy, int T, int epi = 0);
void gemv_bf16(cudaStream_t s, const BF16W& w, const float* x, int ldx, float* y, int ldy, int T);

// Tensor-core GEMMs for T rows (1..TMAX): x is [T][ldx] fp32, y is [T][ldy] fp32. epi 1: relu(v)^2.
// Programmatic dependent launch (BOLT_PDL, default on): every kernel of the forward waits at entry for its predecessor's
// writes (griddepcontrol.wait, a no-op when launched plainly) and the launches carry the attribute, so a kernel's launch
// and scheduling overlap the tail of the one before it instead of following the gap between them.
__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}
// a kernel's blocks let the dependent grid be scheduled once their own work is done (the dependent still waits for
// this grid's completion and memory before it reads, so this only moves scheduling, never data)
__device__ __forceinline__ void pdl_trigger() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}
bool pdl_on();
template <typename... KArgs, typename... Args>
inline void launch_pdl(void (*kern)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t s, Args... args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid; cfg.blockDim = block; cfg.dynamicSmemBytes = smem; cfg.stream = s;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = pdl_on() ? 1 : 0;
    cfg.attrs = at; cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, kern, static_cast<KArgs>(args)...);
}

void mma_decode_rows(bool on);
void prompt_min_rows_set(int rows);   // rows from which the prompt-chunk GEMM kernels apply (256; the perplexity harness lowers it)
// a scratch of `elems` halves for the GEMMs launched on stream `s` to pack their rows into (BOLT_PACK=0 leaves the rows fp32)
void mma_pack_buffer(cudaStream_t s, uint16_t* buf, size_t elems);   // the GEMMs of the blocks enqueued next are decode or verify rows (summation order independent of the row count)
void mma_nv4(cudaStream_t s, const TNV4& w, const float* x, int ldx, float* y, int ldy, int T);
// BOLT_NV4_FP4: prompt chunks' NVFP4 projections on the block-scaled fp4 tensor cores, the rows quantized to NVFP4
// (e2m1 with an e4m3 scale a block of 16) first; q4/sf are the scratch for them ([T][k/2] and [T][k/16] bytes)
void tile_nv4w(cudaStream_t s, const uint8_t* q, const uint8_t* sc, void* qw, void* sfw, int n, int k);
void quant_nv4_rows(cudaStream_t s, const float* x, int ldx, uint8_t* q4, uint8_t* sf, int T, int k);
void mma_nv4_fp4(cudaStream_t s, const TNV4W& w, const uint8_t* q4, const uint8_t* sf, float* y, int ldy, int T);
void mma_q8(cudaStream_t s, const TQ8& w, const float* x, int ldx, float* y, int ldy, int T);
void mma_bf16(cudaStream_t s, const TBF& w, const float* x, int ldx, float* y, int ldy, int T);
void mma_iq2(cudaStream_t s, const TIQ2& w, const float* x, int ldx, float* y, int ldy, int T, int epi = 0);
struct TQ8Multi { TQ8 w[3]; float* y[3]; int ldy[3]; };
void mma_q8_multi(cudaStream_t s, const TQ8Multi& m, const float* x, int ldx, int T);

// Load-time repacking into the tile layouts. Sources are the checkpoint's own layouts; the destination
// buffers must hold ceil(n/16) tiles. Sizes in bytes: see tile_bytes_*.
size_t tile_bytes_nv4(int n, int k);
size_t tile_scale_bytes_nv4(int n, int k);
size_t tile_bytes_q8(int n, int k);
size_t tile_scale_bytes_q8(int n, int k);
size_t tile_bytes_bf16(int n, int k);
size_t tile_bytes_iq2(int n, int k);
size_t tile_scale_bytes_iq2(int n, int k);
void tile_nv4(cudaStream_t s, const uint8_t* q, const uint8_t* sc, void* qt, void* sct, int n, int k);
void tile_q8(cudaStream_t s, const uint8_t* raw, void* qt, void* dt, int n, int k);
void tile_bf16(cudaStream_t s, const uint16_t* w, void* wt, int n, int k);
void tile_iq2(cudaStream_t s, const uint8_t* raw, void* qt, void* dt, int n, int k);

// Expert kernels: one launch covers every (token, expert) pair of the block.
// Work items of the routed experts: pairs grouped by expert, at most 16 pairs per item (see bolt_kernels.cu).
struct MoeWork { int *order, *first, *count, *expert, *n_work; };
void k_moe_group(cudaStream_t s, const int* ridx, int pairs, int* order, int* work_first, int* work_n, int* work_exp, int* n_work, int* stat);
void k_quant_rows(cudaStream_t s, const float* x, int ldx, int8_t* q, int ldq, float* sc, int ldsc, int rows, int k);
// One launch covers every item. xq/xs are the quantized inputs: up reads one row per token, down one per pair.
void moe_up(cudaStream_t s, const IQ2Slab& up, const MoeWork& w, const int8_t* xq, const float* xs, float* act, int width, int pairs);
void moe_down(cudaStream_t s, const IQ2Slab& down, const MoeWork& w, const int8_t* xq, const float* xs, float* out, int width, int pairs);
// BOLT_MOE_BF=1: the same on bf16 tensor cores from the fp32 rows (x: a row per token for up, per pair for down)
// The shared expert of a prompt chunk on the int8 tile kernel as one expert over all T rows (int8 rows per 32 instead
// of bf16 ones, as the routed experts have always had); BOLT_SHARED_TILE=0 keeps the bf16 path
void moe_tile_dense(cudaStream_t s, const IQ2Slab& w, const MoeWork& mw, int T, const int8_t* xq, const float* xs, float* y, int ldy, int n, int k, int epi);
void moe_coop_dense(cudaStream_t s, const IQ2Slab& w, const MoeWork& mw, int T, const int8_t* xq, const float* xs, float* y, int ldy, int n, int k, int epi);
bool moe_bf16();
void moe_up_bf(cudaStream_t s, const IQ2Slab& up, const MoeWork& w, const float* x, int ldx, float* act, int width, int pairs);
void moe_down_bf(cudaStream_t s, const IQ2Slab& down, const MoeWork& w, const float* act, int ldx, float* out, int width, int pairs);

// Repack and format helpers used at load time.
void repack_q8(cudaStream_t s, const uint8_t* src, int8_t* q, void* d, size_t rows, int k);
void repack_iq2(cudaStream_t s, const uint8_t* src, uint8_t* qs, void* d, size_t rows, int k);

// Block building blocks.
// Rows whose token is negative come from row t of `extra` (bf16 [T][HID]): media embeddings.
void k_embed(cudaStream_t s, const uint16_t* table, const uint16_t* extra, const Ctl* ctl, float* x, int T);
void k_add_rms(cudaStream_t s, float* x, const float* r1, const float* r2, const float* w, float* xn, float* cap,
               int cap_stride, float eps, int T);
// Conv state: pool + seq * (win + 1) * CONV_ELEMS + slot * CONV_ELEMS. A record block writes the state after every row
// into the next slots of the ring (so a window can be cut anywhere); otherwise only the state after the segment, one on.
// hist_x / hist_dt: a record block saves each row's scan inputs there, by block row; with `lazy` they are per-sequence
// pools ([seq][2][win] rows) and the row goes to slot seg_hist ^ 1 of its sequence, at its index in the segment.
void k_conv(cudaStream_t s, const Ctl* ctl, float* conv_pool, const float* w, const float* b,
            const float* proj, int ldp, float* xbc, int T, int record, float* hist_x, float* hist_dt, int nseg, int lazy);
// SSM state: pool + seq * 2 * SSM_ELEMS + slot * SSM_ELEMS; the segment reads its slot and writes the other one.
// With `lazy` a sequence has one slot (pool + seq * SSM_ELEMS): the scan first folds seg_replay rows of hist slot
// seg_hist into it (and writes it back there, the committed prefix), then runs the block's rows from that state; a
// record block writes nothing more (its rows stay pending until keep), any other block writes its final state in place.
void k_scan(cudaStream_t s, const Ctl* ctl, float* ssm_pool, const float* xbc,
            const float* proj, int ldp, const float* A_log, const float* dt_bias, const float* D, float* y, int T, int record, int nseg,
            int lazy, const float* hist_x, const float* hist_dt);
void k_gate_rms(cudaStream_t s, const float* y, const float* proj, int ldp, const float* w, float* out, float eps, int T);
void k_route(cudaStream_t s, const float* logits, int ldl, const float* bias, int* ridx, float* rw, int E, float scale, int T);
void k_combine(cudaStream_t s, const float* o, const float* rw, float* out, int T);
// The attention cache of one layer: K and V rows [seq][ctx][2 heads][128]. bf16, or (fp8) e4m3 bytes with one fp32 scale
// per (token, head) in ks/vs [seq][ctx][2]; a value is the byte times its scale. ctx is the per-sequence capacity.
struct KvView { void* k; void* v; float* ks; float* vs; int ctx; int fp8; };
void k_kv_store(cudaStream_t s, const Ctl* ctl, const float* k, const float* v, const KvView& kv, int T);
// multi: rows of several sequences (or a verify window): every row runs on its own. Otherwise one prompt segment.
void k_attn(cudaStream_t s, const Ctl* ctl, const float* q, const KvView& kv, float* part, float* out,
            int T, int splits, int multi, int nseg_unused = 0);
// Row-wise top candidates of the logits: vals/ids are [T][CAND] sorted descending (ties to the lower id).
void k_candidates(cudaStream_t s, const float* logits, int T, int topk, float* vals, int* ids, float* scratch_v, int* scratch_i);
// The same for the n rows sel[i] among the tokens allowed by masks + i * words (structured output); a candidate list shorter
// than topk (fewer allowed tokens) ends in entries of value -3.4e38.
void k_candidates_masked(cudaStream_t s, const float* logits, const int* sel, const uint32_t* masks, int words, int n, int topk,
                         float* vals, int* ids, float* scratch_v, int* scratch_i);

// Draft-side kernels reuse the same primitives; see bolt_draft.cu.

// Verify windows keep only the final SSM state; the scan inputs of each row are saved so a shorter accepted
// prefix can be replayed from the pre-window state (one launch over all Mamba layers).
struct ReplayLayer { float *ssm, *A_log, *dt_bias, *hist_x, *hist_dt; };
// Rebuild the SSM state after the first n rows of the segment whose rows start at row0 of the last block: read slot
// `slot` of sequence `seq`, write the other slot.
void k_scan_replay(cudaStream_t s, const ReplayLayer* layers, int nlayers, int seq, int slot, int row0, int n);
// Lazy state: fold the first n saved rows of hist slot `hslot` of the sequence into its one state slot, in place.
void k_scan_fold(cudaStream_t s, const ReplayLayer* layers, int nlayers, int seq, int hslot, int n, int win);
// Prompt attention on tensor cores (bolt_mma.cu); part must hold attn_mma_part_floats(T, splits) floats.
void k_attn_mma(cudaStream_t s, const Ctl* ctl, const float* q, const KvView& kv, float* part, float* out, int T, int splits);
size_t attn_mma_part_floats(int T, int splits);
// The same for verify windows: one CTA per query block of a sequence's window (WIN_Q rows; ctl->nqb blocks), nseg a launch
// bound on the sequences, part holds QB_MAX * 2 * splits * 16640 floats.
void k_attn_mma_windows(cudaStream_t s, const Ctl* ctl, const float* q, const KvView& kv, float* part, float* out,
                        int T, int nseg, int splits);

// Sizes of one sequence's recurrent state (floats per Mamba layer).
constexpr int SSM_ELEMS = 128 * 64 * 128;
constexpr int CONV_ELEMS = 10240 * 3;
