// Bolt kernels. Weights stay packed (NVFP4, Q8_0, IQ2_XXS) and are dequantized in registers while a warp
// streams its rows once; activations are fp32. Tables for IQ2_XXS are ggml's (MIT).

#include "bolt_common.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdio>
#include <cstring>

#define GGML_TABLE_BEGIN(type, name, size) static __device__ type name[size] = {
#define GGML_TABLE_END() };
#include "iq2_tables.inc"

static __device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}

static __device__ __forceinline__ float silu(float v) { return v / (1.f + __expf(-v)); }

// ---------------------------------------------------------------------------------------------------------
// Format policies. A chunk is 32 consecutive weights of one row; a lane owns every 32nd chunk.
// load() fetches the raw bytes of a chunk, deq() turns 8 of its weights (sub 0..3) into scaled floats.
// ---------------------------------------------------------------------------------------------------------

struct FmtQ8 {
    using W = Q8W;
    struct Chunk { uint4 a, b; float d; };
    static constexpr bool kGrid = false;
    static __device__ __forceinline__ Chunk load(const W& w, int row, int c) {
        Chunk ch;
        const uint4* p = reinterpret_cast<const uint4*>(w.q + (size_t)row * w.k + (size_t)c * 32);
        ch.a = __ldg(p);
        ch.b = __ldg(p + 1);
        ch.d = __half2float(__ldg(reinterpret_cast<const __half*>(w.d) + (size_t)row * (w.k >> 5) + c));
        return ch;
    }
    static __device__ __forceinline__ void deq(const Chunk& ch, int sub, float* out, const void*) {
        uint2 v;
        if (sub == 0) v = make_uint2(ch.a.x, ch.a.y);
        else if (sub == 1) v = make_uint2(ch.a.z, ch.a.w);
        else if (sub == 2) v = make_uint2(ch.b.x, ch.b.y);
        else v = make_uint2(ch.b.z, ch.b.w);
#pragma unroll
        for (int j = 0; j < 4; j++) {
            out[j] = (float)(int8_t)((v.x >> (8 * j)) & 0xff) * ch.d;
            out[4 + j] = (float)(int8_t)((v.y >> (8 * j)) & 0xff) * ch.d;
        }
    }
};

struct FmtNV4 {
    using W = NV4W;
    struct Chunk { uint4 q; unsigned sc; float s2; };
    static constexpr bool kGrid = false;
    static __device__ __forceinline__ Chunk load(const W& w, int row, int c) {
        Chunk ch;
        ch.q = __ldg(reinterpret_cast<const uint4*>(w.q + (size_t)row * (w.k >> 1) + (size_t)c * 16));
        ch.sc = __ldg(reinterpret_cast<const unsigned short*>(w.sc + (size_t)row * (w.k >> 4) + (size_t)c * 2));
        ch.s2 = w.s2;
        return ch;
    }
    static __device__ __forceinline__ float e4m3(unsigned b) {
        __nv_fp8_e4m3 f;
        f.__x = (unsigned char)b;
        return (float)f;
    }
    static __device__ __forceinline__ void deq(const Chunk& ch, int sub, float* out, const void*) {
        unsigned word = sub == 0 ? ch.q.x : sub == 1 ? ch.q.y : sub == 2 ? ch.q.z : ch.q.w;
        float s = e4m3((ch.sc >> (8 * (sub >> 1))) & 0xff) * ch.s2;
        // e2m1 -> float: the three magnitude bits land in the exponent's low bits and the mantissa MSB,
        // and 2^126 puts the exponent back (denormal e=0 gives the 0.5 step).
        const float fix = 8.507059173023462e37f * 1.f;  // 2^126
#pragma unroll
        for (int j = 0; j < 8; j++) {
            unsigned nib = (word >> (4 * j)) & 0xf;
            unsigned bits = ((nib & 7u) << 22) | ((nib & 8u) << 28);
            out[j] = __uint_as_float(bits) * fix * s;
        }
    }
};

struct FmtBF16 {
    using W = BF16W;
    struct Chunk { uint4 a, b, c, d; };
    static constexpr bool kGrid = false;
    static __device__ __forceinline__ Chunk load(const W& w, int row, int c) {
        Chunk ch;
        const uint4* p = reinterpret_cast<const uint4*>(w.w + (size_t)row * w.k + (size_t)c * 32);
        ch.a = __ldg(p); ch.b = __ldg(p + 1); ch.c = __ldg(p + 2); ch.d = __ldg(p + 3);
        return ch;
    }
    static __device__ __forceinline__ void deq(const Chunk& ch, int sub, float* out, const void*) {
        uint4 v = sub == 0 ? ch.a : sub == 1 ? ch.b : sub == 2 ? ch.c : ch.d;
        out[0] = __uint_as_float(v.x << 16); out[1] = __uint_as_float(v.x & 0xffff0000u);
        out[2] = __uint_as_float(v.y << 16); out[3] = __uint_as_float(v.y & 0xffff0000u);
        out[4] = __uint_as_float(v.z << 16); out[5] = __uint_as_float(v.z & 0xffff0000u);
        out[6] = __uint_as_float(v.w << 16); out[7] = __uint_as_float(v.w & 0xffff0000u);
    }
};

// IQ2_XXS: a chunk is one 8-byte group (four grid indices, four 7-bit sign indices, a 4-bit scale).
struct FmtIQ2 {
    using W = IQ2W;
    struct Chunk { uint2 q; float db; };
    static constexpr bool kGrid = true;
    static __device__ __forceinline__ Chunk load(const W& w, int row, int c) {
        Chunk ch;
        int nb = w.k >> 8;
        const uint2* p = reinterpret_cast<const uint2*>(w.qs + ((size_t)row * nb) * 64) + c;
        ch.q = __ldg(p);
        float d = __half2float(__ldg(reinterpret_cast<const __half*>(w.d) + (size_t)row * nb + (c >> 3)));
        ch.db = d * (0.5f + (float)(ch.q.y >> 28)) * 0.25f;
        return ch;
    }
    static __device__ __forceinline__ void deq(const Chunk& ch, int sub, float* out, const void* grid_smem) {
        const uint2* grid = reinterpret_cast<const uint2*>(grid_smem);
        unsigned gi = (ch.q.x >> (8 * sub)) & 0xff;
        uint2 g = grid[gi];
        unsigned si = (ch.q.y >> (7 * sub)) & 127u;
        unsigned sg = si | ((__popc(si) & 1u) << 7);
#pragma unroll
        for (int j = 0; j < 8; j++) {
            unsigned byte = ((j < 4 ? g.x : g.y) >> (8 * (j & 3))) & 0xffu;
            float v = __uint_as_float(0x4B000000u | byte) - 8388608.f;
            unsigned flip = (sg << (31 - j)) & 0x80000000u;
            out[j] = __uint_as_float(__float_as_uint(v) ^ flip) * ch.db;
        }
    }
};

// ---------------------------------------------------------------------------------------------------------
// The shared GEMV body: T activation rows, R weight rows per warp, four warps per block.
// ---------------------------------------------------------------------------------------------------------

template <class F, int T, int R, int EPI>
static __device__ __forceinline__ void gemv_body(const typename F::W& w, const float* __restrict__ x, int ldx,
                                                 float* __restrict__ y, int ldy, int N, int rows_base, int Tn) {
    __shared__ uint2 grid_s[F::kGrid ? 256 : 1];
    if constexpr (F::kGrid) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) {
            unsigned long long g = iq2xxs_grid[i];
            grid_s[i] = make_uint2((unsigned)g, (unsigned)(g >> 32));
        }
        __syncthreads();
    }
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = rows_base + warp * R;
    if (row0 >= N) return;
    const int nchunk = w.k >> 5;
    float acc[R][T];
#pragma unroll
    for (int r = 0; r < R; r++)
#pragma unroll
        for (int t = 0; t < T; t++) acc[r][t] = 0.f;

    typename F::Chunk cur[R], nxt[R];
    int c = lane;
    if (c < nchunk) {
#pragma unroll
        for (int r = 0; r < R; r++) cur[r] = F::load(w, min(row0 + r, N - 1), c);
    }
    for (; c < nchunk; c += 32) {
        const int cn = c + 32;
        if (cn < nchunk) {
#pragma unroll
            for (int r = 0; r < R; r++) nxt[r] = F::load(w, min(row0 + r, N - 1), cn);
        }
#pragma unroll
        for (int sub = 0; sub < 4; sub++) {
            float wv[R][8];
#pragma unroll
            for (int r = 0; r < R; r++) F::deq(cur[r], sub, wv[r], grid_s);
#pragma unroll
            for (int t = 0; t < T; t++) {
                const float4* xp = reinterpret_cast<const float4*>(x + (size_t)t * ldx + (size_t)c * 32 + sub * 8);
                float4 xa = __ldg(xp), xb = __ldg(xp + 1);
#pragma unroll
                for (int r = 0; r < R; r++) {
                    acc[r][t] += wv[r][0] * xa.x + wv[r][1] * xa.y + wv[r][2] * xa.z + wv[r][3] * xa.w +
                                 wv[r][4] * xb.x + wv[r][5] * xb.y + wv[r][6] * xb.z + wv[r][7] * xb.w;
                }
            }
        }
        if (cn < nchunk) {
#pragma unroll
            for (int r = 0; r < R; r++) cur[r] = nxt[r];
        }
    }
#pragma unroll
    for (int r = 0; r < R; r++) {
#pragma unroll
        for (int t = 0; t < T; t++) {
            float v = warp_sum(acc[r][t]);
            if (lane == ((r * T + t) & 31) && row0 + r < N && t < Tn) {
                if (EPI == 1) v = v > 0.f ? v * v : 0.f;
                y[(size_t)t * ldy + row0 + r] = v;
            }
        }
    }
}

template <class F, int T, int R, int EPI>
__global__ void __launch_bounds__(128) gemv_k(typename F::W w, const float* __restrict__ x, int ldx, float* __restrict__ y, int ldy, int Tn) {
    pdl_wait();
    gemv_body<F, T, R, EPI>(w, x, ldx, y, ldy, w.n, blockIdx.x * 4 * R, Tn);
    pdl_trigger();
}

template <class F, int T, int R, int EPI>
static void launch_gemv(cudaStream_t s, const typename F::W& w, const float* x, int ldx, float* y, int ldy, int Tn) {
    int rows_per_cta = 4 * R;
    launch_pdl(gemv_k<F, T, R, EPI>, (w.n + rows_per_cta - 1) / rows_per_cta, 128, 0, s, w, x, ldx, y, ldy, Tn);
}

template <class F, int EPI>
static void dispatch_gemv(cudaStream_t s, const typename F::W& w, const float* x, int ldx, float* y, int ldy, int T) {
    if (T <= 1) launch_gemv<F, 1, 4, EPI>(s, w, x, ldx, y, ldy, T);
    else if (T == 2) launch_gemv<F, 2, 4, EPI>(s, w, x, ldx, y, ldy, T);
    else if (T <= 4) launch_gemv<F, 4, 4, EPI>(s, w, x, ldx, y, ldy, T);
    else if (T <= 8) launch_gemv<F, 8, 2, EPI>(s, w, x, ldx, y, ldy, T);
    else launch_gemv<F, 16, 2, EPI>(s, w, x, ldx, y, ldy, T);
}

void gemv_q8(cudaStream_t s, const Q8W& w, const float* x, int ldx, float* y, int ldy, int T, int epi) {
    if (epi) dispatch_gemv<FmtQ8, 1>(s, w, x, ldx, y, ldy, T);
    else dispatch_gemv<FmtQ8, 0>(s, w, x, ldx, y, ldy, T);
}
void gemv_nv4(cudaStream_t s, const NV4W& w, const float* x, int ldx, float* y, int ldy, int T) {
    dispatch_gemv<FmtNV4, 0>(s, w, x, ldx, y, ldy, T);
}
void gemv_iq2(cudaStream_t s, const IQ2W& w, const float* x, int ldx, float* y, int ldy, int T, int epi) {
    if (epi) dispatch_gemv<FmtIQ2, 1>(s, w, x, ldx, y, ldy, T);
    else dispatch_gemv<FmtIQ2, 0>(s, w, x, ldx, y, ldy, T);
}
void gemv_bf16(cudaStream_t s, const BF16W& w, const float* x, int ldx, float* y, int ldy, int T) {
    dispatch_gemv<FmtBF16, 0>(s, w, x, ldx, y, ldy, T);
}

// q, k and v in one launch: a block belongs to one matrix.
template <int T, int R>
__global__ void __launch_bounds__(128) q8_multi_k(Q8Multi m, const float* __restrict__ x, int ldx, int Tn) {
    pdl_wait();
    int rows_per_cta = 4 * R;
    int b = blockIdx.x * rows_per_cta;
    int which = 0;
    while (which < 2 && b >= m.rows[which]) { b -= m.rows[which]; which++; }
    gemv_body<FmtQ8, T, R, 0>(m.w[which], x, ldx, m.y[which], m.ldy[which], m.w[which].n, b, Tn);
    pdl_trigger();
}

void gemv_q8_multi(cudaStream_t s, const Q8Multi& m, const float* x, int ldx, int T) {
    int total = m.rows[0] + m.rows[1] + m.rows[2];
    auto grid = [&](int rpc) { return (total + rpc - 1) / rpc; };
    if (T <= 1) launch_pdl(q8_multi_k<1, 4>, grid(16), 128, 0, s, m, x, ldx, T);
    else if (T == 2) launch_pdl(q8_multi_k<2, 4>, grid(16), 128, 0, s, m, x, ldx, T);
    else if (T <= 4) launch_pdl(q8_multi_k<4, 4>, grid(16), 128, 0, s, m, x, ldx, T);
    else if (T <= 8) launch_pdl(q8_multi_k<8, 2>, grid(8), 128, 0, s, m, x, ldx, T);
    else launch_pdl(q8_multi_k<16, 2>, grid(8), 128, 0, s, m, x, ldx, T);
}

// One launch for every (token, expert) pair. Pair order[blockIdx.y] belongs to token pair / TOPN and uses expert
// ridx[pair]. The activation vector is quantized to int8 per 32 (llama.cpp's q8_1 scheme), so a group of
// eight weights is two dp4a; the grid magnitudes get their signs from ggml's ksigns64 byte masks.
// Routed experts on int8 tensor cores. The (token, expert) pairs are grouped by expert and cut into work items of at
// most MOE_B pairs. Activations are quantized to int8 per 32 (a row's scale is amax / 127); an item's tokens sit in
// shared memory as the B operand while each warp walks a 16-row tile of the expert's rows chunk by chunk (32
// weights): the IQ2_XXS grid and sign tables turn the chunk into an int8 A fragment of m16n8k32, one mma gives the
// tile's 16 rows x up to 16 tokens for the chunk, and the chunk's scales are applied to the int32 result.
constexpr int MOE_B = 64;   // most pairs of a work item; the grouping cuts at the launch's item size (8, 16, 32, or 64 for the tile kernel)

__global__ void moe_group_k(const int* __restrict__ ridx, int pairs, int item_b, int* __restrict__ order, int* __restrict__ work_first,
                            int* __restrict__ work_n, int* __restrict__ work_exp, int* __restrict__ n_work, int* __restrict__ stat) {
    pdl_wait();
    __shared__ int cnt[512], start[512], fill[512], act[512], sstart[512], slen[512], sexp[512], nbat[512];
    const int tid = threadIdx.x;  // 512 threads
    cnt[tid] = 0; fill[tid] = 0;
    __syncthreads();
    for (int p = tid; p < pairs; p += 512) atomicAdd(&cnt[ridx[p]], 1);
    __syncthreads();
    start[tid] = cnt[tid];
    act[tid] = cnt[tid] > 0;
    __syncthreads();
    for (int off = 1; off < 512; off <<= 1) {
        int a = tid >= off ? start[tid - off] : 0, b = tid >= off ? act[tid - off] : 0;
        __syncthreads();
        start[tid] += a; act[tid] += b;
        __syncthreads();
    }
    const int pos0 = start[tid] - cnt[tid];
    const int slot = act[tid] - (cnt[tid] > 0);
    const int nact = act[511];
    sstart[tid] = 0; slen[tid] = 0; sexp[tid] = 0;
    __syncthreads();
    if (cnt[tid] > 0) { sstart[slot] = pos0; slen[slot] = cnt[tid]; sexp[slot] = tid; }
    __syncthreads();
    nbat[tid] = tid < nact ? (slen[tid] + item_b - 1) / item_b : 0;
    __syncthreads();
    const int mine = nbat[tid];
    for (int off = 1; off < 512; off <<= 1) {
        int a = tid >= off ? nbat[tid - off] : 0;
        __syncthreads();
        nbat[tid] += a;
        __syncthreads();
    }
    if (tid < nact) {
        int w0 = nbat[tid] - mine;
        for (int i = 0; i < mine; i++) {
            work_first[w0 + i] = sstart[tid] + i * item_b;
            work_n[w0 + i] = min(item_b, slen[tid] - i * item_b);
            work_exp[w0 + i] = sexp[tid];
        }
    }
    if (tid == 511) { n_work[0] = nbat[511]; if (stat) { atomicAdd(stat, nact); atomicAdd(stat + 1, nbat[511]); } }
    start[tid] = pos0;
    __syncthreads();
    for (int p = tid; p < pairs; p += 512) {
        int e = ridx[p];
        order[start[e] + atomicAdd(&fill[e], 1)] = p;
    }
    pdl_trigger();
}
// Pairs a work item holds for a launch of `pairs` pairs.
bool pdl_on() { static const char* e = getenv("BOLT_PDL"); static const bool on = !(e && atoi(e) == 0); return on; }
bool moe_tile();
static int moe_tile_pairs() { static const char* e = getenv("BOLT_MOE_TILE_PAIRS"); static const int v = e ? atoi(e) : 4096; return v; }   // the harness lowers it
constexpr int MT_BM = 128, MT_BN = 64, MT_BK = 64, MT_STR = MT_BK + 16, MT_STAGES = 3, MT_ITEM = 64;   // the prompt tile kernel's shape
int moe_item_pairs(int pairs) {
    static const char* e = getenv("BOLT_MOE_B");    // prompt chunks (over 4096 pairs)
    static const char* ed = getenv("BOLT_MOE_BD");  // decode rounds: 8, 16 or 32 pairs an item (an expert's weights are read once an item)
    int v = pairs <= 8 * 9 ? 8 : 16;
    if (pairs > 4096 && moe_tile()) v = MT_ITEM;   // prompt chunks on the tile kernel (the threshold is BOLT_MOE_TILE_PAIRS)
    else if (pairs <= 4096 && moe_tile() && pairs > moe_tile_pairs()) v = MT_ITEM;
    else if (e && pairs > 4096) v = atoi(e);
    else if (ed && pairs <= 4096) { const int d = atoi(ed); if (d == 8 || d == 16 || d == 32) v = d; }
    if (v == 32 && moe_bf16()) v = 16;   // the bf16 kernel stages at most 16 rows (shared memory)
    return v;
}
void k_moe_group(cudaStream_t s, const int* ridx, int pairs, int* order, int* work_first, int* work_n, int* work_exp, int* n_work, int* stat) {
    launch_pdl(moe_group_k, 1, 512, 0, s, ridx, pairs, moe_item_pairs(pairs), order, work_first, work_n, work_exp, n_work, stat);
}

// int8 rows with one scale per 32: q[r][k], sc[r][k/32] = amax / 127
__global__ void quant_rows_k(const float* __restrict__ x, int ldx, int8_t* __restrict__ q, int ldq, float* __restrict__ sc, int ldsc, int k) {
    pdl_wait();
    const int r = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int b = warp; b < (k >> 5); b += blockDim.x >> 5) {
        float v = x[(size_t)r * ldx + b * 32 + lane];
        float am = fabsf(v);
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, m));
        float inv = am > 0.f ? 127.f / am : 0.f;
        q[(size_t)r * ldq + b * 32 + lane] = (int8_t)__float2int_rn(v * inv);
        if (lane == 0) sc[(size_t)r * ldsc + b] = am / 127.f;
    }
    pdl_trigger();
}
void k_quant_rows(cudaStream_t s, const float* x, int ldx, int8_t* q, int ldq, float* sc, int ldsc, int rows, int k) {
    launch_pdl(quant_rows_k, rows, 256, 0, s, x, ldx, q, ldq, sc, ldsc, k);
}

static __device__ __forceinline__ void mma_s8(int* d, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

constexpr int MOE_SMEM_K = 2560;

template <int NT, int EPI, bool WHOLE>
__global__ void __launch_bounds__(256) moe_mma_k(IQ2Slab slab, const int* __restrict__ order, const int* __restrict__ work_first,
                                                 const int* __restrict__ work_n, const int* __restrict__ work_exp,
                                                 const int* __restrict__ n_work, const int8_t* __restrict__ xq,
                                                 const float* __restrict__ xs, int x_per_pair, float* __restrict__ y, int ldy,
                                                 int n, int k) {
    pdl_wait();
    if ((int)blockIdx.y >= n_work[0]) return;
    extern __shared__ __align__(16) unsigned char smem[];
    const int stride = k + 16;                       // bytes per staged token row (bank-conflict free fragment loads)
    int8_t* xsm = reinterpret_cast<int8_t*>(smem);   // [NT*8][stride]
    float* ssm = reinterpret_cast<float*>(smem + (size_t)NT * 8 * stride);  // [NT*8][k/32]
    __shared__ unsigned gx[256], gy[256], sx[128], sy[128];   // half entries (k 0-3 and 4-7 of a grid entry)
    __shared__ uint2 g2[256], s2[128];                          // whole entries, for WHOLE
    __shared__ int pid[MOE_B];
    const int nb = k >> 8, nchunk = k >> 5;
    const int e = work_exp[blockIdx.y], first = work_first[blockIdx.y], cnt = work_n[blockIdx.y];
    const int tid = threadIdx.x;
    for (int i = tid; i < 256; i += 256) {
        unsigned long long g = iq2xxs_grid[i];
        gx[i] = (unsigned)g; gy[i] = (unsigned)(g >> 32);
        g2[i] = make_uint2((unsigned)g, (unsigned)(g >> 32));
    }
    if (tid < 128) { unsigned long long m = ksigns64[tid]; sx[tid] = (unsigned)m; sy[tid] = (unsigned)(m >> 32); s2[tid] = make_uint2((unsigned)m, (unsigned)(m >> 32)); }
    if (tid < MOE_B) pid[tid] = tid < cnt ? order[first + tid] : -1;
    __syncthreads();
    // stage the item's tokens, all (row, 16-byte piece) pairs at once so the loads overlap; the columns past cnt hold
    // whatever the shared memory had and are never stored
    {
        const int per = k >> 4;   // 16-byte pieces a row
#pragma unroll 4
        for (int idx = tid; idx < cnt * per; idx += 256) {
            const int r = idx / per, i = idx - r * per;
            const int p = pid[r];
            const int row = x_per_pair ? p : p / TOPN;
            *reinterpret_cast<uint4*>(xsm + (size_t)r * stride + i * 16) = __ldg(reinterpret_cast<const uint4*>(xq + (size_t)row * k) + i);
        }
        for (int idx = tid; idx < cnt * nchunk; idx += 256) {
            const int r = idx / nchunk, i = idx - r * nchunk;
            const int p = pid[r];
            const int row = x_per_pair ? p : p / TOPN;
            ssm[r * nchunk + i] = __ldg(xs + (size_t)row * nchunk + i);
        }
    }
    __syncthreads();

    const int warp = tid >> 5, lane = tid & 31, gi = lane >> 2, ti = lane & 3;
    for (int tile = blockIdx.x * 8 + warp; tile * 16 < n; tile += gridDim.x * 8) {
        const int rowA = min(tile * 16 + gi, n - 1), rowB = min(tile * 16 + gi + 8, n - 1);
        const uint8_t* qbase = slab.qs + (size_t)e * slab.qs_stride;
        const __half* dbase = reinterpret_cast<const __half*>(slab.d) + (size_t)e * slab.d_stride;
        const uint4* qa_p = reinterpret_cast<const uint4*>(qbase + (size_t)rowA * nb * 64);   // 16 bytes = two chunks
        const uint4* qb_p = reinterpret_cast<const uint4*>(qbase + (size_t)rowB * nb * 64);
        const __half* da_p = dbase + (size_t)rowA * nb;
        const __half* db_p = dbase + (size_t)rowB * nb;
        const int l0 = ti >> 1, l2 = l0 + 2;
        const unsigned* gt = (ti & 1) ? gy : gx;
        const unsigned* mt = (ti & 1) ? sy : sx;
        float acc[NT][4];
    #pragma unroll
        for (int i = 0; i < NT; i++) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;

        // four chunks (one 32-byte sector of each row) per step, the next step's loads in flight
        uint4 a0 = __ldg(qa_p), a1 = __ldg(qa_p + 1), b0 = __ldg(qb_p), b1 = __ldg(qb_p + 1);
        for (int c4 = 0; c4 < nchunk; c4 += 4) {
            const uint4 A0 = a0, A1 = a1, B0 = b0, B1 = b1;
            if (c4 + 4 < nchunk) {
                const int u = (c4 + 4) >> 1;
                a0 = __ldg(qa_p + u); a1 = __ldg(qa_p + u + 1); b0 = __ldg(qb_p + u); b1 = __ldg(qb_p + u + 1);
            }
            const float dhA = __half2float(__ldg(da_p + (c4 >> 3))), dhB = __half2float(__ldg(db_p + (c4 >> 3)));
    #pragma unroll
            for (int cc = 0; cc < 4; cc++) {
                const int c = c4 + cc;
                const uint2 ca = cc == 0 ? make_uint2(A0.x, A0.y) : cc == 1 ? make_uint2(A0.z, A0.w) : cc == 2 ? make_uint2(A1.x, A1.y) : make_uint2(A1.z, A1.w);
                const uint2 cb = cc == 0 ? make_uint2(B0.x, B0.y) : cc == 1 ? make_uint2(B0.z, B0.w) : cc == 2 ? make_uint2(B1.x, B1.y) : make_uint2(B1.z, B1.w);
                const float dA = dhA * (0.5f + (float)(ca.y >> 28)) * 0.25f;
                const float dB = dhB * (0.5f + (float)(cb.y >> 28)) * 0.25f;
                unsigned q0, q1, q2, q3;
                if constexpr (WHOLE) {
                    // lane t takes grid entry t of the chunk (k 8t..8t+7) for its MMA slots 4t..4t+3 and 16+4t..16+4t+3;
                    // the activations are read at the same real k, so the 32-term dot product is the same integer
                    const uint2 ga = g2[(ca.x >> (8 * ti)) & 255], ma = s2[(ca.y >> (7 * ti)) & 127];
                    const uint2 gb = g2[(cb.x >> (8 * ti)) & 255], mb = s2[(cb.y >> (7 * ti)) & 127];
                    q0 = __vsub4(ga.x ^ ma.x, ma.x); q2 = __vsub4(ga.y ^ ma.y, ma.y);
                    q1 = __vsub4(gb.x ^ mb.x, mb.x); q3 = __vsub4(gb.y ^ mb.y, mb.y);
                } else {
                    unsigned ga0 = gt[(ca.x >> (8 * l0)) & 255], ma0 = mt[(ca.y >> (7 * l0)) & 127];
                    unsigned ga2 = gt[(ca.x >> (8 * l2)) & 255], ma2 = mt[(ca.y >> (7 * l2)) & 127];
                    unsigned gb0 = gt[(cb.x >> (8 * l0)) & 255], mb0 = mt[(cb.y >> (7 * l0)) & 127];
                    unsigned gb2 = gt[(cb.x >> (8 * l2)) & 255], mb2 = mt[(cb.y >> (7 * l2)) & 127];
                    q0 = __vsub4(ga0 ^ ma0, ma0); q1 = __vsub4(gb0 ^ mb0, mb0);
                    q2 = __vsub4(ga2 ^ ma2, ma2); q3 = __vsub4(gb2 ^ mb2, mb2);
                }
    #pragma unroll
                for (int nt = 0; nt < NT; nt++) {
                    int dd[4] = {0, 0, 0, 0};
                    if constexpr (WHOLE) {
                        const uint2 bb = *reinterpret_cast<const uint2*>(xsm + (size_t)(nt * 8 + gi) * stride + c * 32 + 8 * ti);
                        mma_s8(dd, q0, q1, q2, q3, bb.x, bb.y);
                    } else {
                        const int8_t* xr = xsm + (size_t)(nt * 8 + gi) * stride + c * 32 + 4 * ti;
                        mma_s8(dd, q0, q1, q2, q3, *reinterpret_cast<const unsigned*>(xr), *reinterpret_cast<const unsigned*>(xr + 16));
                    }
                    const float s0 = ssm[(nt * 8 + 2 * ti) * nchunk + c], s1 = ssm[(nt * 8 + 2 * ti + 1) * nchunk + c];
                    acc[nt][0] += dA * s0 * (float)dd[0];
                    acc[nt][1] += dA * s1 * (float)dd[1];
                    acc[nt][2] += dB * s0 * (float)dd[2];
                    acc[nt][3] += dB * s1 * (float)dd[3];
                }
            }
        }
    #pragma unroll
        for (int nt = 0; nt < NT; nt++)
    #pragma unroll
            for (int j = 0; j < 4; j++) {
                const int tok = nt * 8 + 2 * ti + (j & 1);
                const int row = tile * 16 + gi + ((j >> 1) ? 8 : 0);
                if (tok < cnt && row < n) {
                    float v = acc[nt][j];
                    if (EPI == 1) v = v > 0.f ? v * v : 0.f;
                    y[(size_t)pid[tok] * ldy + row] = v;
                }
            }
    }
    pdl_trigger();
}

static size_t moe_smem(int nt, int k) { return (size_t)nt * 8 * (k + 16) + (size_t)nt * 8 * (k >> 5) * 4; }

template <int NT, int EPI, bool WHOLE>
static void launch_moe_mma(cudaStream_t s, const IQ2Slab& slab, const MoeWork& w, const int8_t* xq, const float* xs, int x_per_pair,
                           float* y, int ldy, int n, int k, int items) {
    static bool opted = false;
    size_t sm = moe_smem(NT, MOE_SMEM_K);
    if (!opted) {
        cudaFuncSetAttribute(moe_mma_k<NT, EPI, WHOLE>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
        opted = true;
    }
    // each CTA stages its item's tokens once and walks several weight tiles per warp; items past n_work exit at once, so
    // the launch only has to cover the most items `items` pairs can make
    static const char* ec = getenv("BOLT_MOE_CTAS");
    const int ctas = ec ? atoi(ec) : 4;
    items = std::min(items, items / (NT * 8) + 512);
    dim3 grid(std::min((n / 16 + 7) / 8, ctas), items);
    launch_pdl(moe_mma_k<NT, EPI, WHOLE>, grid, 256, moe_smem(NT, k), s, slab, w.order, w.first, w.count, w.expert, w.n_work, xq, xs, x_per_pair, y, ldy, n, k);
}
// cp.async and ldmatrix, for the prefetching kernels below
__device__ __forceinline__ void k_cp_async16(void* smem, const void* gmem) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(a), "l"(gmem));
}
__device__ __forceinline__ void k_cp_async8(void* smem, const void* gmem) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(a), "l"(gmem));
}
__device__ __forceinline__ void k_cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void k_cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void k_ldsm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}

// Decode rounds, deep prefetch (BOLT_MOE_COOP, default): the same items, tiles and arithmetic as moe_mma_k with whole
// entries, but a warp streams its two rows' weights through a per-warp cp.async ring, 64 bytes a row a step and two
// steps ahead, so about 2 KB a warp is in flight instead of 128 bytes; the token rows are staged as before.
constexpr int MC_STAGES = 3, MC_CH = 8;   // chunks (of 32 k) a step: 64 bytes a row

template <int NT, int EPI>
__global__ void __launch_bounds__(256) moe_coop_k(IQ2Slab slab, const int* __restrict__ order, const int* __restrict__ work_first,
                                                  const int* __restrict__ work_n, const int* __restrict__ work_exp,
                                                  const int* __restrict__ n_work, const int8_t* __restrict__ xq,
                                                  const float* __restrict__ xs, int x_per_pair, float* __restrict__ y, int ldy,
                                                  int n, int k) {
    pdl_wait();
    if ((int)blockIdx.y >= n_work[0]) return;
    extern __shared__ __align__(16) unsigned char smem[];
    const int stride = k + 16;
    int8_t* xsm = reinterpret_cast<int8_t*>(smem);                                   // [NT*8][stride]
    float* ssm = reinterpret_cast<float*>(smem + (size_t)NT * 8 * stride);           // [NT*8][k/32]
    uint4* ring = reinterpret_cast<uint4*>(smem + (size_t)NT * 8 * stride + (size_t)NT * 8 * (k >> 5) * 4);   // [STAGES][8 warps][16 rows][4]
    __shared__ uint2 g2[256], s2[128];
    __shared__ int pid[MOE_B];
    const int nb = k >> 8, nchunk = k >> 5, nstep = nchunk / MC_CH;
    const int e = work_exp[blockIdx.y], first = work_first[blockIdx.y], cnt = work_n[blockIdx.y];
    const int tid = threadIdx.x;
    for (int i = tid; i < 256; i += 256) { const unsigned long long g = iq2xxs_grid[i]; g2[i] = make_uint2((unsigned)g, (unsigned)(g >> 32)); }
    if (tid < 128) { const unsigned long long m = ksigns64[tid]; s2[tid] = make_uint2((unsigned)m, (unsigned)(m >> 32)); }
    if (tid < MOE_B) pid[tid] = tid < cnt ? order[first + tid] : -1;
    __syncthreads();
    {
        const int per = k >> 4;
#pragma unroll 4
        for (int idx = tid; idx < cnt * per; idx += 256) {
            const int r = idx / per, i = idx - r * per;
            const int p = pid[r];
            const int row = x_per_pair ? p : p / TOPN;
            *reinterpret_cast<uint4*>(xsm + (size_t)r * stride + i * 16) = __ldg(reinterpret_cast<const uint4*>(xq + (size_t)row * k) + i);
        }
        for (int idx = tid; idx < cnt * nchunk; idx += 256) {
            const int r = idx / nchunk, i = idx - r * nchunk;
            const int p = pid[r];
            const int row = x_per_pair ? p : p / TOPN;
            ssm[r * nchunk + i] = __ldg(xs + (size_t)row * nchunk + i);
        }
    }
    __syncthreads();
    const int warp = tid >> 5, lane = tid & 31, gi = lane >> 2, ti = lane & 3;
    const uint8_t* qbase = slab.qs + (size_t)e * slab.qs_stride;
    const __half* dbase = reinterpret_cast<const __half*>(slab.d) + (size_t)e * slab.d_stride;
    uint4* wring = ring + (size_t)warp * 16 * 4;   // this warp's slot of a stage: [16 rows][4 pieces]
    const int tiles_per = (gridDim.x * 8 > 0) ? ((n / 16 - (blockIdx.x * 8 + warp) + gridDim.x * 8 - 1) / (gridDim.x * 8)) : 0;   // tiles this warp walks
    const int total = tiles_per * nstep;
    auto issue = [&](int idx) {   // (tile, step) idx of this warp's walk: 64 bytes of each of its two rows
        const int tl = idx / nstep, st = idx - tl * nstep;
        const int tile = blockIdx.x * 8 + warp + tl * gridDim.x * 8;
        const int rowA = min(tile * 16 + gi, n - 1), rowB = min(tile * 16 + gi + 8, n - 1);
        uint4* dst = wring + (size_t)(idx % MC_STAGES) * 8 * 16 * 4;
        k_cp_async16(dst + gi * 4 + ti, qbase + (size_t)rowA * nb * 64 + (size_t)st * 64 + ti * 16);
        k_cp_async16(dst + (gi + 8) * 4 + ti, qbase + (size_t)rowB * nb * 64 + (size_t)st * 64 + ti * 16);
        k_cp_commit();
    };
    if (total > 0) issue(0);
    if (total > 1) issue(1);
    float acc[NT][4];
    for (int idx = 0; idx < total; idx++) {
        const int tl = idx / nstep, st = idx - tl * nstep;
        const int tile = blockIdx.x * 8 + warp + tl * gridDim.x * 8;
        const int rowA = min(tile * 16 + gi, n - 1), rowB = min(tile * 16 + gi + 8, n - 1);
        if (st == 0) {
#pragma unroll
            for (int i = 0; i < NT; i++) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
        }
        if (idx + 1 < total) k_cp_wait<1>(); else k_cp_wait<0>();
        __syncwarp();
        const uint4* src = wring + (size_t)(idx % MC_STAGES) * 8 * 16 * 4;
        const float dhA = __half2float(__ldg(dbase + (size_t)rowA * nb + (st * MC_CH >> 3)));
        const float dhB = __half2float(__ldg(dbase + (size_t)rowB * nb + (st * MC_CH >> 3)));
#pragma unroll
        for (int cc = 0; cc < MC_CH; cc++) {
            const int c = st * MC_CH + cc;
            const uint4 pa = src[gi * 4 + (cc >> 1)], pb = src[(gi + 8) * 4 + (cc >> 1)];
            const uint2 ca = (cc & 1) ? make_uint2(pa.z, pa.w) : make_uint2(pa.x, pa.y);
            const uint2 cb = (cc & 1) ? make_uint2(pb.z, pb.w) : make_uint2(pb.x, pb.y);
            const float dA = dhA * (0.5f + (float)(ca.y >> 28)) * 0.25f;
            const float dB = dhB * (0.5f + (float)(cb.y >> 28)) * 0.25f;
            const uint2 ga = g2[(ca.x >> (8 * ti)) & 255], ma = s2[(ca.y >> (7 * ti)) & 127];
            const uint2 gb = g2[(cb.x >> (8 * ti)) & 255], mb = s2[(cb.y >> (7 * ti)) & 127];
            const unsigned q0 = __vsub4(ga.x ^ ma.x, ma.x), q2 = __vsub4(ga.y ^ ma.y, ma.y);
            const unsigned q1 = __vsub4(gb.x ^ mb.x, mb.x), q3 = __vsub4(gb.y ^ mb.y, mb.y);
#pragma unroll
            for (int nt = 0; nt < NT; nt++) {
                int dd[4] = {0, 0, 0, 0};
                const uint2 bb = *reinterpret_cast<const uint2*>(xsm + (size_t)(nt * 8 + gi) * stride + c * 32 + 8 * ti);
                mma_s8(dd, q0, q1, q2, q3, bb.x, bb.y);
                const float s0 = ssm[(nt * 8 + 2 * ti) * nchunk + c], s1 = ssm[(nt * 8 + 2 * ti + 1) * nchunk + c];
                acc[nt][0] += dA * s0 * (float)dd[0];
                acc[nt][1] += dA * s1 * (float)dd[1];
                acc[nt][2] += dB * s0 * (float)dd[2];
                acc[nt][3] += dB * s1 * (float)dd[3];
            }
        }
        __syncwarp();   // every lane is done with this stage before it is refilled
        if (idx + 2 < total) issue(idx + 2);
        if (st == nstep - 1) {
#pragma unroll
            for (int nt = 0; nt < NT; nt++)
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const int tok = nt * 8 + 2 * ti + (j & 1);
                    const int row = tile * 16 + gi + ((j >> 1) ? 8 : 0);
                    if (tok < cnt && row < n) {
                        float v = acc[nt][j];
                        if (EPI == 1) v = v > 0.f ? v * v : 0.f;
                        y[(size_t)pid[tok] * ldy + row] = v;
                    }
                }
        }
    }
    pdl_trigger();
}

static size_t moe_coop_smem(int nt, int k) { return (size_t)nt * 8 * (k + 16) + (size_t)nt * 8 * (k >> 5) * 4 + (size_t)MC_STAGES * 8 * 16 * 64; }
template <int NT, int EPI>
static void launch_moe_coop(cudaStream_t s, const IQ2Slab& slab, const MoeWork& w, const int8_t* xq, const float* xs, int x_per_pair,
                            float* y, int ldy, int n, int k, int items, int ctas_all = 0) {
    static size_t opted = 0;   // the dynamic shared memory allowed so far (the shared expert's down projection has k 5376)
    if (moe_coop_smem(NT, k) > opted) { opted = std::max(moe_coop_smem(NT, k), moe_coop_smem(NT, MOE_SMEM_K)); cudaFuncSetAttribute(moe_coop_k<NT, EPI>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)opted); }
    static const char* ec = getenv("BOLT_MOE_CTAS");
    const int ctas = ctas_all ? (n / 16 + 7) / 8 : (ec ? atoi(ec) : 4);   // a single expert over a block: every tile group at once
    items = std::min(items, items / (NT * 8) + 512);
    dim3 grid(std::min((n / 16 + 7) / 8, ctas), items);
    launch_pdl(moe_coop_k<NT, EPI>, grid, 256, moe_coop_smem(NT, k), s, slab, w.order, w.first, w.count, w.expert, w.n_work, xq, xs, x_per_pair, y, ldy, n, k);
}
static bool moe_coop() { static const char* e = getenv("BOLT_MOE_COOP"); return !(e && atoi(e) == 0); }

// BOLT_MOE_HALF=1 keeps the half-entry gathers (eight 4-byte table reads a lane and chunk); whole entries need four 8-byte ones
static bool moe_half() { static const char* e = getenv("BOLT_MOE_HALF"); return e && atoi(e) != 0; }
template <int NT, int EPI>
static void launch_moe(cudaStream_t s, const IQ2Slab& slab, const MoeWork& w, const int8_t* xq, const float* xs, int x_per_pair,
                       float* y, int ldy, int n, int k, int items) {
    if (moe_half()) launch_moe_mma<NT, EPI, false>(s, slab, w, xq, xs, x_per_pair, y, ldy, n, k, items);
    else if (moe_coop() && NT <= 2 && items > TOPN) launch_moe_coop<NT, EPI>(s, slab, w, xq, xs, x_per_pair, y, ldy, n, k, items);   // a lone row's nine experts run faster without the ring
    else launch_moe_mma<NT, EPI, true>(s, slab, w, xq, xs, x_per_pair, y, ldy, n, k, items);
}

void moe_up_tile(cudaStream_t s, const IQ2Slab& up, const MoeWork& w, const int8_t* xq, const float* xs, float* act, int width, int pairs);
void moe_down_tile(cudaStream_t s, const IQ2Slab& down, const MoeWork& w, const int8_t* xq, const float* xs, float* out, int width, int pairs);
void moe_up(cudaStream_t s, const IQ2Slab& up, const MoeWork& w, const int8_t* xq, const float* xs, float* act, int width, int pairs) {
    const int b = moe_item_pairs(pairs);
    if (b == MT_ITEM) { moe_up_tile(s, up, w, xq, xs, act, width, pairs); return; }
    if (b == 8) launch_moe<1, 1>(s, up, w, xq, xs, 0, act, width, width, 1024, pairs);
    else if (b == 16) launch_moe<2, 1>(s, up, w, xq, xs, 0, act, width, width, 1024, pairs);
    else launch_moe<4, 1>(s, up, w, xq, xs, 0, act, width, width, 1024, pairs);
}
void moe_down(cudaStream_t s, const IQ2Slab& down, const MoeWork& w, const int8_t* xq, const float* xs, float* out, int width, int pairs) {
    const int b = moe_item_pairs(pairs);
    if (b == MT_ITEM) { moe_down_tile(s, down, w, xq, xs, out, width, pairs); return; }
    if (b == 8) launch_moe<1, 0>(s, down, w, xq, xs, 1, out, 1024, 1024, width, pairs);
    else if (b == 16) launch_moe<2, 0>(s, down, w, xq, xs, 1, out, 1024, 1024, width, pairs);
    else launch_moe<4, 0>(s, down, w, xq, xs, 1, out, 1024, 1024, width, pairs);
}

// ---------------------------------------------------------------------------------------------------------
// Routed experts for prompt chunks (BOLT_MOE_TILE, items of 64 pairs): a CTA of 8 warps owns one expert's 128
// weight rows x 64 of its tokens. Each K step of 64, every warp decodes one 16-row tile to int8 in shared memory
// (whole grid entries, as moe_mma_k) with the chunk scales beside it; the tokens' int8 rows and scales arrive by
// cp.async in a three-stage ring; both operands reach the int8 tensor cores through ldmatrix. The epilogue is the
// same expression in the same chunk order as moe_mma_k (acc += (dA * s) * dd), so the outputs are identical.
// ---------------------------------------------------------------------------------------------------------



template <int EPI>
__global__ void __launch_bounds__(256, 2) moe_tile_k(IQ2Slab slab, const int* __restrict__ order, const int* __restrict__ work_first,
                                                     const int* __restrict__ work_n, const int* __restrict__ work_exp,
                                                     const int* __restrict__ n_work, const int8_t* __restrict__ xq,
                                                     const float* __restrict__ xs, int x_per_pair, float* __restrict__ y, int ldy,
                                                     int n, int k) {
    pdl_wait();
    if ((int)blockIdx.y >= n_work[0]) return;
    __shared__ __align__(16) int8_t wa[2][MT_BM * MT_STR];      // the step's int8 weights [row][k]
    __shared__ float ws[2][MT_BM][2];                            // their chunk scales
    __shared__ __align__(16) int8_t xb[MT_STAGES][MT_BN * MT_STR]; // the tokens' int8 rows
    __shared__ __align__(16) float xsb[MT_STAGES][MT_BN][2];      // their chunk scales
    __shared__ __align__(16) uint4 wq[MT_STAGES][MT_BM];           // the step's raw weight pieces, 16 B a row
    __shared__ uint2 g2[256], s2[128];
    __shared__ int pid[MT_ITEM];
    const int nb = k >> 8, nchunk = k >> 5, steps = k >> 6;
    const int e = work_exp[blockIdx.y], first = work_first[blockIdx.y], cnt = work_n[blockIdx.y];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gi = lane >> 2, ti = lane & 3;
    const int wr = warp >> 1, wt = warp & 1;    // 32 rows and 32 tokens a warp
    for (int i = tid; i < 256; i += 256) { const unsigned long long g = iq2xxs_grid[i]; g2[i] = make_uint2((unsigned)g, (unsigned)(g >> 32)); }
    if (tid < 128) { const unsigned long long m = ksigns64[tid]; s2[tid] = make_uint2((unsigned)m, (unsigned)(m >> 32)); }
    if (tid < MT_ITEM) pid[tid] = tid < cnt ? order[first + tid] : -1;
    __syncthreads();
    const int row_base = blockIdx.x * MT_BM;
    const int myrow0 = min(row_base + warp * 16 + gi, n - 1), myrow1 = min(row_base + warp * 16 + gi + 8, n - 1);
    const uint8_t* qbase = slab.qs + (size_t)e * slab.qs_stride;
    const __half* dbase = reinterpret_cast<const __half*>(slab.d) + (size_t)e * slab.d_stride;
    const __half* da_p = dbase + (size_t)myrow0 * nb;
    const __half* db_p = dbase + (size_t)myrow1 * nb;
    auto issue_w = [&](int stage, int s) {   // the 128 rows' 16 bytes of step s (threads 0..127)
        if (tid < MT_BM) {
            const int grow = min(row_base + tid, n - 1);
            k_cp_async16(&wq[stage][tid], qbase + (size_t)grow * nb * 64 + (size_t)s * 16);
        }
    };
    auto dequant = [&](int s, int buf) {
        const uint4 ra = wq[s % MT_STAGES][warp * 16 + gi], rb = wq[s % MT_STAGES][warp * 16 + gi + 8];
        const float rda = __half2float(__ldg(da_p + (s >> 2))), rdb = __half2float(__ldg(db_p + (s >> 2)));
        struct { uint4 a, b; float da, db; } r{ra, rb, rda, rdb};
        int8_t* wrow = wa[buf] + (warp * 16 + gi) * MT_STR;
#pragma unroll
        for (int cc = 0; cc < 2; cc++) {
            const uint2 ca = cc ? make_uint2(r.a.z, r.a.w) : make_uint2(r.a.x, r.a.y);
            const uint2 cb = cc ? make_uint2(r.b.z, r.b.w) : make_uint2(r.b.x, r.b.y);
            const uint2 ga = g2[(ca.x >> (8 * ti)) & 255], ma = s2[(ca.y >> (7 * ti)) & 127];
            const uint2 gb = g2[(cb.x >> (8 * ti)) & 255], mb = s2[(cb.y >> (7 * ti)) & 127];
            const uint2 qa = make_uint2(__vsub4(ga.x ^ ma.x, ma.x), __vsub4(ga.y ^ ma.y, ma.y));
            const uint2 qb = make_uint2(__vsub4(gb.x ^ mb.x, mb.x), __vsub4(gb.y ^ mb.y, mb.y));
            *reinterpret_cast<uint2*>(wrow + cc * 32 + 8 * ti) = qa;
            *reinterpret_cast<uint2*>(wrow + 8 * MT_STR + cc * 32 + 8 * ti) = qb;
            if (ti == 0) {
                ws[buf][warp * 16 + gi][cc] = r.da * (0.5f + (float)(ca.y >> 28)) * 0.25f;
                ws[buf][warp * 16 + gi + 8][cc] = r.db * (0.5f + (float)(cb.y >> 28)) * 0.25f;
            }
        }
    };
    auto issue = [&](int stage, int s) {   // 64 tokens x 64 int8 and their two scales
        for (int i = tid; i < MT_BN * 4; i += 256) {
            const int r = i >> 2, p = (i & 3) * 16;
            int8_t* dst = xb[stage] + r * MT_STR + p;
            if (r < cnt) {
                const int pp = pid[r];
                const int row = x_per_pair ? pp : pp / TOPN;
                k_cp_async16(dst, xq + (size_t)row * k + s * MT_BK + p);
            } else *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
        }
        if (tid < MT_BN) {
            if (tid < cnt) {
                const int pp = pid[tid];
                const int row = x_per_pair ? pp : pp / TOPN;
                k_cp_async8(&xsb[stage][tid][0], xs + (size_t)row * nchunk + s * 2);
            } else { xsb[stage][tid][0] = 0.f; xsb[stage][tid][1] = 0.f; }
        }
        k_cp_commit();
    };
    float acc[2][4][4];
#pragma unroll
    for (int r = 0; r < 2; r++)
#pragma unroll
        for (int t4 = 0; t4 < 4; t4++) { acc[r][t4][0] = acc[r][t4][1] = acc[r][t4][2] = acc[r][t4][3] = 0.f; }
    const int lm = lane >> 3, lr = lane & 7;
    // token-tile pairs this warp has tokens for (an item rarely fills all 64 slots): the MMAs of empty ones are skipped
    const int ntp = min(2, (cnt - wt * 32 + 15) >> 4);
    // the tokens and the weight pieces of two steps in flight; a step waits for its own, decodes its weights
    // (two barriers: the copies, then the decoded tile), issues step s + 2 and runs its MMAs
    issue_w(0, 0); issue(0, 0);
    if (steps > 1) { issue_w(1, 1); issue(1, 1); }
    for (int s = 0; s < steps; s++) {
        if (s + 1 < steps) k_cp_wait<1>(); else k_cp_wait<0>();
        __syncthreads();
        dequant(s, s & 1);
        __syncthreads();
        if (s + 2 < steps) { issue_w((s + 2) % MT_STAGES, s + 2); issue((s + 2) % MT_STAGES, s + 2); }
        const int8_t* wbuf = wa[s & 1];
        const int8_t* xbuf = xb[s % MT_STAGES];
#pragma unroll
        for (int cc = 0; cc < 2; cc++) {   // the step's two chunks of 32 k
            if (ntp <= 0) break;
            uint32_t a[2][4];
#pragma unroll
            for (int rt = 0; rt < 2; rt++) {
                const int row = wr * 32 + rt * 16 + lr + (lm & 1) * 8;
                k_ldsm_x4(a[rt][0], a[rt][1], a[rt][2], a[rt][3], wbuf + row * MT_STR + cc * 32 + (lm >> 1) * 16);
            }
            const float dA0 = ws[s & 1][wr * 32 + gi][cc], dA1 = ws[s & 1][wr * 32 + gi + 8][cc];
            const float dB0 = ws[s & 1][wr * 32 + 16 + gi][cc], dB1 = ws[s & 1][wr * 32 + 16 + gi + 8][cc];
#pragma unroll
            for (int tp = 0; tp < 2; tp++) {   // two token tiles an ldmatrix
                if (tp >= ntp) break;
                uint32_t b0, b1, b2, b3;
                const int tok = wt * 32 + tp * 16 + lr + (lm >> 1) * 8;
                k_ldsm_x4(b0, b1, b2, b3, xbuf + tok * MT_STR + cc * 32 + (lm & 1) * 16);
                const int t0 = wt * 32 + tp * 16 + 2 * ti, t1 = t0 + 8;
                const float s00 = xsb[s % MT_STAGES][t0][cc], s01 = xsb[s % MT_STAGES][t0 + 1][cc];
                const float s10 = xsb[s % MT_STAGES][t1][cc], s11 = xsb[s % MT_STAGES][t1 + 1][cc];
                int d0[4] = {0, 0, 0, 0}, d1[4] = {0, 0, 0, 0}, d2[4] = {0, 0, 0, 0}, d3[4] = {0, 0, 0, 0};
                mma_s8(d0, a[0][0], a[0][1], a[0][2], a[0][3], b0, b1);
                mma_s8(d1, a[0][0], a[0][1], a[0][2], a[0][3], b2, b3);
                mma_s8(d2, a[1][0], a[1][1], a[1][2], a[1][3], b0, b1);
                mma_s8(d3, a[1][0], a[1][1], a[1][2], a[1][3], b2, b3);
                acc[0][2 * tp][0] += dA0 * s00 * (float)d0[0]; acc[0][2 * tp][1] += dA0 * s01 * (float)d0[1];
                acc[0][2 * tp][2] += dA1 * s00 * (float)d0[2]; acc[0][2 * tp][3] += dA1 * s01 * (float)d0[3];
                acc[0][2 * tp + 1][0] += dA0 * s10 * (float)d1[0]; acc[0][2 * tp + 1][1] += dA0 * s11 * (float)d1[1];
                acc[0][2 * tp + 1][2] += dA1 * s10 * (float)d1[2]; acc[0][2 * tp + 1][3] += dA1 * s11 * (float)d1[3];
                acc[1][2 * tp][0] += dB0 * s00 * (float)d2[0]; acc[1][2 * tp][1] += dB0 * s01 * (float)d2[1];
                acc[1][2 * tp][2] += dB1 * s00 * (float)d2[2]; acc[1][2 * tp][3] += dB1 * s01 * (float)d2[3];
                acc[1][2 * tp + 1][0] += dB0 * s10 * (float)d3[0]; acc[1][2 * tp + 1][1] += dB0 * s11 * (float)d3[1];
                acc[1][2 * tp + 1][2] += dB1 * s10 * (float)d3[2]; acc[1][2 * tp + 1][3] += dB1 * s11 * (float)d3[3];
            }
        }
    }
#pragma unroll
    for (int rt = 0; rt < 2; rt++)
#pragma unroll
        for (int t4 = 0; t4 < 4; t4++)
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int tok = wt * 32 + t4 * 8 + 2 * ti + (j & 1);
                const int row = row_base + wr * 32 + rt * 16 + gi + ((j >> 1) ? 8 : 0);
                if (tok < cnt && row < n) {
                    float v = acc[rt][t4][j];
                    if (EPI == 1) v = v > 0.f ? v * v : 0.f;
                    y[(size_t)pid[tok] * ldy + row] = v;
                }
            }
    pdl_trigger();
}

bool moe_tile() { static const char* e = getenv("BOLT_MOE_TILE"); return !(e && atoi(e) == 0); }
template <int EPI>
static void launch_moe_tile(cudaStream_t s, const IQ2Slab& slab, const MoeWork& w, const int8_t* xq, const float* xs, int x_per_pair,
                            float* y, int ldy, int n, int k, int items) {
    items = std::min(items, items / MT_ITEM + 512);
    dim3 grid((n + MT_BM - 1) / MT_BM, items);
    launch_pdl(moe_tile_k<EPI>, grid, 256, 0, s, slab, w.order, w.first, w.count, w.expert, w.n_work, xq, xs, x_per_pair, y, ldy, n, k);
}
// One tensor over all of a chunk's rows (the shared expert as a single "expert"): order is the identity, items of 64 rows
__global__ void shared_items_k(int T, int* __restrict__ order, int* __restrict__ first, int* __restrict__ count, int* __restrict__ exp, int* __restrict__ nwork) {
    pdl_wait();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < T) order[i] = i;
    const int items = (T + MT_ITEM - 1) / MT_ITEM;
    if (i < items) { first[i] = i * MT_ITEM; count[i] = min(MT_ITEM, T - i * MT_ITEM); exp[i] = 0; }
    if (i == 0) nwork[0] = items;
    pdl_trigger();
}
// the same for verify blocks (T rows, items of 8 or 16): the deep-prefetch kernel as a single expert
__global__ void shared_items_small_k(int T, int item, int* __restrict__ order, int* __restrict__ first, int* __restrict__ count, int* __restrict__ exp, int* __restrict__ nwork) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < T) order[i] = i;
    const int items = (T + item - 1) / item;
    if (i < items) { first[i] = i * item; count[i] = min(item, T - i * item); exp[i] = 0; }
    if (i == 0) nwork[0] = items;
}
void moe_coop_dense(cudaStream_t s, const IQ2Slab& w, const MoeWork& mw, int T, const int8_t* xq, const float* xs, float* y, int ldy, int n, int k, int epi) {
    const int item = (T <= 8 || moe_coop_smem(2, k) > 96 * 1024) ? 8 : 16;   // two token tiles do not fit shared memory at k 5376
    shared_items_small_k<<<(T + 255) / 256, 256, 0, s>>>(T, item, mw.order, mw.first, mw.count, mw.expert, mw.n_work);
    const int items = (T + item - 1) / item;
    if (item == 8) { if (epi == 1) launch_moe_coop<1, 1>(s, w, mw, xq, xs, 1, y, ldy, n, k, items, 1); else launch_moe_coop<1, 0>(s, w, mw, xq, xs, 1, y, ldy, n, k, items, 1); }
    else { if (epi == 1) launch_moe_coop<2, 1>(s, w, mw, xq, xs, 1, y, ldy, n, k, items, 1); else launch_moe_coop<2, 0>(s, w, mw, xq, xs, 1, y, ldy, n, k, items, 1); }
}
void moe_tile_dense(cudaStream_t s, const IQ2Slab& w, const MoeWork& mw, int T, const int8_t* xq, const float* xs, float* y, int ldy, int n, int k, int epi) {
    launch_pdl(shared_items_k, (T + 255) / 256, 256, 0, s, T, mw.order, mw.first, mw.count, mw.expert, mw.n_work);
    const int items = (T + MT_ITEM - 1) / MT_ITEM;
    if (epi == 1) launch_moe_tile<1>(s, w, mw, xq, xs, 1, y, ldy, n, k, items);
    else launch_moe_tile<0>(s, w, mw, xq, xs, 1, y, ldy, n, k, items);
}
void moe_up_tile(cudaStream_t s, const IQ2Slab& up, const MoeWork& w, const int8_t* xq, const float* xs, float* act, int width, int pairs) {
    launch_moe_tile<1>(s, up, w, xq, xs, 0, act, width, width, 1024, pairs);
}
void moe_down_tile(cudaStream_t s, const IQ2Slab& down, const MoeWork& w, const int8_t* xq, const float* xs, float* out, int width, int pairs) {
    launch_moe_tile<0>(s, down, w, xq, xs, 1, out, 1024, 1024, width, pairs);
}

// ---------------------------------------------------------------------------------------------------------
// Routed experts on bf16 tensor cores (BOLT_MOE_BF=1): the same items and tiles as moe_mma_k, but the chunk scale is
// folded into the bf16 A fragments (as the dense IQ2 path does) and the tensor core accumulates across all of K in
// fp32, so a chunk costs its MMAs and nothing else; the tokens are staged as bf16 rows straight from the fp32 input
// (no int8 quantization pass). Per lane t the tables hold values 2t, 2t+1 of each grid entry as a bf16 pair and
// their sign bits.
// ---------------------------------------------------------------------------------------------------------

__device__ __forceinline__ void mma_bf16_k(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t bmul2u_k(uint32_t a, __nv_bfloat162 s) {
    __nv_bfloat162 v = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), s);
    return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ uint32_t bf16pair(float a, float b) {
    __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<uint32_t*>(&h);
}

template <int NT, int EPI>
__global__ void __launch_bounds__(256) moe_mma_bf_k(IQ2Slab slab, const int* __restrict__ order, const int* __restrict__ work_first,
                                                    const int* __restrict__ work_n, const int* __restrict__ work_exp,
                                                    const int* __restrict__ n_work, const float* __restrict__ x, int ldx,
                                                    int x_per_pair, float* __restrict__ y, int ldy, int n, int k) {
    pdl_wait();
    if ((int)blockIdx.y >= n_work[0]) return;
    extern __shared__ __align__(16) unsigned char smem[];
    const int stride = k + 8;                                   // halves per staged token row
    uint16_t* xsm = reinterpret_cast<uint16_t*>(smem);          // [NT*8][stride] bf16
    __shared__ uint32_t tbl[4][256], msk[4][128];
    __shared__ int pid[MOE_B];
    const int nb = k >> 8, nchunk = k >> 5;
    const int e = work_exp[blockIdx.y], first = work_first[blockIdx.y], cnt = work_n[blockIdx.y];
    const int tid = threadIdx.x;
    for (int i = tid; i < 4 * 256; i += 256) {
        const int t = i >> 8, idx = i & 255;
        const unsigned long long gr = iq2xxs_grid[idx];
        tbl[t][idx] = bf16pair((float)((gr >> (16 * t)) & 0xff), (float)((gr >> (16 * t + 8)) & 0xff));
    }
    for (int i = tid; i < 4 * 128; i += 256) {
        const int t = i >> 7, sidx = i & 127;
        const unsigned sg = sidx | ((__popc(sidx) & 1) << 7);
        msk[t][sidx] = (((sg >> (2 * t)) & 1) ? 0x00008000u : 0u) | (((sg >> (2 * t + 1)) & 1) ? 0x80000000u : 0u);
    }
    if (tid < MOE_B) pid[tid] = tid < cnt ? order[first + tid] : -1;
    __syncthreads();
    {   // stage the item's tokens as bf16, four floats a thread per step
        const int per = k >> 2;
        for (int idx = tid; idx < cnt * per; idx += 256) {
            const int r = idx / per, i = idx - r * per;
            const int p = pid[r];
            const int row = x_per_pair ? p : p / TOPN;
            const float4 v = __ldg(reinterpret_cast<const float4*>(x + (size_t)row * ldx) + i);
            *reinterpret_cast<uint2*>(xsm + (size_t)r * stride + i * 4) = make_uint2(bf16pair(v.x, v.y), bf16pair(v.z, v.w));
        }
    }
    __syncthreads();

    const int warp = tid >> 5, lane = tid & 31, gi = lane >> 2, ti = lane & 3;
    const uint32_t* tb = tbl[ti];
    const uint32_t* mk = msk[ti];
    for (int tile = blockIdx.x * 8 + warp; tile * 16 < n; tile += gridDim.x * 8) {
        const int rowA = min(tile * 16 + gi, n - 1), rowB = min(tile * 16 + gi + 8, n - 1);
        const uint8_t* qbase = slab.qs + (size_t)e * slab.qs_stride;
        const __half* dbase = reinterpret_cast<const __half*>(slab.d) + (size_t)e * slab.d_stride;
        const uint4* qa_p = reinterpret_cast<const uint4*>(qbase + (size_t)rowA * nb * 64);
        const uint4* qb_p = reinterpret_cast<const uint4*>(qbase + (size_t)rowB * nb * 64);
        const __half* da_p = dbase + (size_t)rowA * nb;
        const __half* db_p = dbase + (size_t)rowB * nb;
        float acc[NT][4];
#pragma unroll
        for (int i = 0; i < NT; i++) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
        uint4 a0 = __ldg(qa_p), a1 = __ldg(qa_p + 1), b0 = __ldg(qb_p), b1 = __ldg(qb_p + 1);
        for (int c4 = 0; c4 < nchunk; c4 += 4) {
            const uint4 A0 = a0, A1 = a1, B0 = b0, B1 = b1;
            if (c4 + 4 < nchunk) {
                const int u = (c4 + 4) >> 1;
                a0 = __ldg(qa_p + u); a1 = __ldg(qa_p + u + 1); b0 = __ldg(qb_p + u); b1 = __ldg(qb_p + u + 1);
            }
            const float dhA = __half2float(__ldg(da_p + (c4 >> 3))), dhB = __half2float(__ldg(db_p + (c4 >> 3)));
#pragma unroll
            for (int cc = 0; cc < 4; cc++) {
                const int c = c4 + cc;
                const uint2 ca = cc == 0 ? make_uint2(A0.x, A0.y) : cc == 1 ? make_uint2(A0.z, A0.w) : cc == 2 ? make_uint2(A1.x, A1.y) : make_uint2(A1.z, A1.w);
                const uint2 cb = cc == 0 ? make_uint2(B0.x, B0.y) : cc == 1 ? make_uint2(B0.z, B0.w) : cc == 2 ? make_uint2(B1.x, B1.y) : make_uint2(B1.z, B1.w);
                const __nv_bfloat162 sa = __float2bfloat162_rn(dhA * (0.5f + (float)(ca.y >> 28)) * 0.25f);
                const __nv_bfloat162 sb = __float2bfloat162_rn(dhB * (0.5f + (float)(cb.y >> 28)) * 0.25f);
#pragma unroll
                for (int u = 0; u < 2; u++) {
                    const int l0 = 2 * u, l1 = 2 * u + 1;
                    const uint32_t fa0 = bmul2u_k(tb[(ca.x >> (8 * l0)) & 255] ^ mk[(ca.y >> (7 * l0)) & 127], sa);
                    const uint32_t fa2 = bmul2u_k(tb[(ca.x >> (8 * l1)) & 255] ^ mk[(ca.y >> (7 * l1)) & 127], sa);
                    const uint32_t fa1 = bmul2u_k(tb[(cb.x >> (8 * l0)) & 255] ^ mk[(cb.y >> (7 * l0)) & 127], sb);
                    const uint32_t fa3 = bmul2u_k(tb[(cb.x >> (8 * l1)) & 255] ^ mk[(cb.y >> (7 * l1)) & 127], sb);
                    const int kb = c * 32 + u * 16 + 2 * ti;
#pragma unroll
                    for (int nt = 0; nt < NT; nt++) {
                        const uint16_t* xr = xsm + (size_t)(nt * 8 + gi) * stride + kb;
                        mma_bf16_k(acc[nt], fa0, fa1, fa2, fa3, *reinterpret_cast<const uint32_t*>(xr), *reinterpret_cast<const uint32_t*>(xr + 8));
                    }
                }
            }
        }
#pragma unroll
        for (int nt = 0; nt < NT; nt++)
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int tok = nt * 8 + 2 * ti + (j & 1);
                const int row = tile * 16 + gi + ((j >> 1) ? 8 : 0);
                if (tok < cnt && row < n) {
                    float v = acc[nt][j];
                    if (EPI == 1) v = v > 0.f ? v * v : 0.f;
                    y[(size_t)pid[tok] * ldy + row] = v;
                }
            }
    }
    pdl_trigger();
}

static size_t moe_bf_smem(int nt, int k) { return (size_t)nt * 8 * (k + 8) * 2; }

template <int NT, int EPI>
static void launch_moe_bf(cudaStream_t s, const IQ2Slab& slab, const MoeWork& w, const float* x, int ldx, int x_per_pair,
                          float* y, int ldy, int n, int k, int items) {
    static bool opted = false;
    if (!opted) {
        cudaFuncSetAttribute(moe_mma_bf_k<NT, EPI>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)moe_bf_smem(NT, MOE_SMEM_K));
        opted = true;
    }
    static const char* ec = getenv("BOLT_MOE_CTAS");
    const int ctas = ec ? atoi(ec) : 4;
    items = std::min(items, items / (NT * 8) + 512);
    dim3 grid(std::min((n / 16 + 7) / 8, ctas), items);
    launch_pdl(moe_mma_bf_k<NT, EPI>, grid, 256, moe_bf_smem(NT, k), s, slab, w.order, w.first, w.count, w.expert, w.n_work, x, ldx, x_per_pair, y, ldy, n, k);
}

bool moe_bf16() { static const char* e = getenv("BOLT_MOE_BF"); return e && atoi(e) != 0; }
void moe_up_bf(cudaStream_t s, const IQ2Slab& up, const MoeWork& w, const float* x, int ldx, float* act, int width, int pairs) {
    if (moe_item_pairs(pairs) == 8) launch_moe_bf<1, 1>(s, up, w, x, ldx, 0, act, width, width, 1024, pairs);
    else launch_moe_bf<2, 1>(s, up, w, x, ldx, 0, act, width, width, 1024, pairs);
}
void moe_down_bf(cudaStream_t s, const IQ2Slab& down, const MoeWork& w, const float* act, int ldx, float* out, int width, int pairs) {
    if (moe_item_pairs(pairs) == 8) launch_moe_bf<1, 0>(s, down, w, act, ldx, 1, out, 1024, 1024, width, pairs);
    else launch_moe_bf<2, 0>(s, down, w, act, ldx, 1, out, 1024, 1024, width, pairs);
}

// ---------------------------------------------------------------------------------------------------------
// Load-time repacking of routed experts: ggml's interleaved blocks into planes so every load is aligned.
// ---------------------------------------------------------------------------------------------------------

__global__ void repack_iq2_k(const uint8_t* __restrict__ src, uint8_t* __restrict__ qs, __half* __restrict__ d, size_t nblocks) {
    pdl_wait();
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nblocks) return;
    const uint8_t* b = src + i * 66;
    unsigned short h = (unsigned short)b[0] | ((unsigned short)b[1] << 8);
    d[i] = __ushort_as_half(h);
    unsigned short* o = reinterpret_cast<unsigned short*>(qs + i * 64);
    for (int j = 0; j < 32; j++) o[j] = (unsigned short)b[2 + 2 * j] | ((unsigned short)b[3 + 2 * j] << 8);
    pdl_trigger();
}
void repack_iq2(cudaStream_t s, const uint8_t* src, uint8_t* qs, void* d, size_t rows, int k) {
    size_t nb = rows * (size_t)(k >> 8);
    launch_pdl(repack_iq2_k, (unsigned)((nb + 255) / 256), 256, 0, s, src, qs, (__half*)d, nb);
}
void repack_q8(cudaStream_t, const uint8_t*, int8_t*, void*, size_t, int) {}

// ---------------------------------------------------------------------------------------------------------
// Block building blocks
// ---------------------------------------------------------------------------------------------------------

__global__ void embed_k(const uint16_t* __restrict__ table, const uint16_t* __restrict__ extra, const Ctl* __restrict__ ctl, float* __restrict__ x) {
    pdl_wait();
    int t = blockIdx.x;
    int id = ctl->tokens[t];
    // a negative token takes row t of `extra` (an image or video embedding copied there for this block)
    const uint16_t* row = id < 0 ? extra + (size_t)t * HID : table + (size_t)((unsigned)id >= (unsigned)VOCAB ? 0 : id) * HID;
    const uint2* src = reinterpret_cast<const uint2*>(row);
    float4* dst = reinterpret_cast<float4*>(x + (size_t)t * HID);
    for (int i = threadIdx.x; i < HID / 4; i += blockDim.x) {
        uint2 v = __ldg(src + i);
        dst[i] = make_float4(__uint_as_float(v.x << 16), __uint_as_float(v.x & 0xffff0000u),
                             __uint_as_float(v.y << 16), __uint_as_float(v.y & 0xffff0000u));
    }
    pdl_trigger();
}
void k_embed(cudaStream_t s, const uint16_t* table, const uint16_t* extra, const Ctl* ctl, float* x, int T) { launch_pdl(embed_k, T, 256, 0, s, table, extra, ctl, x); }

// x += r1 + r2 (when given), xn = rmsnorm(x) * w. cap (optional) receives the updated residual.
__global__ void add_rms_k(float* __restrict__ x, const float* __restrict__ r1, const float* __restrict__ r2,
                          const float* __restrict__ w, float* __restrict__ xn, float* __restrict__ cap, int cap_stride, float eps) {
    pdl_wait();
    int t = blockIdx.x;
    float4* xp = reinterpret_cast<float4*>(x + (size_t)t * HID);
    const float4* a = r1 ? reinterpret_cast<const float4*>(r1 + (size_t)t * HID) : nullptr;
    const float4* b = r2 ? reinterpret_cast<const float4*>(r2 + (size_t)t * HID) : nullptr;
    float4 v[4];
    float ss = 0.f;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        int i = threadIdx.x + j * 256;
        float4 u = xp[i];
        if (a) { float4 q = a[i]; u.x += q.x; u.y += q.y; u.z += q.z; u.w += q.w; }
        if (b) { float4 q = b[i]; u.x += q.x; u.y += q.y; u.z += q.z; u.w += q.w; }
        v[j] = u;
        ss += u.x * u.x + u.y * u.y + u.z * u.z + u.w * u.w;
    }
    ss = warp_sum(ss);
    __shared__ float sm[8];
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) sm[warp] = ss;
    __syncthreads();
    float tot = 0.f;
#pragma unroll
    for (int i = 0; i < 8; i++) tot += sm[i];
    float inv = rsqrtf(tot / (float)HID + eps);
    float4* np = reinterpret_cast<float4*>(xn + (size_t)t * HID);
    const float4* wp = reinterpret_cast<const float4*>(w);
    float4* cp = cap ? reinterpret_cast<float4*>(cap + (size_t)t * cap_stride) : nullptr;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        int i = threadIdx.x + j * 256;
        xp[i] = v[j];
        float4 g = __ldg(wp + i);
        np[i] = make_float4(v[j].x * inv * g.x, v[j].y * inv * g.y, v[j].z * inv * g.z, v[j].w * inv * g.w);
        if (cp) cp[i] = v[j];
    }
    pdl_trigger();
}
void k_add_rms(cudaStream_t s, float* x, const float* r1, const float* r2, const float* w, float* xn, float* cap,
               int cap_stride, float eps, int T) {
    launch_pdl(add_rms_k, T, 256, 0, s, x, r1, r2, w, xn, cap, cap_stride, eps);
}

__device__ __forceinline__ int seg_of(const Ctl* ctl, int row) {
    int i = 0;
    while (i + 1 < ctl->nseg && row >= ctl->seg_row0[i + 1]) i++;
    return i;
}

// Mamba conv1d (kernel 4, causal, SiLU) over the block's rows, one thread per channel and segment. State is the
// last three inputs. The segment's slot is read; with record every row's state goes to the following slots of its
// ring (so a window can be cut after any row), otherwise only the final state goes to the next slot.
__global__ void conv_k(const Ctl* __restrict__ ctl, float* __restrict__ pool,
                       const float* __restrict__ w, const float* __restrict__ bias, const float* __restrict__ proj, int ldp,
                       float* __restrict__ xbc, int record, float* __restrict__ hist_x, float* __restrict__ hist_dt, int lazy) {
    pdl_wait();
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= 10240) return;
    const int sg = blockIdx.y;
    if (sg >= ctl->nseg) return;
    const int r0 = ctl->seg_row0[sg], r1 = ctl->seg_row0[sg + 1];
    if (hist_x && lazy) {   // the sequence's own rows, in the slot its next block does not fold from
        const size_t base = ((size_t)ctl->seg_seq[sg] * 2 + (ctl->seg_hist[sg] ^ 1)) * ctl->win;
        hist_x += (ptrdiff_t)base * HIST_ROW - (ptrdiff_t)r0 * HIST_ROW;
        hist_dt += (ptrdiff_t)base * 128 - (ptrdiff_t)r0 * 128;
    }
    const int slots = ctl->win + 1;
    float* ring = pool + (size_t)ctl->seg_seq[sg] * slots * CONV_ELEMS;
    const int cur = ctl->seg_conv[sg];
    const float* in = ring + (size_t)cur * CONV_ELEMS + (size_t)c * 3;
    float s0 = in[0], s1 = in[1], s2 = in[2];
    float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3], b = bias[c];
    for (int t = r0; t < r1; t++) {
        float xin = proj[(size_t)t * ldp + 8192 + c];
        float a = b + w0 * s0 + w1 * s1 + w2 * s2 + w3 * xin;
        float act = silu(a);
        xbc[(size_t)t * 10240 + c] = act;
        if (hist_x) {
            hist_x[(size_t)t * 10240 + c] = act;
            if (c < 128) hist_dt[t * 128 + c] = proj[(size_t)t * ldp + 18432 + c];
        }
        s0 = s1; s1 = s2; s2 = xin;
        if (record) {
            float* o = ring + (size_t)((cur + 1 + (t - r0)) % slots) * CONV_ELEMS + (size_t)c * 3;
            o[0] = s0; o[1] = s1; o[2] = s2;
        }
    }
    if (!record) {
        float* o = ring + (size_t)((cur + 1) % slots) * CONV_ELEMS + (size_t)c * 3;
        o[0] = s0; o[1] = s1; o[2] = s2;
    }
    pdl_trigger();
}
// Long single-segment blocks: the conv is a 4-tap filter with no recurrence, so rows are split over blocks.
#define CONV_ROWS 16
__global__ void conv_long_k(const Ctl* __restrict__ ctl, float* __restrict__ pool,
                            const float* __restrict__ w, const float* __restrict__ bias, const float* __restrict__ proj, int ldp,
                            float* __restrict__ xbc, int T) {
    pdl_wait();
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= 10240) return;
    const int slots = ctl->win + 1;
    float* ring = pool + (size_t)ctl->seg_seq[0] * slots * CONV_ELEMS;
    const int cur = ctl->seg_conv[0];
    const float* in = ring + (size_t)cur * CONV_ELEMS + (size_t)c * 3;
    const int t0 = blockIdx.y * CONV_ROWS, t1 = min(T, t0 + CONV_ROWS);
    // inputs t0-3 .. t0-1 (state when negative)
    float s0, s1, s2;
    {
        float v[3];
#pragma unroll
        for (int j = 0; j < 3; j++) {
            int t = t0 - 3 + j;
            v[j] = t >= 0 ? proj[(size_t)t * ldp + 8192 + c] : in[3 + t];
        }
        s0 = v[0]; s1 = v[1]; s2 = v[2];
    }
    float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3], b = bias[c];
    float xin[CONV_ROWS];
#pragma unroll
    for (int j = 0; j < CONV_ROWS; j++) xin[j] = t0 + j < t1 ? proj[(size_t)(t0 + j) * ldp + 8192 + c] : 0.f;
#pragma unroll
    for (int j = 0; j < CONV_ROWS; j++) {
        if (t0 + j < t1) {
            float a = b + w0 * s0 + w1 * s1 + w2 * s2 + w3 * xin[j];
            xbc[(size_t)(t0 + j) * 10240 + c] = silu(a);
        }
        s0 = s1; s1 = s2; s2 = xin[j];
    }
    if (t1 == T) {   // the block holding the last row writes the new state: inputs T-3 .. T-1
        float* o = ring + (size_t)((cur + 1) % slots) * CONV_ELEMS + (size_t)c * 3;
#pragma unroll
        for (int j = 0; j < 3; j++) {
            int t = T - 3 + j;
            o[j] = t >= 0 ? proj[(size_t)t * ldp + 8192 + c] : in[3 + t];
        }
    }
    pdl_trigger();
}

void k_conv(cudaStream_t s, const Ctl* ctl, float* pool, const float* w, const float* b,
            const float* proj, int ldp, float* xbc, int T, int record, float* hist_x, float* hist_dt, int nseg, int lazy) {
    if (nseg == 1 && T >= 64 && !record && !hist_x) {
        launch_pdl(conv_long_k, dim3(10240 / 128, (T + CONV_ROWS - 1) / CONV_ROWS), 128, 0, s, ctl, pool, w, b, proj, ldp, xbc, T);
        return;
    }
    launch_pdl(conv_k, dim3((10240 + 255) / 256, nseg), 256, 0, s, ctl, pool, w, b, proj, ldp, xbc, record, hist_x, hist_dt, lazy);
}

// Mamba-2 selective scan, one warp per (head, channel, segment): the warp's 128 state entries live 4 to a lane.
// The segment reads its SSM slot and writes the other one.
__global__ void __launch_bounds__(256) scan_k(const Ctl* __restrict__ ctl, float* __restrict__ pool,
                                               const float* __restrict__ xbc, const float* __restrict__ proj, int ldp,
                                               const float* __restrict__ A_log, const float* __restrict__ dt_bias,
                                               const float* __restrict__ Dp, float* __restrict__ y, int record, int lazy,
                                               const float* __restrict__ hist_x, const float* __restrict__ hist_dt) {
    pdl_wait();
    const int sg = blockIdx.y;
    if (sg >= ctl->nseg) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int h = blockIdx.x >> 3;
    const int d = (blockIdx.x & 7) * 8 + warp;
    const int g = h >> 4;  // 16 heads per group
    const int r0 = ctl->seg_row0[sg], r1 = ctl->seg_row0[sg + 1];
    float* st_pool = pool + (size_t)ctl->seg_seq[sg] * (lazy ? 1 : 2) * SSM_ELEMS;
    const int cur = lazy ? 0 : ctl->seg_ssm[sg];
    const size_t off = ((size_t)h * 64 + d) * 128 + 4 * lane;
    float4 st = *reinterpret_cast<const float4*>(st_pool + (size_t)cur * SSM_ELEMS + off);
    const float A = -__expf(A_log[h]);
    const float bias = dt_bias[h];
    const float Dh = Dp[h];
    if (lazy) {
        // the kept rows of the sequence's last window, from the inputs its conv saved: the state after them is committed
        const int nrep = ctl->seg_replay[sg];
        if (nrep > 0) {
            const size_t base = ((size_t)ctl->seg_seq[sg] * 2 + ctl->seg_hist[sg]) * ctl->win;
            for (int r = 0; r < nrep; r++) {
                float v = hist_dt[(base + r) * 128 + h] + bias;
                float sp = v > 20.f ? v : log1pf(__expf(v));
                float dA = __expf(sp * A);
                const float* row = hist_x + (base + r) * HIST_ROW;
                float sx = sp * row[h * 64 + d];
                float4 B = *reinterpret_cast<const float4*>(row + 8192 + g * 128 + 4 * lane);
                st.x = st.x * dA + sx * B.x;
                st.y = st.y * dA + sx * B.y;
                st.z = st.z * dA + sx * B.z;
                st.w = st.w * dA + sx * B.w;
            }
            if (record) *reinterpret_cast<float4*>(st_pool + off) = st;   // a prompt block commits at its end instead
        }
    }
    for (int t = r0; t < r1; t++) {
        float v = proj[(size_t)t * ldp + 18432 + h] + bias;
        float sp = v > 20.f ? v : log1pf(__expf(v));
        float dA = __expf(sp * A);
        const float* row = xbc + (size_t)t * 10240;
        float xv = row[h * 64 + d];
        float4 B = *reinterpret_cast<const float4*>(row + 8192 + g * 128 + 4 * lane);
        float4 C = *reinterpret_cast<const float4*>(row + 9216 + g * 128 + 4 * lane);
        float sx = sp * xv;
        st.x = st.x * dA + sx * B.x;
        st.y = st.y * dA + sx * B.y;
        st.z = st.z * dA + sx * B.z;
        st.w = st.w * dA + sx * B.w;
        float part = st.x * C.x + st.y * C.y + st.z * C.z + st.w * C.w;
        part = warp_sum(part);
        if (lane == 0) y[(size_t)t * 8192 + h * 64 + d] = part + Dh * xv;
    }
    if (!lazy) *reinterpret_cast<float4*>(st_pool + (size_t)(cur ^ 1) * SSM_ELEMS + off) = st;
    else if (!record) *reinterpret_cast<float4*>(st_pool + off) = st;   // a window's rows stay pending until keep
    pdl_trigger();
}

// Long single-segment blocks (prompt chunks): a thread pair owns one channel, 64 of the 128 state entries each, so
// the dot with C needs one shuffle instead of five. B, C, x and the decay of 16 steps are staged in shared memory;
// one block per head.
#define SCAN_TC 16
__global__ void __launch_bounds__(128) scan_t_k(const Ctl* __restrict__ ctl, float* __restrict__ pool,
                                                 const float* __restrict__ xbc, const float* __restrict__ proj, int ldp,
                                                 const float* __restrict__ A_log, const float* __restrict__ dt_bias,
                                                 const float* __restrict__ Dp, float* __restrict__ y, int T, int lazy,
                                                 const float* __restrict__ hist_x, const float* __restrict__ hist_dt) {
    pdl_wait();
    __shared__ __align__(16) float Bs[SCAN_TC][136], Cs[SCAN_TC][136], xs[SCAN_TC][64];
    __shared__ float dAs[SCAN_TC], sps[SCAN_TC];
    const int tid = threadIdx.x, half = tid & 1, d = tid >> 1;
    const int h = blockIdx.x, g = h >> 4;
    float* st_pool = pool + (size_t)ctl->seg_seq[0] * (lazy ? 1 : 2) * SSM_ELEMS;
    const int cur = lazy ? 0 : ctl->seg_ssm[0];
    float4 st[16];
    {
        const float4* src = reinterpret_cast<const float4*>(st_pool + (size_t)cur * SSM_ELEMS + ((size_t)h * 64 + d) * 128 + half * 64);
#pragma unroll
        for (int i = 0; i < 16; i++) st[i] = src[i];
    }
    const float A = -__expf(A_log[h]);
    const float bias = dt_bias[h];
    const float Dh = Dp[h];
    if (lazy) {   // pending rows of the sequence's last window (none after a prompt piece; kept for a block after a window)
        const int nrep = ctl->seg_replay[0];
        const size_t base = ((size_t)ctl->seg_seq[0] * 2 + ctl->seg_hist[0]) * ctl->win;
        for (int r = 0; r < nrep; r++) {
            float v = hist_dt[(base + r) * 128 + h] + bias;
            float sp = v > 20.f ? v : log1pf(__expf(v));
            float dA = __expf(sp * A);
            const float* row = hist_x + (base + r) * HIST_ROW;
            const float sx = sp * row[h * 64 + d];
            const float4* Bp = reinterpret_cast<const float4*>(row + 8192 + g * 128 + half * 64);
#pragma unroll
            for (int i = 0; i < 16; i++) {
                float4 B = Bp[i];
                st[i].x = st[i].x * dA + sx * B.x;
                st[i].y = st[i].y * dA + sx * B.y;
                st[i].z = st[i].z * dA + sx * B.z;
                st[i].w = st[i].w * dA + sx * B.w;
            }
        }
    }
    for (int c0 = 0; c0 < T; c0 += SCAN_TC) {
        const int tc = min(SCAN_TC, T - c0);
        for (int idx = tid; idx < tc * 32; idx += 128) {
            int row = idx >> 5, q = idx & 31, n = 4 * q;
            const float* r = xbc + (size_t)(c0 + row) * 10240;
            *reinterpret_cast<float4*>(&Bs[row][n + (n >> 6) * 4]) = *reinterpret_cast<const float4*>(r + 8192 + g * 128 + n);
            *reinterpret_cast<float4*>(&Cs[row][n + (n >> 6) * 4]) = *reinterpret_cast<const float4*>(r + 9216 + g * 128 + n);
        }
        for (int idx = tid; idx < tc * 16; idx += 128) {
            int row = idx >> 4, q = idx & 15;
            *reinterpret_cast<float4*>(&xs[row][4 * q]) = *reinterpret_cast<const float4*>(xbc + (size_t)(c0 + row) * 10240 + h * 64 + 4 * q);
        }
        if (tid < tc) {
            float v = proj[(size_t)(c0 + tid) * ldp + 18432 + h] + bias;
            float sp = v > 20.f ? v : log1pf(__expf(v));
            sps[tid] = sp;
            dAs[tid] = __expf(sp * A);
        }
        __syncthreads();
        for (int tl = 0; tl < tc; tl++) {
            const float sp = sps[tl], dA = dAs[tl], xv = xs[tl][d], sx = sp * xv;
            const float4* Bp = reinterpret_cast<const float4*>(&Bs[tl][half * 68]);
            const float4* Cp = reinterpret_cast<const float4*>(&Cs[tl][half * 68]);
            float part = 0.f;
#pragma unroll
            for (int i = 0; i < 16; i++) {
                float4 B = Bp[i], C = Cp[i];
                st[i].x = st[i].x * dA + sx * B.x;
                st[i].y = st[i].y * dA + sx * B.y;
                st[i].z = st[i].z * dA + sx * B.z;
                st[i].w = st[i].w * dA + sx * B.w;
                part += st[i].x * C.x + st[i].y * C.y + st[i].z * C.z + st[i].w * C.w;
            }
            part += __shfl_xor_sync(0xffffffffu, part, 1);
            if (half == 0) y[(size_t)(c0 + tl) * 8192 + h * 64 + d] = part + Dh * xv;
        }
        __syncthreads();
    }
    float4* dst = reinterpret_cast<float4*>(st_pool + (size_t)(lazy ? 0 : (cur ^ 1)) * SSM_ELEMS + ((size_t)h * 64 + d) * 128 + half * 64);
#pragma unroll
    for (int i = 0; i < 16; i++) dst[i] = st[i];
    pdl_trigger();
}

// ---------------------------------------------------------------------------------------------------------
// Prompt scan on tensor cores (BOLT_SCAN_SSD, default): Mamba-2's chunked form. For a head and a chunk of L = 64
// rows with decays a_t = exp(dt_t A) and cum_t = sum of dt up to t: G = C B^T (L x L), Gm[t][tau] = G[t][tau]
// exp(A (cum_t - cum_tau)) dt_tau for tau <= t, y = Gm X + diag(exp(A cum_t)) (C S0^T) + D x, and the next state
// S = exp(A cum_L) S0 + X^T (w o B) with w_tau = exp(A (cum_L - cum_tau)) dt_tau. The four products run on
// m16n8k8 tf32 MMAs (inputs rounded to 10 mantissa bits, fp32 sums); the state stays fp32 in shared memory. One CTA
// of 8 warps a head walks the chunks in order; n (128) is taken in two halves to fit shared memory.
// ---------------------------------------------------------------------------------------------------------

__device__ __forceinline__ uint32_t tf32(float v) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(r) : "f"(v)); return r; }
__device__ __forceinline__ void mma_tf32(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
constexpr int SSD_L = 64, SSD_STR = 68, SSD_SSTR = 132;   // chunk rows; row strides (floats) of the 64-wide and 128-wide tiles

__global__ void __launch_bounds__(256, 1) scan_ssd_k(const Ctl* __restrict__ ctl, float* __restrict__ pool,
                                                     const float* __restrict__ xbc, const float* __restrict__ proj, int ldp,
                                                     const float* __restrict__ A_log, const float* __restrict__ dt_bias,
                                                     const float* __restrict__ Dp, float* __restrict__ y, int T, int lazy,
                                                     const float* __restrict__ hist_x, const float* __restrict__ hist_dt) {
    pdl_wait();
    extern __shared__ __align__(16) float ssm[];
    float* S = ssm;                                   // [64 d][SSD_SSTR] the state, fp32
    float* Bh = S + 64 * SSD_SSTR;                     // [64 t][SSD_STR] this half's B
    float* Ch = Bh + 64 * SSD_STR;                     // [64 t][SSD_STR] this half's C, then Gm
    float* X = Ch + 64 * SSD_STR;                      // [64 t][SSD_STR] x
    __shared__ float dts[SSD_L], cum[SSD_L], et[SSD_L];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t4 = lane & 3;
    const int h = blockIdx.x, grp = h >> 4;
    float* st_pool = pool + (size_t)ctl->seg_seq[0] * (lazy ? 1 : 2) * SSM_ELEMS;
    const int cur = lazy ? 0 : ctl->seg_ssm[0];
    const float A = -__expf(A_log[h]), bias = dt_bias[h], Dh = Dp[h];
    // the state in: [d][n] (pool layout: (h * 64 + d) * 128 + n)
    for (int i = tid; i < 64 * 128; i += 256) S[(i >> 7) * SSD_SSTR + (i & 127)] = st_pool[(size_t)cur * SSM_ELEMS + (size_t)h * 64 * 128 + i];
    __syncthreads();
    if (lazy) {   // pending rows of the sequence's last window first, as scan_t_k does
        const int nrep = ctl->seg_replay[0];
        const size_t base = ((size_t)ctl->seg_seq[0] * 2 + ctl->seg_hist[0]) * ctl->win;
        for (int r = 0; r < nrep; r++) {
            const float v = hist_dt[(base + r) * 128 + h] + bias;
            const float sp = v > 20.f ? v : log1pf(__expf(v));
            const float dA = __expf(sp * A);
            const float* row = hist_x + (base + r) * HIST_ROW;
            for (int i = tid; i < 64 * 128; i += 256) {
                const int d = i >> 7, n = i & 127;
                S[d * SSD_SSTR + n] = S[d * SSD_SSTR + n] * dA + sp * row[h * 64 + d] * row[8192 + grp * 128 + n];
            }
            __syncthreads();
        }
    }
    for (int c0 = 0; c0 < T; c0 += SSD_L) {
        const int tc = min(SSD_L, T - c0);
        // dt, its prefix sum and the decays of the chunk (one warp, in order)
        if (warp == 0) {
            float v = 0.f;
            if (lane < 2) {
                // lanes 0 and 1 each take 32 rows in sequence; lane 1's sums add lane 0's total afterwards
                for (int i = 0; i < 32; i++) {
                    const int t = lane * 32 + i;
                    float sp = 0.f;
                    if (t < tc) { const float z = proj[(size_t)(c0 + t) * ldp + 18432 + h] + bias; sp = z > 20.f ? z : log1pf(__expf(z)); }
                    v += sp; dts[t] = sp; cum[t] = v;
                }
            }
            __syncwarp();
            if (lane == 0) { const float c = cum[31]; for (int t = 32; t < 64; t++) cum[t] += c; }
            __syncwarp();
            for (int t = lane; t < 64; t += 32) et[t] = __expf(A * cum[t]);
        }
        // x of the chunk (zero past tc)
        for (int i = tid; i < 64 * 16; i += 256) {
            const int t = i >> 4, q = (i & 15) * 4;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (t < tc) v = *reinterpret_cast<const float4*>(xbc + (size_t)(c0 + t) * 10240 + h * 64 + q);
            *reinterpret_cast<float4*>(X + t * SSD_STR + q) = v;
        }
        __syncthreads();
        const float eL = et[tc - 1];
        // accumulators: G (this warp: t-tile warp >> 1, tau-tiles (warp & 1) * 4 .. + 3) and Y (d-tile warp, t-tiles 0..3)
        float G[4][4], Y[4][4];
#pragma unroll
        for (int i = 0; i < 4; i++) { G[i][0] = G[i][1] = G[i][2] = G[i][3] = 0.f; Y[i][0] = Y[i][1] = Y[i][2] = Y[i][3] = 0.f; }
        for (int hh = 0; hh < 2; hh++) {
            // B and C of this n half
            for (int i = tid; i < 64 * 16; i += 256) {
                const int t = i >> 4, q = (i & 15) * 4;
                float4 b = make_float4(0.f, 0.f, 0.f, 0.f), c = b;
                if (t < tc) {
                    const float* r = xbc + (size_t)(c0 + t) * 10240 + grp * 128 + hh * 64 + q;
                    b = *reinterpret_cast<const float4*>(r + 8192);
                    c = *reinterpret_cast<const float4*>(r + 9216);
                }
                *reinterpret_cast<float4*>(Bh + t * SSD_STR + q) = b;
                *reinterpret_cast<float4*>(Ch + t * SSD_STR + q) = c;
            }
            __syncthreads();
            // G += C_h B_h^T: A = C [t][n], B operand = B^T: (k = n, col = tau) -> B[tau][n]
            {
                const int mt = warp >> 1, nb = (warp & 1) * 4;
#pragma unroll
                for (int ks = 0; ks < 8; ks++) {
                    const int k0 = ks * 8;
                    const uint32_t a0 = tf32(Ch[(mt * 16 + g) * SSD_STR + k0 + t4]), a1 = tf32(Ch[(mt * 16 + g + 8) * SSD_STR + k0 + t4]);
                    const uint32_t a2 = tf32(Ch[(mt * 16 + g) * SSD_STR + k0 + t4 + 4]), a3 = tf32(Ch[(mt * 16 + g + 8) * SSD_STR + k0 + t4 + 4]);
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        const int tau = (nb + j) * 8 + g;
                        mma_tf32(G[j], a0, a1, a2, a3, tf32(Bh[tau * SSD_STR + k0 + t4]), tf32(Bh[tau * SSD_STR + k0 + t4 + 4]));
                    }
                }
            }
            // Y += C_h S0_h^T: A = C [t][n], B operand (k = n, col = d) -> S[d][n]
            {
                const int d0 = warp * 8;
#pragma unroll
                for (int ks = 0; ks < 8; ks++) {
                    const int k0 = hh * 64 + ks * 8;
                    const uint32_t b0 = tf32(S[(d0 + g) * SSD_SSTR + k0 + t4]), b1 = tf32(S[(d0 + g) * SSD_SSTR + k0 + t4 + 4]);
#pragma unroll
                    for (int m = 0; m < 4; m++) {
                        const int kk = ks * 8;
                        mma_tf32(Y[m], tf32(Ch[(m * 16 + g) * SSD_STR + kk + t4]), tf32(Ch[(m * 16 + g + 8) * SSD_STR + kk + t4]),
                                 tf32(Ch[(m * 16 + g) * SSD_STR + kk + t4 + 4]), tf32(Ch[(m * 16 + g + 8) * SSD_STR + kk + t4 + 4]), b0, b1);
                    }
                }
            }
            __syncthreads();   // every read of S0's half is done before it is replaced
            // S_h = eL S0_h + X^T (w o B_h): output [d][n] tiles: d-tile warp >> 1 (16 d), n-tiles (warp & 1) * 4 .. + 3 of this half
            {
                const int dt16 = warp >> 1, nb = (warp & 1) * 4;
                float Sn[4][4];
#pragma unroll
                for (int j = 0; j < 4; j++) Sn[j][0] = Sn[j][1] = Sn[j][2] = Sn[j][3] = 0.f;
#pragma unroll
                for (int ks = 0; ks < 8; ks++) {
                    const int k0 = ks * 8;   // tau
                    // A = X^T: (row d, col tau) = X[tau][d]
                    const uint32_t a0 = tf32(X[(k0 + t4) * SSD_STR + dt16 * 16 + g]), a1 = tf32(X[(k0 + t4) * SSD_STR + dt16 * 16 + g + 8]);
                    const uint32_t a2 = tf32(X[(k0 + t4 + 4) * SSD_STR + dt16 * 16 + g]), a3 = tf32(X[(k0 + t4 + 4) * SSD_STR + dt16 * 16 + g + 8]);
                    const float w0 = __expf(A * (cum[tc - 1] - cum[k0 + t4])) * dts[k0 + t4];
                    const float w1 = __expf(A * (cum[tc - 1] - cum[k0 + t4 + 4])) * dts[k0 + t4 + 4];
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        const int n = (nb + j) * 8 + g;
                        mma_tf32(Sn[j], a0, a1, a2, a3, tf32(w0 * Bh[(k0 + t4) * SSD_STR + n]), tf32(w1 * Bh[(k0 + t4 + 4) * SSD_STR + n]));
                    }
                }
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const int n = hh * 64 + (nb + j) * 8 + 2 * t4;
                    float* r0 = S + (dt16 * 16 + g) * SSD_SSTR + n;
                    float* r1 = S + (dt16 * 16 + g + 8) * SSD_SSTR + n;
                    r0[0] = eL * r0[0] + Sn[j][0]; r0[1] = eL * r0[1] + Sn[j][1];
                    r1[0] = eL * r1[0] + Sn[j][2]; r1[1] = eL * r1[1] + Sn[j][3];
                }
            }
            __syncthreads();   // the half's B and C are free again
        }
        // the mask on G -> Gm in shared memory (over C's buffer), and Y scaled by the row decays
        {
            const int mt = warp >> 1, nb = (warp & 1) * 4;
#pragma unroll
            for (int j = 0; j < 4; j++) {
#pragma unroll
                for (int q = 0; q < 4; q++) {
                    const int t = mt * 16 + g + ((q >> 1) ? 8 : 0), tau = (nb + j) * 8 + 2 * t4 + (q & 1);
                    const float m = (tau <= t && t < tc) ? __expf(A * (cum[t] - cum[tau])) * dts[tau] : 0.f;
                    Ch[t * SSD_STR + tau] = G[j][q] * m;
                }
            }
#pragma unroll
            for (int m = 0; m < 4; m++) {
                const int t0 = m * 16 + g, t1 = t0 + 8;
                Y[m][0] *= et[t0]; Y[m][1] *= et[t0]; Y[m][2] *= et[t1]; Y[m][3] *= et[t1];
            }
        }
        __syncthreads();
        // Y += Gm X: A = Gm [t][tau], B operand (k = tau, col = d) -> X[tau][d]; d-tile warp
        {
            const int d0 = warp * 8;
#pragma unroll
            for (int ks = 0; ks < 8; ks++) {
                const int k0 = ks * 8;
                const uint32_t b0 = tf32(X[(k0 + t4) * SSD_STR + d0 + g]), b1 = tf32(X[(k0 + t4 + 4) * SSD_STR + d0 + g]);
#pragma unroll
                for (int m = 0; m < 4; m++)
                    mma_tf32(Y[m], tf32(Ch[(m * 16 + g) * SSD_STR + k0 + t4]), tf32(Ch[(m * 16 + g + 8) * SSD_STR + k0 + t4]),
                             tf32(Ch[(m * 16 + g) * SSD_STR + k0 + t4 + 4]), tf32(Ch[(m * 16 + g + 8) * SSD_STR + k0 + t4 + 4]), b0, b1);
            }
            // out: y[t][h * 64 + d] = Y + D x
#pragma unroll
            for (int m = 0; m < 4; m++)
#pragma unroll
                for (int q = 0; q < 4; q++) {
                    const int t = m * 16 + g + ((q >> 1) ? 8 : 0), d = d0 + 2 * t4 + (q & 1);
                    if (t < tc) y[(size_t)(c0 + t) * 8192 + h * 64 + d] = Y[m][q] + Dh * X[t * SSD_STR + d];
                }
        }
        __syncthreads();   // Gm (C's buffer) and X are free for the next chunk
    }
    // the state out
    float* dst = st_pool + (size_t)(lazy ? 0 : (cur ^ 1)) * SSM_ELEMS + (size_t)h * 64 * 128;
    for (int i = tid; i < 64 * 128; i += 256) dst[i] = S[(i >> 7) * SSD_SSTR + (i & 127)];
    pdl_trigger();
}
constexpr size_t SSD_SMEM = (size_t)(64 * SSD_SSTR + 3 * 64 * SSD_STR) * 4;   // 86 KB

void k_scan(cudaStream_t s, const Ctl* ctl, float* pool, const float* xbc, const float* proj,
            int ldp, const float* A_log, const float* dt_bias, const float* D, float* y, int T, int record, int nseg,
            int lazy, const float* hist_x, const float* hist_dt) {
    if (nseg == 1 && T >= 64 && !record) {
        // BOLT_SCAN_SSD=1: the chunked tensor-core scan. Correct (rms relative difference 4e-4, tf32) but 2.5x slower as
        // written: 86 KB of shared memory allows one CTA an SM and each chunk's loads, barriers and prefix sum leave the
        // tensor cores idle; it would need the state in registers, double-buffered loads and a parallel prefix to pay.
        static const bool ssd = getenv("BOLT_SCAN_SSD") && atoi(getenv("BOLT_SCAN_SSD")) != 0;
        if (ssd) {
            static bool opted = false;
            if (!opted) { cudaFuncSetAttribute(scan_ssd_k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SSD_SMEM); opted = true; }
            launch_pdl(scan_ssd_k, 128, 256, SSD_SMEM, s, ctl, pool, xbc, proj, ldp, A_log, dt_bias, D, y, T, lazy, hist_x, hist_dt);
        } else launch_pdl(scan_t_k, 128, 128, 0, s, ctl, pool, xbc, proj, ldp, A_log, dt_bias, D, y, T, lazy, hist_x, hist_dt);
        return;
    }
    launch_pdl(scan_k, dim3(128 * 8, nseg), 256, 0, s, ctl, pool, xbc, proj, ldp, A_log, dt_bias, D, y, record, lazy, hist_x, hist_dt);
}

// out = RMSNorm_group(y * silu(z)) * w, groups of 1024.
__global__ void gate_rms_k(const float* __restrict__ y, const float* __restrict__ proj, int ldp, const float* __restrict__ w,
                           float* __restrict__ out, float eps) {
    pdl_wait();
    int t = blockIdx.x >> 3, g = blockIdx.x & 7;
    const float4* yp = reinterpret_cast<const float4*>(y + (size_t)t * 8192 + g * 1024);
    const float4* zp = reinterpret_cast<const float4*>(proj + (size_t)t * ldp + g * 1024);
    float4 v = {0, 0, 0, 0};
    float ss = 0.f;
    {
        float4 a = yp[threadIdx.x], z = zp[threadIdx.x];
        v = make_float4(a.x * silu(z.x), a.y * silu(z.y), a.z * silu(z.z), a.w * silu(z.w));
        ss = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    ss = warp_sum(ss);
    __shared__ float sm[8];
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) sm[warp] = ss;
    __syncthreads();
    float tot = 0.f;
#pragma unroll
    for (int i = 0; i < 8; i++) tot += sm[i];
    float inv = rsqrtf(tot / 1024.f + eps);
    float4 g4 = __ldg(reinterpret_cast<const float4*>(w + g * 1024) + threadIdx.x);
    reinterpret_cast<float4*>(out + (size_t)t * 8192 + g * 1024)[threadIdx.x] =
        make_float4(v.x * inv * g4.x, v.y * inv * g4.y, v.z * inv * g4.z, v.w * inv * g4.w);
    pdl_trigger();
}
void k_gate_rms(cudaStream_t s, const float* y, const float* proj, int ldp, const float* w, float* out, float eps, int T) {
    launch_pdl(gate_rms_k, T * 8, 256, 0, s, y, proj, ldp, w, out, eps);
}

// Sigmoid router: the top TOPN of sigmoid(logit) + bias (ties to the lower index), weights are the plain
// sigmoids normalised to one and scaled. One warp per token; E <= 512.
__global__ void route_k(const float* __restrict__ logits, int ldl, const float* __restrict__ bias, int* __restrict__ ridx,
                        float* __restrict__ rw, int E, float scale) {
    pdl_wait();
    int t = blockIdx.x, lane = threadIdx.x;
    float sig[16], sel[16];
#pragma unroll
    for (int j = 0; j < 16; j++) {
        int i = lane + 32 * j;
        if (i < E) {
            float z = logits[(size_t)t * ldl + i];
            sig[j] = 1.f / (1.f + __expf(-z));
            sel[j] = sig[j] + bias[i];
        } else {
            sig[j] = 0.f;
            sel[j] = -1e30f;
        }
    }
    float picked[TOPN];
    int pid[TOPN];
#pragma unroll
    for (int k = 0; k < TOPN; k++) {
        float bv = -3e38f;
        int bi = 0x7fffffff;
#pragma unroll
        for (int j = 0; j < 16; j++) {
            int i = lane + 32 * j;
            if (sel[j] > bv) { bv = sel[j]; bi = i; }
        }
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) {
            float ov = __shfl_xor_sync(0xffffffffu, bv, m);
            int oi = __shfl_xor_sync(0xffffffffu, bi, m);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        pid[k] = bi;
        float ps = 0.f;
#pragma unroll
        for (int j = 0; j < 16; j++) {
            if (lane + 32 * j == bi) { ps = sig[j]; sel[j] = -1e30f; }
        }
        picked[k] = warp_sum(ps);
    }
    float sum = 1e-20f;
#pragma unroll
    for (int k = 0; k < TOPN; k++) sum += picked[k];
    if (lane < TOPN) {
        int idx = 0;
        float val = 0.f;
#pragma unroll
        for (int k = 0; k < TOPN; k++) if (k == lane) { idx = pid[k]; val = picked[k]; }
        ridx[t * TOPN + lane] = idx;
        rw[t * TOPN + lane] = val / sum * scale;
    }
    pdl_trigger();
}
void k_route(cudaStream_t s, const float* logits, int ldl, const float* bias, int* ridx, float* rw, int E, float scale, int T) {
    launch_pdl(route_k, T, 32, 0, s, logits, ldl, bias, ridx, rw, E, scale);
}

// out[t] = sum_k rw[t][k] * o[t*TOPN + k]  (1024 wide)
__global__ void combine_k(const float* __restrict__ o, const float* __restrict__ rw, float* __restrict__ out) {
    pdl_wait();
    int t = blockIdx.x;
    float4 acc = {0, 0, 0, 0};
    for (int k = 0; k < TOPN; k++) {
        float wk = rw[t * TOPN + k];
        float4 v = reinterpret_cast<const float4*>(o + (size_t)(t * TOPN + k) * 1024)[threadIdx.x];
        acc.x += wk * v.x; acc.y += wk * v.y; acc.z += wk * v.z; acc.w += wk * v.w;
    }
    reinterpret_cast<float4*>(out + (size_t)t * 1024)[threadIdx.x] = acc;
    pdl_trigger();
}
void k_combine(cudaStream_t s, const float* o, const float* rw, float* out, int T) { launch_pdl(combine_k, T, 256, 0, s, o, rw, out); }

// ---------------------------------------------------------------------------------------------------------
// Attention: 32 query heads over 2 KV heads, bf16 cache [pos][2][128]. Split-KV so long contexts use the GPU.
// ---------------------------------------------------------------------------------------------------------

// fp8 e4m3 cache: one fp32 scale per (token, head), amax / 448, so every row uses the format's whole range.
// A block takes one token: warps 0-1 are K heads 0-1, warps 2-3 V heads 0-1.
__global__ void __launch_bounds__(128) kv_store_k(const Ctl* __restrict__ ctl, const float* __restrict__ k, const float* __restrict__ v, KvView kv) {
    pdl_wait();
    int t = blockIdx.x;
    if (t >= ctl->nrows) return;   // padding rows of a block rounded up to a stable shape
    int sg = seg_of(ctl, t);
    size_t pos = (size_t)ctl->seg_seq[sg] * kv.ctx + (size_t)(ctl->seg_base[sg] + t - ctl->seg_row0[sg]);
    if (!kv.fp8) {
        uint16_t* kc = (uint16_t*)kv.k;
        uint16_t* vc = (uint16_t*)kv.v;
        for (int i = threadIdx.x; i < 256; i += blockDim.x) {
            kc[pos * 256 + i] = __bfloat16_as_ushort(__float2bfloat16(k[t * 256 + i]));
            vc[pos * 256 + i] = __bfloat16_as_ushort(__float2bfloat16(v[t * 256 + i]));
        }
        return;
    }
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int isv = warp >> 1, h = warp & 1;
    const float* src = (isv ? v : k) + (size_t)t * 256 + h * 128;
    float4 x = reinterpret_cast<const float4*>(src)[lane];
    float am = fmaxf(fmaxf(fabsf(x.x), fabsf(x.y)), fmaxf(fabsf(x.z), fabsf(x.w)));
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, m));
    const float sc = am > 0.f ? am * (1.f / 448.f) : 1.f, inv = 1.f / sc;
    __nv_fp8x4_e4m3 q(make_float4(x.x * inv, x.y * inv, x.z * inv, x.w * inv));
    uint8_t* dst = (uint8_t*)(isv ? kv.v : kv.k) + (pos * 2 + h) * 128;
    reinterpret_cast<uint32_t*>(dst)[lane] = *reinterpret_cast<uint32_t*>(&q);
    if (lane == 0) (isv ? kv.vs : kv.ks)[pos * 2 + h] = sc;
    pdl_trigger();
}
void k_kv_store(cudaStream_t s, const Ctl* ctl, const float* k, const float* v, const KvView& kv, int T) {
    launch_pdl(kv_store_k, T, 128, 0, s, ctl, k, v, kv);
}

// One K or V element of (sequence, position, head, dim) for the CUDA-core kernels.
__device__ __forceinline__ float kv_elem(const KvView& kv, bool isv, size_t seq, size_t pos, int g, int d) {
    const size_t row = (seq * kv.ctx + pos) * 2 + g;
    if (!kv.fp8) return __uint_as_float((unsigned)((const uint16_t*)(isv ? kv.v : kv.k))[row * 128 + d] << 16);
    const uint8_t b = ((const uint8_t*)(isv ? kv.v : kv.k))[row * 128 + d];
    __half_raw hr = __nv_cvt_fp8_to_halfraw(b, __NV_E4M3);
    return __half2float(__half(hr)) * (isv ? kv.vs : kv.ks)[row];
}

constexpr int ATT_PART = 16 + 16 + 16 * 128;  // m[16], l[16], o[16][128]

__global__ void __launch_bounds__(256) attn_part_k(const Ctl* __restrict__ ctl, const float* __restrict__ q, KvView kvc,
                                                   float* __restrict__ part, int splits) {
    pdl_wait();
    const int sp = blockIdx.x, g = blockIdx.y, t = blockIdx.z;
    const bool pad = t >= ctl->nrows;   // padding rows of a block rounded up to a stable shape read one key of sequence 0
    const int sg = pad ? 0 : seg_of(ctl, t);
    const int len = pad ? 1 : ctl->seg_base[sg] + (t - ctl->seg_row0[sg]) + 1;
    const size_t seq = pad ? 0 : ctl->seg_seq[sg];
    int chunk = ((len + splits - 1) / splits + 31) / 32 * 32;
    const int p0 = sp * chunk, p1 = min(len, p0 + chunk);
    float* out = part + (size_t)((t * 2 + g) * splits + sp) * ATT_PART;

    __shared__ float qs[16][128];
    __shared__ float ks[128][33];
    __shared__ float vs[32][128];
    __shared__ float ps[16][32];
    __shared__ float mrun[16], lrun[16], scl[16];
    const int tid = threadIdx.x;
    const float qscale = rsqrtf(128.f);
    for (int i = tid; i < 16 * 128; i += 256) qs[i >> 7][i & 127] = q[(size_t)t * 4096 + g * 2048 + i] * qscale;
    if (tid < 16) { mrun[tid] = -1e30f; lrun[tid] = 0.f; }
    const int ph = tid >> 4, pl = tid & 15;
    float acc[8];
#pragma unroll
    for (int i = 0; i < 8; i++) acc[i] = 0.f;
    __syncthreads();
    for (int base = p0; base < p1; base += 32) {
        int n = min(32, p1 - base);
        // stage the K tile transposed and the V tile
        for (int i = tid; i < 32 * 128; i += 256) {
            int p = i >> 7, d = i & 127;
            float kv = 0.f, vv = 0.f;
            if (p < n) {
                kv = kv_elem(kvc, false, seq, base + p, g, d);
                vv = kv_elem(kvc, true, seq, base + p, g, d);
            }
            ks[d][p] = kv;
            vs[p][d] = vv;
        }
        __syncthreads();
        for (int pair = tid; pair < 16 * 32; pair += 256) {
            int h = pair >> 5, p = pair & 31;
            float dot = 0.f;
#pragma unroll 8
            for (int d = 0; d < 128; d++) dot += qs[h][d] * ks[d][p];
            ps[h][p] = p < n ? dot : -1e30f;
        }
        __syncthreads();
        if (tid < 16) {
            float mx = mrun[tid];
            for (int p = 0; p < 32; p++) mx = fmaxf(mx, ps[tid][p]);
            float sc = __expf(mrun[tid] - mx);
            float l = lrun[tid] * sc;
            for (int p = 0; p < 32; p++) {
                float e = ps[tid][p] > -1e29f ? __expf(ps[tid][p] - mx) : 0.f;
                ps[tid][p] = e;
                l += e;
            }
            mrun[tid] = mx; lrun[tid] = l; scl[tid] = sc;
        }
        __syncthreads();
        {
            float sc = scl[ph];
#pragma unroll
            for (int i = 0; i < 8; i++) acc[i] *= sc;
            for (int p = 0; p < 32; p++) {
                float pr = ps[ph][p];
#pragma unroll
                for (int i = 0; i < 8; i++) acc[i] += pr * vs[p][pl + 16 * i];
            }
        }
        __syncthreads();
    }
    if (tid < 16) { out[tid] = mrun[tid]; out[16 + tid] = lrun[tid]; }
#pragma unroll
    for (int i = 0; i < 8; i++) out[32 + ph * 128 + pl + 16 * i] = acc[i];
    pdl_trigger();
}

__global__ void attn_comb_k(const float* __restrict__ part, float* __restrict__ out, int splits) {
    pdl_wait();
    int t = blockIdx.x, h = blockIdx.y;
    int g = h >> 4, hh = h & 15;
    int d = threadIdx.x;
    const float* base = part + (size_t)((t * 2 + g) * splits) * ATT_PART;
    float M = -1e30f;
    for (int s = 0; s < splits; s++) M = fmaxf(M, base[(size_t)s * ATT_PART + hh]);
    float L = 0.f, O = 0.f;
    for (int s = 0; s < splits; s++) {
        const float* p = base + (size_t)s * ATT_PART;
        float w = __expf(p[hh] - M);
        L += w * p[16 + hh];
        O += w * p[32 + hh * 128 + d];
    }
    out[(size_t)t * 4096 + h * 128 + d] = O / L;
    pdl_trigger();
}

// Prompt chunks: a CTA takes QB queries x 16 heads of one KV head against a split of the keys, so each K/V tile is
// staged once for QB queries instead of once per query.
constexpr int QB = 4;
constexpr int BROWS = QB * 16;
constexpr int BPART = BROWS + BROWS + BROWS * 128;  // m, l, o[BROWS][128]
constexpr int BLK_SMEM = (BROWS * 128 + 128 * 33 + 32 * 128 + BROWS * 32) * 4;

__global__ void __launch_bounds__(256) attn_blk_part_k(const Ctl* __restrict__ ctl, const float* __restrict__ q,
                                                       const uint16_t* __restrict__ kc_all, const uint16_t* __restrict__ vc_all,
                                                       float* __restrict__ part, int splits, int T, int ctx) {
    pdl_wait();   // bf16 cache only
    const uint16_t* kc = kc_all + (size_t)ctl->seg_seq[0] * ctx * 256;
    const uint16_t* vc = vc_all + (size_t)ctl->seg_seq[0] * ctx * 256;
    const int base0 = ctl->seg_base[0];
    extern __shared__ float sm[];
    float* qs = sm;                       // [BROWS][128]
    float* ks = qs + BROWS * 128;         // [128][33]
    float* vs = ks + 128 * 33;            // [32][128]
    float* ps = vs + 32 * 128;            // [BROWS][32]
    __shared__ float mrun[BROWS], lrun[BROWS], scl[BROWS];
    __shared__ int rlen[BROWS];
    const int sp = blockIdx.x, g = blockIdx.y, qt = blockIdx.z;
    const int tid = threadIdx.x;
    const int maxlen = base0 + min(T, qt * QB + QB);
    int chunk = ((maxlen + splits - 1) / splits + 31) / 32 * 32;
    const int p0 = sp * chunk, p1 = min(maxlen, p0 + chunk);
    float* out = part + (size_t)((qt * 2 + g) * splits + sp) * BPART;
    const float qscale = rsqrtf(128.f);
    for (int i = tid; i < BROWS * 128; i += 256) {
        int r = i >> 7, d = i & 127;
        int t = qt * QB + (r >> 4), hh = r & 15;
        qs[i] = t < T ? q[(size_t)t * 4096 + (g * 16 + hh) * 128 + d] * qscale : 0.f;
    }
    if (tid < BROWS) {
        int t = qt * QB + (tid >> 4);
        rlen[tid] = t < T ? base0 + t + 1 : 0;
        mrun[tid] = -1e30f;
        lrun[tid] = 0.f;
    }
    const int pr = tid >> 2, pc = (tid & 3) * 32;  // PV: row, first dim
    float acc[32];
#pragma unroll
    for (int i = 0; i < 32; i++) acc[i] = 0.f;
    __syncthreads();
    for (int base = p0; base < p1; base += 32) {
        const int n = min(32, p1 - base);
        for (int i = tid; i < 32 * 128; i += 256) {
            int p = i >> 7, d = i & 127;
            float kv = 0.f, vv = 0.f;
            if (p < n) {
                size_t o = ((size_t)(base + p) * 2 + g) * 128 + d;
                kv = __uint_as_float((unsigned)kc[o] << 16);
                vv = __uint_as_float((unsigned)vc[o] << 16);
            }
            ks[d * 33 + p] = kv;
            vs[p * 128 + d] = vv;
        }
        __syncthreads();
        for (int pair = tid; pair < BROWS * 32; pair += 256) {
            int r = pair >> 5, p = pair & 31;
            float dot = 0.f;
#pragma unroll 8
            for (int d = 0; d < 128; d++) dot += qs[r * 128 + d] * ks[d * 33 + p];
            ps[r * 32 + p] = (p < n && base + p < rlen[r]) ? dot : -1e30f;
        }
        __syncthreads();
        if (tid < BROWS) {
            float mx = mrun[tid];
            for (int p = 0; p < 32; p++) mx = fmaxf(mx, ps[tid * 32 + p]);
            float sc = __expf(mrun[tid] - mx);
            float l = lrun[tid] * sc;
            for (int p = 0; p < 32; p++) {
                float e = ps[tid * 32 + p] > -1e29f ? __expf(ps[tid * 32 + p] - mx) : 0.f;
                ps[tid * 32 + p] = e;
                l += e;
            }
            mrun[tid] = mx; lrun[tid] = l; scl[tid] = sc;
        }
        __syncthreads();
        {
            float sc = scl[pr];
#pragma unroll
            for (int i = 0; i < 32; i++) acc[i] *= sc;
            for (int p = 0; p < 32; p++) {
                float pe = ps[pr * 32 + p];
#pragma unroll
                for (int i = 0; i < 32; i++) acc[i] += pe * vs[p * 128 + pc + i];
            }
        }
        __syncthreads();
    }
    if (tid < BROWS) { out[tid] = mrun[tid]; out[BROWS + tid] = lrun[tid]; }
#pragma unroll
    for (int i = 0; i < 32; i++) out[2 * BROWS + pr * 128 + pc + i] = acc[i];
    pdl_trigger();
}

__global__ void attn_blk_comb_k(const float* __restrict__ part, float* __restrict__ out, int splits) {
    pdl_wait();
    int t = blockIdx.x, h = blockIdx.y;
    int g = h >> 4, hh = h & 15;
    int qt = t / QB, r = (t % QB) * 16 + hh;
    int d = threadIdx.x;
    const float* base = part + (size_t)((qt * 2 + g) * splits) * BPART;
    float M = -1e30f;
    for (int s = 0; s < splits; s++) M = fmaxf(M, base[(size_t)s * BPART + r]);
    float L = 0.f, O = 0.f;
    for (int s = 0; s < splits; s++) {
        const float* p = base + (size_t)s * BPART;
        float w = __expf(p[r] - M);
        L += w * p[BROWS + r];
        O += w * p[2 * BROWS + r * 128 + d];
    }
    out[(size_t)t * 4096 + h * 128 + d] = O / L;
    pdl_trigger();
}

void k_attn(cudaStream_t s, const Ctl* ctl, const float* q, const KvView& kv, float* part, float* out,
            int T, int splits, int multi, int) {
    if (T <= 8 || multi) {
        launch_pdl(attn_part_k, dim3(splits, 2, T), 256, 0, s, ctl, q, kv, part, splits);
        launch_pdl(attn_comb_k, dim3(T, 32), 128, 0, s, part, out, splits);
        return;
    }
    // the query-blocked prompt kernel (bf16 cache only; prompts use k_attn_mma)
    static bool opted = false;
    if (!opted) {
        cudaFuncSetAttribute(attn_blk_part_k, cudaFuncAttributeMaxDynamicSharedMemorySize, BLK_SMEM);
        opted = true;
    }
    launch_pdl(attn_blk_part_k, dim3(splits, 2, (T + QB - 1) / QB), 256, BLK_SMEM, s, ctl, q, (const uint16_t*)kv.k, (const uint16_t*)kv.v, part, splits, T, kv.ctx);
    launch_pdl(attn_blk_comb_k, dim3(T, 32), 128, 0, s, part, out, splits);
}

// ---------------------------------------------------------------------------------------------------------
// Row-wise top candidates of the logits. Stage one: 128 blocks per row, each selects its slice's best
// `topk`; stage two merges them. Ties go to the lower id, so the order is total.
// ---------------------------------------------------------------------------------------------------------

constexpr int CAND_BLOCKS = 128;

static __device__ __forceinline__ bool better(float v, int i, float bv, int bi) { return v > bv || (v == bv && i < bi); }

__global__ void __launch_bounds__(256) cand1_k(const float* __restrict__ logits, int topk, float* __restrict__ sv, int* __restrict__ si) {
    pdl_wait();
    const int t = blockIdx.y, b = blockIdx.x;
    const float* row = logits + (size_t)t * VOCAB + (size_t)b * 1024;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; j++) v[j] = row[threadIdx.x + 256 * j];
    __shared__ float wv[2][8];
    __shared__ int wi[2][8];
    for (int r = 0; r < topk; r++) {
        float bv = -3.4e38f;
        int bj = 0;
#pragma unroll
        for (int j = 0; j < 4; j++) if (v[j] > bv) { bv = v[j]; bj = j; }
        int bi = threadIdx.x + 256 * bj;
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) {
            float ov = __shfl_xor_sync(0xffffffffu, bv, m);
            int oi = __shfl_xor_sync(0xffffffffu, bi, m);
            if (better(ov, oi, bv, bi)) { bv = ov; bi = oi; }
        }
        int slot = r & 1;
        if ((threadIdx.x & 31) == 0) { wv[slot][threadIdx.x >> 5] = bv; wi[slot][threadIdx.x >> 5] = bi; }
        __syncthreads();
        float fv = wv[slot][0];
        int fi = wi[slot][0];
#pragma unroll
        for (int w = 1; w < 8; w++) if (better(wv[slot][w], wi[slot][w], fv, fi)) { fv = wv[slot][w]; fi = wi[slot][w]; }
        if (threadIdx.x == 0) {
            size_t o = ((size_t)t * CAND_BLOCKS + b) * CAND + r;
            sv[o] = fv;
            si[o] = b * 1024 + fi;
        }
        if ((fi & 255) == threadIdx.x) v[fi >> 8] = -3.4e38f;
    }
    pdl_trigger();
}

__global__ void __launch_bounds__(256) cand2_k(int topk, const float* __restrict__ sv, const int* __restrict__ si,
                                                float* __restrict__ vals, int* __restrict__ ids) {
    pdl_wait();
    const int t = blockIdx.x;
    // CAND_BLOCKS * topk candidates, 32 per thread at most
    const int total = CAND_BLOCKS * topk;
    float v[32];
    int id[32];
#pragma unroll
    for (int j = 0; j < 32; j++) {
        int k = threadIdx.x + 256 * j;
        if (k < total) {
            int blk = k / topk, r = k % topk;
            size_t o = ((size_t)t * CAND_BLOCKS + blk) * CAND + r;
            v[j] = sv[o];
            id[j] = si[o];
        } else {
            v[j] = -3.4e38f;
            id[j] = 0x7fffffff;
        }
    }
    __shared__ float wv[2][8];
    __shared__ int wi[2][8];
    for (int r = 0; r < topk; r++) {
        float bv = -3.4e38f;
        int bi = 0x7fffffff, bj = 0;
#pragma unroll
        for (int j = 0; j < 32; j++) if (better(v[j], id[j], bv, bi)) { bv = v[j]; bi = id[j]; bj = j; }
        float lv = bv; int li = bi;
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) {
            float ov = __shfl_xor_sync(0xffffffffu, bv, m);
            int oi = __shfl_xor_sync(0xffffffffu, bi, m);
            if (better(ov, oi, bv, bi)) { bv = ov; bi = oi; }
        }
        int slot = r & 1;
        if ((threadIdx.x & 31) == 0) { wv[slot][threadIdx.x >> 5] = bv; wi[slot][threadIdx.x >> 5] = bi; }
        __syncthreads();
        float fv = wv[slot][0];
        int fi = wi[slot][0];
#pragma unroll
        for (int w = 1; w < 8; w++) if (better(wv[slot][w], wi[slot][w], fv, fi)) { fv = wv[slot][w]; fi = wi[slot][w]; }
        if (threadIdx.x == 0) {
            vals[(size_t)t * CAND + r] = fv;
            ids[(size_t)t * CAND + r] = fi;
        }
        // retire the winner from the thread that holds it
        if (li == fi && lv == fv) {
#pragma unroll
            for (int j = 0; j < 32; j++) if (j == bj) { v[j] = -3.4e38f; id[j] = 0x7fffffff; }
        }
    }
    pdl_trigger();
}

// The same for chosen rows among only the tokens a structured-output mask allows (bit t of a row's `words` u32s): stage one
// of row sel[i] reads its mask from masks + i * words, a disallowed token counting as -infinity.
__global__ void __launch_bounds__(256) cand1m_k(const float* __restrict__ logits, const int* __restrict__ sel, const uint32_t* __restrict__ masks,
                                                 int words, int topk, float* __restrict__ sv, int* __restrict__ si) {
    pdl_wait();
    const int i = blockIdx.y, b = blockIdx.x;
    const float* row = logits + (size_t)sel[i] * VOCAB + (size_t)b * 1024;
    const uint32_t* mk = masks + (size_t)i * words;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const int tok = b * 1024 + threadIdx.x + 256 * j;
        const bool ok = (tok >> 5) < words && ((mk[tok >> 5] >> (tok & 31)) & 1u);
        v[j] = ok ? row[threadIdx.x + 256 * j] : -3.4e38f;
    }
    __shared__ float wv[2][8];
    __shared__ int wi[2][8];
    for (int r = 0; r < topk; r++) {
        float bv = -3.4e38f;
        int bj = 0;
#pragma unroll
        for (int j = 0; j < 4; j++) if (v[j] > bv) { bv = v[j]; bj = j; }
        int bi = threadIdx.x + 256 * bj;
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) {
            float ov = __shfl_xor_sync(0xffffffffu, bv, m);
            int oi = __shfl_xor_sync(0xffffffffu, bi, m);
            if (better(ov, oi, bv, bi)) { bv = ov; bi = oi; }
        }
        int slot = r & 1;
        if ((threadIdx.x & 31) == 0) { wv[slot][threadIdx.x >> 5] = bv; wi[slot][threadIdx.x >> 5] = bi; }
        __syncthreads();
        float fv = wv[slot][0];
        int fi = wi[slot][0];
#pragma unroll
        for (int w = 1; w < 8; w++) if (better(wv[slot][w], wi[slot][w], fv, fi)) { fv = wv[slot][w]; fi = wi[slot][w]; }
        if (threadIdx.x == 0) {
            size_t o = ((size_t)i * CAND_BLOCKS + b) * CAND + r;
            sv[o] = fv;
            si[o] = b * 1024 + fi;
        }
        if ((fi & 255) == threadIdx.x) v[fi >> 8] = -3.4e38f;
    }
    pdl_trigger();
}

void k_candidates_masked(cudaStream_t s, const float* logits, const int* sel, const uint32_t* masks, int words, int n, int topk,
                         float* vals, int* ids, float* sv, int* si) {
    if (topk < 1) topk = 1;
    if (topk > CAND) topk = CAND;
    launch_pdl(cand1m_k, dim3(CAND_BLOCKS, n), 256, 0, s, logits, sel, masks, words, topk, sv, si);
    launch_pdl(cand2_k, n, 256, 0, s, topk, sv, si, vals, ids);
}

void k_candidates(cudaStream_t s, const float* logits, int T, int topk, float* vals, int* ids, float* sv, int* si) {
    if (topk < 1) topk = 1;
    if (topk > CAND) topk = CAND;
    launch_pdl(cand1_k, dim3(CAND_BLOCKS, T), 256, 0, s, logits, topk, sv, si);
    launch_pdl(cand2_k, T, 256, 0, s, topk, sv, si, vals, ids);
}

// Replay the first n rows of a verify window from the pre-window state: grid.y selects the layer.
__global__ void __launch_bounds__(256) scan_replay_k(const ReplayLayer* __restrict__ layers, int seq, int slot, int row0, int n) {
    pdl_wait();
    const ReplayLayer L = layers[blockIdx.y];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int h = blockIdx.x >> 3;
    const int d = (blockIdx.x & 7) * 8 + warp;
    const int g = h >> 4;
    const size_t off = ((size_t)h * 64 + d) * 128 + 4 * lane;
    float* pool = L.ssm + (size_t)seq * 2 * SSM_ELEMS;
    float4 st = *reinterpret_cast<const float4*>(pool + (size_t)slot * SSM_ELEMS + off);
    const float A = -__expf(L.A_log[h]);
    const float bias = L.dt_bias[h];
    for (int t = row0; t < row0 + n; t++) {
        float v = L.hist_dt[t * 128 + h] + bias;
        float sp = v > 20.f ? v : log1pf(__expf(v));
        float dA = __expf(sp * A);
        const float* row = L.hist_x + (size_t)t * 10240;
        float sx = sp * row[h * 64 + d];
        float4 B = *reinterpret_cast<const float4*>(row + 8192 + g * 128 + 4 * lane);
        st.x = st.x * dA + sx * B.x;
        st.y = st.y * dA + sx * B.y;
        st.z = st.z * dA + sx * B.z;
        st.w = st.w * dA + sx * B.w;
    }
    *reinterpret_cast<float4*>(pool + (size_t)(slot ^ 1) * SSM_ELEMS + off) = st;
    pdl_trigger();
}
void k_scan_replay(cudaStream_t s, const ReplayLayer* layers, int nlayers, int seq, int slot, int row0, int n) {
    launch_pdl(scan_replay_k, dim3(128 * 8, nlayers), 256, 0, s, layers, seq, slot, row0, n);
}

// Lazy state: the first n saved rows of hist slot hslot of the sequence, folded into its one state slot in place (each
// warp reads and writes its own entries alone).
__global__ void __launch_bounds__(256) scan_fold_k(const ReplayLayer* __restrict__ layers, int seq, int hslot, int n, int win) {
    pdl_wait();
    const ReplayLayer L = layers[blockIdx.y];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int h = blockIdx.x >> 3;
    const int d = (blockIdx.x & 7) * 8 + warp;
    const int g = h >> 4;
    const size_t off = ((size_t)h * 64 + d) * 128 + 4 * lane;
    float* st_p = L.ssm + (size_t)seq * SSM_ELEMS + off;
    float4 st = *reinterpret_cast<const float4*>(st_p);
    const float A = -__expf(L.A_log[h]);
    const float bias = L.dt_bias[h];
    const size_t base = ((size_t)seq * 2 + hslot) * win;
    for (int r = 0; r < n; r++) {
        float v = L.hist_dt[(base + r) * 128 + h] + bias;
        float sp = v > 20.f ? v : log1pf(__expf(v));
        float dA = __expf(sp * A);
        const float* row = L.hist_x + (base + r) * HIST_ROW;
        float sx = sp * row[h * 64 + d];
        float4 B = *reinterpret_cast<const float4*>(row + 8192 + g * 128 + 4 * lane);
        st.x = st.x * dA + sx * B.x;
        st.y = st.y * dA + sx * B.y;
        st.z = st.z * dA + sx * B.z;
        st.w = st.w * dA + sx * B.w;
    }
    *reinterpret_cast<float4*>(st_p) = st;
    pdl_trigger();
}
void k_scan_fold(cudaStream_t s, const ReplayLayer* layers, int nlayers, int seq, int hslot, int n, int win) {
    if (n > 0) launch_pdl(scan_fold_k, dim3(128 * 8, nlayers), 256, 0, s, layers, seq, hslot, n, win);
}
