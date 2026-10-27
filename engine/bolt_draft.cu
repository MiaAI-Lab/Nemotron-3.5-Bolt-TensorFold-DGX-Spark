// LiLiCorr draft: a 5-layer DFlash backbone (NVFP4 MLP, grouped conv, sliding-window sink attention) over
// the target's hidden states, plus the candidate-lattice head. The target's embeddings and LM head are
// borrowed. One block is the anchor token and seven mask slots; the output is seven draft tokens.

#include "bolt_draft.h"

#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define CK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { fprintf(stderr, "bolt draft: %s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(_e)); } } while (0)

namespace {

constexpr int H = 4096, HD = 128, HEADS = 32, KVH = 2, INTER = 12288;
constexpr int BLOCK = 8, SLOTS = 7, TOPK = 8, MASK = 990, NODES = SLOTS * TOPK;
constexpr int LH = 1024, LHEADS = 8, LDIM = 128;
constexpr int DL = 5, RING = 2048, WINDOW = 1024;
constexpr float EPS = 1e-6f, ROPE = 10000.f, VEPS = 1e-4f;

struct Info { std::string dtype; std::vector<long> shape; size_t off = 0, len = 0; };

struct File {
    void* map = nullptr;
    size_t size = 0;
    std::unordered_map<std::string, Info> tab;
    const uint8_t* data_base = nullptr;
};

bool parse(const uint8_t* file, size_t n, std::unordered_map<std::string, Info>& out) {
    if (n < 8) return false;
    uint64_t hlen; memcpy(&hlen, file, 8);
    if (8 + hlen > n) return false;
    std::string h((const char*)file + 8, hlen);
    size_t i = 0;
    while (true) {
        size_t k = h.find("\"dtype\"", i);
        if (k == std::string::npos) break;
        size_t name_end = h.rfind('"', k - 2);
        size_t name_start = h.rfind('"', name_end - 1);
        if (name_start == std::string::npos) break;
        std::string name = h.substr(name_start + 1, name_end - name_start - 1);
        Info info;
        size_t dq = h.find('"', h.find("dtype", k) + 6);
        size_t dq2 = h.find('"', dq + 1);
        info.dtype = h.substr(dq + 1, dq2 - dq - 1);
        size_t sb = h.find('[', dq2);
        size_t se = h.find(']', sb);
        std::string sh = h.substr(sb + 1, se - sb - 1);
        long v = 0; bool any = false;
        for (char c : sh) {
            if (c >= '0' && c <= '9') { v = v * 10 + (c - '0'); any = true; }
            else if (any) { info.shape.push_back(v); v = 0; any = false; }
        }
        if (any) info.shape.push_back(v);
        size_t ob = h.find("data_offsets", se);
        size_t br = h.find('[', ob);
        unsigned long long a = 0, b = 0;
        sscanf(h.c_str() + br, "[%llu, %llu]", &a, &b);
        info.off = a;
        info.len = b - a;
        out[name] = info;
        i = br;
    }
    return !out.empty();
}

bool open_file(const char* path, File& f) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return false;
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return false; }
    void* p = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (p == MAP_FAILED) return false;
    f.map = p; f.size = st.st_size;
    if (!parse((const uint8_t*)p, f.size, f.tab)) return false;
    uint64_t hs; memcpy(&hs, p, 8);
    f.data_base = (const uint8_t*)p + 8 + hs;
    return true;
}

bool get(File& f, const std::string& name, const uint8_t*& data, Info& info) {
    auto it = f.tab.find(name);
    if (it == f.tab.end()) { fprintf(stderr, "bolt draft: missing %s\n", name.c_str()); return false; }
    info = it->second;
    data = f.data_base + info.off;
    return true;
}

float bf2f(uint16_t v) { uint32_t u = (uint32_t)v << 16; float f; memcpy(&f, &u, 4); return f; }

void* up_raw(const uint8_t* data, size_t n) {
    void* d = nullptr;
    CK(cudaMalloc(&d, n));
    CK(cudaMemcpy(d, data, n, cudaMemcpyHostToDevice));
    return d;
}

float* tensor_f32(File& f, const std::string& name, size_t expect = 0) {
    const uint8_t* d; Info i;
    if (!get(f, name, d, i)) return nullptr;
    size_t n = 1; for (long s : i.shape) n *= (size_t)s;
    if (expect && n != expect) fprintf(stderr, "bolt draft: %s has %zu values, wanted %zu\n", name.c_str(), n, expect);
    std::vector<float> h(n);
    const uint16_t* s = (const uint16_t*)d;
    for (size_t k = 0; k < n; k++) h[k] = bf2f(s[k]);
    float* dev = nullptr;
    CK(cudaMalloc(&dev, n * 4));
    CK(cudaMemcpy(dev, h.data(), n * 4, cudaMemcpyHostToDevice));
    return dev;
}

TBF tensor_bf16(File& f, const std::string& name) {
    TBF w{};
    const uint8_t* d; Info i;
    if (!get(f, name, d, i)) return w;
    w.n = (int)i.shape[0];
    w.k = i.shape.size() > 1 ? (int)i.shape[1] : 1;
    void* raw = up_raw(d, i.len);
    void* wt;
    CK(cudaMalloc(&wt, tile_bytes_bf16(w.n, w.k)));
    tile_bf16(0, (const uint16_t*)raw, wt, w.n, w.k);
    CK(cudaDeviceSynchronize());
    cudaFree(raw);
    w.w = (const uint4*)wt;
    return w;
}

TNV4 tensor_nv4(File& f, const std::string& name) {
    TNV4 w{};
    const uint8_t* d; Info i;
    if (!get(f, name + ".weight", d, i)) return w;
    w.n = (int)i.shape[0];
    w.k = (int)i.shape[1] * 2;
    void* rq = up_raw(d, i.len);
    const uint8_t* sd; Info si;
    get(f, name + ".weight_scale", sd, si);
    void* rs = up_raw(sd, si.len);
    const uint8_t* gd; Info gi;
    get(f, name + ".weight_scale_2", gd, gi);
    memcpy(&w.s2, gd, 4);
    void *qt, *sct;
    CK(cudaMalloc(&qt, tile_bytes_nv4(w.n, w.k)));
    CK(cudaMalloc(&sct, tile_scale_bytes_nv4(w.n, w.k)));
    tile_nv4(0, (const uint8_t*)rq, (const uint8_t*)rs, qt, sct, w.n, w.k);
    CK(cudaDeviceSynchronize());
    cudaFree(rq); cudaFree(rs);
    w.q = (const uint4*)qt; w.sc = (const uint2*)sct;
    return w;
}

// A bf16 matrix of the checkpoint as the kernels read it: Q8_0 tiles (BOLT_DRAFT_Q8=1, the default), else bf16 tiles.
struct Lin { TBF bf{}; TQ8 q8{}; };

bool draft_q8() {
    static int on = -1;
    if (on < 0) { const char* e = getenv("BOLT_DRAFT_Q8"); on = e ? atoi(e) != 0 : 1; }
    return on != 0;
}

// bf16 -> Q8_0 on the host (ggml's blocks: a half scale and 32 int8 a block of 32 along k), then the target's tile layout.
// The drafter's attention and kernel projections read half the bytes a proposal; the outputs they draft are checked by
// the target, so the rounding changes what is proposed, never what is served.
Lin tensor_lin(File& f, const std::string& name) {
    Lin w;
    const uint8_t* d; Info i;
    if (!get(f, name, d, i)) return w;
    if (!draft_q8() || i.shape.size() != 2 || i.shape[1] % 32 != 0) { w.bf = tensor_bf16(f, name); return w; }
    const int n = (int)i.shape[0], k = (int)i.shape[1], KB = k / 32;
    std::vector<uint8_t> raw((size_t)n * KB * 34);
    const uint16_t* src = (const uint16_t*)d;
    for (int row = 0; row < n; row++) for (int kb = 0; kb < KB; kb++) {
        float v[32], amax = 0.f;
        for (int j = 0; j < 32; j++) { v[j] = bf2f(src[(size_t)row * k + kb * 32 + j]); amax = fmaxf(amax, fabsf(v[j])); }
        const float dsc = amax / 127.f, inv = dsc > 0.f ? 1.f / dsc : 0.f;
        uint8_t* blk = &raw[((size_t)row * KB + kb) * 34];
        const __half hd = __float2half(dsc);
        memcpy(blk, &hd, 2);
        for (int j = 0; j < 32; j++) {
            int q = (int)lrintf(v[j] * inv);
            q = q < -128 ? -128 : q > 127 ? 127 : q;
            blk[2 + j] = (uint8_t)(int8_t)q;
        }
    }
    w.q8.n = n; w.q8.k = k;
    void* r = up_raw(raw.data(), raw.size());
    void *qt, *dt;
    CK(cudaMalloc(&qt, tile_bytes_q8(n, k)));
    CK(cudaMalloc(&dt, tile_scale_bytes_q8(n, k)));
    tile_q8(0, (const uint8_t*)r, qt, dt, n, k);
    CK(cudaDeviceSynchronize());
    cudaFree(r);
    w.q8.q = (const uint4*)qt; w.q8.d = (const uint32_t*)dt;
    return w;
}

struct DLayer {
    float *in_norm = nullptr, *post_norm = nullptr, *qn = nullptr, *kn = nullptr, *sink = nullptr, *base_a = nullptr, *base_m = nullptr;
    Lin q{}, k{}, v{}, o{}, att_kp{}, mlp_kp{};
    TNV4 gate{}, up{}, down{};
};

struct LLayer {
    float *an = nullptr, *mn = nullptr, *qkv_b = nullptr, *out_b = nullptr, *up_b = nullptr, *down_b = nullptr;
    TBF qkv{};
    TNV4 out{}, up{}, down{};
};

}  // namespace

struct Draft {
    const uint16_t* embed = nullptr;
    TQ8 head{};
    TNV4 fc{};
    float *hid_norm = nullptr, *final_norm = nullptr;
    DLayer L[DL];
    LLayer ll[2];
    TNV4 tok_p{}, pass_p{}, ctx_p{};
    float *tok_b = nullptr, *pass_b = nullptr, *ctx_b = nullptr;
    TBF factor{}, out_h{}, in_h{}, anc_h{}, feat_down{};
    const uint16_t* feat_up = nullptr;  // [1024][5] bf16, used by a small dedicated kernel
    float *factor_b = nullptr, *out_b = nullptr, *in_b = nullptr, *anc_b = nullptr, *feat_up_b = nullptr, *feat_down_b = nullptr;
    float *feat_nw = nullptr, *feat_nb = nullptr, *out_norm = nullptr, *anc_norm = nullptr;
    float *slot = nullptr, *rank = nullptr, *abias = nullptr;
    float* acat = nullptr;  // [nb][H] anchors of the streams in the call

    // One context per sequence: the K/V rings of the drafter's layers, the anchor row, the next position expected.
    struct SeqCtx { float *ck[DL] = {}, *cv[DL] = {}; float* anchor = nullptr; int clen = 0; };
    std::vector<SeqCtx> sq;
    int nb = 1;  // streams one propose call drafts at most
    // device tables of the call in flight
    const float** ckp = nullptr;   // [DL][nb] ring pointers
    const float** cvp = nullptr;
    int *clen_d = nullptr, *pos_d = nullptr, *anc_d = nullptr;   // [nb]
    const float** ckp_h = nullptr; const float** cvp_h = nullptr;  // pinned mirrors
    int *clen_h = nullptr, *pos_h = nullptr, *anc_h_tok = nullptr;
    // ingest of several sequences at once
    const float** ring_k = nullptr; const float** ring_v = nullptr;   // [DL][nb] every sequence's ring, fixed
    int *irow_seq = nullptr, *irow_pos = nullptr;                      // [TMAX]
    int *irow_seq_h = nullptr, *irow_pos_h = nullptr;
    // scratch
    float *x = nullptr, *r = nullptr, *h = nullptr, *z = nullptr, *y = nullptr, *q = nullptr, *k = nullptr, *v = nullptr;
    float *g = nullptr, *u = nullptr, *hid = nullptr, *dlog = nullptr;
    float *ictx = nullptr, *ik = nullptr, *iv = nullptr;
    float *csv = nullptr, *lse = nullptr, *tokstate = nullptr, *passstate = nullptr, *feat = nullptr, *fh = nullptr;
    float *nodes = nullptr, *ln = nullptr, *lqkv = nullptr, *lqkv3 = nullptr, *lo = nullptr, *cat = nullptr;
    float *edges_out = nullptr, *edges_in = nullptr, *aout = nullptr, *astate = nullptr, *gathered = nullptr, *tmp = nullptr;
    float* cvals = nullptr;
    int *cids = nullptr, *csi = nullptr;
    int* out_dev = nullptr;
    int* out_host = nullptr;   // [nb][SLOTS]
    // the call's anchor rows, gathered on the device from a pinned pointer table (a graph holds no changing addresses)
    const float** ancp = nullptr; const float** ancp_h = nullptr;   // [nb]
    // BOLT_DRAFT_GRAPH (default on): the proposal's launches for B streams, captured once and replayed as one graph
    std::unordered_map<int, cudaGraphExec_t> graphs;
    bool graph_off = false;   // a capture failed: propose eagerly from then on
};

namespace {

// ---------------------------------------------------------------------------------------------------------
// kernels
// ---------------------------------------------------------------------------------------------------------

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}
__device__ __forceinline__ float siluf(float v) { return v / (1.f + __expf(-v)); }
__device__ __forceinline__ float bf2f_dev(uint16_t v) { return __uint_as_float((uint32_t)v << 16); }

// Sum across the block; safe to call repeatedly with the same scratch.
__device__ float block_sum(float v, float* sm) {
    v = warp_sum(v);
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) sm[warp] = v;
    __syncthreads();
    float t = 0.f;
    for (int i = 0; i < (int)(blockDim.x >> 5); i++) t += sm[i];
    return t;
}

__global__ void rms_rows_k(const float* x, const float* w, float* y, int n, float eps) {
    __shared__ float sm[32];
    const float* xr = x + (size_t)blockIdx.x * n;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xr[i] * xr[i];
    ss = block_sum(ss, sm);
    float inv = rsqrtf(ss / (float)n + eps);
    float* yr = y + (size_t)blockIdx.x * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = xr[i] * inv * (w ? w[i] : 1.f);
}

__global__ void add_k(float* y, const float* a, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += a[i];
}

__global__ void bias_k(float* y, const float* b, int n, int rows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * n) y[i] += b[i % n];
}

__global__ void silu_k(float* y, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = siluf(y[i]);
}

__global__ void silu_mul_k(float* y, const float* gate, const float* up, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = siluf(gate[i]) * up[i];
}

// Block rows of every stream: [anchor, mask x 7].
__global__ void block_embed_k(const uint16_t* table, const int* anchors, float* y) {
    int r = blockIdx.x, row = r % BLOCK;
    int id = row == 0 ? anchors[r / BLOCK] : MASK;
    if ((unsigned)id >= (unsigned)VOCAB) id = 0;
    const uint16_t* src = table + (size_t)id * H;
    for (int i = threadIdx.x; i < H; i += blockDim.x) y[(size_t)r * H + i] = bf2f_dev(src[i]);
}

// Embeddings of each stream's 56 candidates (slot-major, rank-minor) from the candidate id table; the candidates of
// slot s of stream b are row b * BLOCK + 1 + s.
__global__ void gather_cand_k(const uint16_t* table, const int* cids, float* y) {
    int node = blockIdx.x, b = node / NODES, li = node % NODES;
    int id = cids[((size_t)(b * BLOCK + 1 + li / TOPK)) * CAND + li % TOPK];
    if ((unsigned)id >= (unsigned)VOCAB) id = 0;
    const uint16_t* src = table + (size_t)id * H;
    for (int i = threadIdx.x; i < H; i += blockDim.x) y[(size_t)node * H + i] = bf2f_dev(src[i]);
}

// Grouped dynamic depthwise conv over the rows of each block (BLOCK rows a stream): tap 0 is the row itself, tap 1 the
// previous row of the same block. delta is [row][side][tap][group] from the kernel projection.
__global__ void gconv_k(const float* x, const float* base, const float* delta, float* y, int T, int side) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * H) return;
    int t = i / H, c = i % H, g = c / 16;
    const float* b = base + side * 2 * H;
    float acc = 0.f;
    for (int tap = 0; tap < 2; tap++) {
        if ((t % BLOCK) - tap < 0) continue;
        int src = t - tap;
        float coef = b[tap * H + c] + delta[(((size_t)t * 2 + side) * 2 + tap) * 256 + g];
        acc += coef * x[(size_t)src * H + c];
    }
    y[i] = acc;
}

// Per-head RMS norm then NeoX rope. One block of 64 threads per (row, head).
// With posv, row t sits at posv[t / BLOCK] + t % BLOCK (a call over streams); otherwise at pos0 + t.
// With rowpos, row t sits at rowpos[t] (rows of several sequences' ingest).
__global__ void qk_norm_rope_k(float* x, const float* w, int heads, int pos0, const int* posv = nullptr, const int* rowpos = nullptr) {
    int t = blockIdx.x, h = blockIdx.y;
    if (h >= heads) return;
    float* r = x + ((size_t)t * heads + h) * HD;
    __shared__ float sm[8];
    float v0 = r[threadIdx.x], v1 = r[threadIdx.x + 64];
    float ss = block_sum(v0 * v0 + v1 * v1, sm);
    float inv = rsqrtf(ss / (float)HD + EPS);
    v0 = v0 * inv * w[threadIdx.x];
    v1 = v1 * inv * w[threadIdx.x + 64];
    float ang = (float)(rowpos ? rowpos[t] : posv ? posv[t / BLOCK] + t % BLOCK : pos0 + t) * powf(ROPE, -2.f * threadIdx.x / (float)HD);
    float c = cosf(ang), s = sinf(ang);
    r[threadIdx.x] = v0 * c - v1 * s;
    r[threadIdx.x + 64] = v0 * s + v1 * c;
}

// Block attention with a sink. Context keys come from the ring (positions [lo, clen)), the block's own
// keys are all visible. One block per (head, query row).
__global__ void __launch_bounds__(128) dattn_k(const float* q, const float* bk_all, const float* bv_all, const float* const* ckp,
                                               const float* const* cvp, const float* sink, float* out, const int* clen_v, const int* pos_v) {
    int h = blockIdx.x, row = blockIdx.y;
    const int b = row / BLOCK, qi = row % BLOCK;
    const float* ck = ckp[b]; const float* cv = cvp[b];
    const float* bk = bk_all + (size_t)b * BLOCK * KVH * HD; const float* bv = bv_all + (size_t)b * BLOCK * KVH * HD;
    const int clen = clen_v[b], pos = pos_v[b];
    int kh = h / (HEADS / KVH);
    __shared__ float qs[HD];
    __shared__ float sc[WINDOW + BLOCK + 8];
    __shared__ float red[32];
    qs[threadIdx.x] = q[((size_t)row * HEADS + h) * HD + threadIdx.x] * rsqrtf((float)HD);
    __syncthreads();
    int lo = pos + qi - (WINDOW - 1);
    if (lo < 0) lo = 0;
    int nctx = clen > lo ? clen - lo : 0;
    if (nctx > WINDOW + 8) nctx = WINDOW + 8;
    lo = clen - nctx;
    int total = nctx + BLOCK;
    float m = sink[h];
    for (int j = threadIdx.x; j < total; j += 128) {
        const float* kk = j < nctx ? ck + ((size_t)((lo + j) & (RING - 1)) * KVH + kh) * HD : bk + ((size_t)(j - nctx) * KVH + kh) * HD;
        float dot = 0.f;
#pragma unroll 8
        for (int d = 0; d < HD; d++) dot += qs[d] * kk[d];
        sc[j] = dot;
        m = fmaxf(m, dot);
    }
    {
        float v = m;
        for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
        __syncthreads();
        if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
        __syncthreads();
        v = red[0];
        for (int i = 1; i < 4; i++) v = fmaxf(v, red[i]);
        m = v;
        __syncthreads();
    }
    float den = 0.f;
    for (int j = threadIdx.x; j < total; j += 128) {
        float e = __expf(sc[j] - m);
        sc[j] = e;
        den += e;
    }
    den = block_sum(den, red) + __expf(sink[h] - m);
    __syncthreads();
    float acc = 0.f;
    for (int j = 0; j < total; j++) {
        const float* vv = j < nctx ? cv + ((size_t)((lo + j) & (RING - 1)) * KVH + kh) * HD : bv + ((size_t)(j - nctx) * KVH + kh) * HD;
        acc += sc[j] * vv[threadIdx.x];
    }
    out[((size_t)row * HEADS + h) * HD + threadIdx.x] = acc / den;
}

// Block attention with a sink, tiled: a CTA takes 4 query heads of one KV head for the 8 rows of a stream's block (32
// queries) and streams the keys through shared memory in tiles of 64, so every key is read once for all of them.
// Visible keys of row qi: the ring's positions [lo, clen) (lo slides with the row, as in dattn_k) and the block's own
// 8 keys. Online softmax; the sink only joins the denominator.
constexpr int DA_KT = 64, DA_KS = 132;
constexpr int DA_SMEM = (32 * HD + DA_KT * DA_KS + DA_KT * HD + 32 * DA_KT) * 4;
__global__ void __launch_bounds__(256) dattn2_k(const float* __restrict__ q, const float* __restrict__ bk_all, const float* __restrict__ bv_all,
                                                const float* const* ckp, const float* const* cvp, const float* __restrict__ sink,
                                                float* __restrict__ out, const int* clen_v, const int* pos_v) {
    extern __shared__ float sm[];
    float* qs = sm;                       // [32][HD]
    float* Ks = qs + 32 * HD;             // [DA_KT][DA_KS]
    float* Vs = Ks + DA_KT * DA_KS;       // [DA_KT][HD]
    float* Ps = Vs + DA_KT * HD;          // [32][DA_KT]
    const int hg = blockIdx.x, kh = blockIdx.y, b = blockIdx.z;
    const int tid = threadIdx.x, u = tid >> 3, jc = tid & 7;   // query u = row * 4 + head-in-group; jc: key lane
    const int row = u >> 2, hl = u & 3, h = kh * (HEADS / KVH) + hg * 4 + hl;
    const float* ck = ckp[b]; const float* cv = cvp[b];
    const float* bk = bk_all + (size_t)b * BLOCK * KVH * HD; const float* bv = bv_all + (size_t)b * BLOCK * KVH * HD;
    const int clen = clen_v[b], pos = pos_v[b];
    // visible ring range of each row
    auto lo_of = [&](int r) {
        int lo = pos + r - (WINDOW - 1);
        if (lo < 0) lo = 0;
        int nctx = clen > lo ? clen - lo : 0;
        if (nctx > WINDOW + 8) nctx = WINDOW + 8;
        return clen - nctx;
    };
    const int lo_min = lo_of(0), my_lo = lo_of(row);
    for (int i = tid; i < 32 * HD; i += 256) {
        int uu = i / HD, d = i % HD;
        int rr = uu >> 2, hh = kh * (HEADS / KVH) + hg * 4 + (uu & 3);
        qs[i] = q[((size_t)(b * BLOCK + rr) * HEADS + hh) * HD + d] * rsqrtf((float)HD);
    }
    float m = sink[h], l = 0.f;
    float acc[16];
#pragma unroll
    for (int i = 0; i < 16; i++) acc[i] = 0.f;
    const int total = (clen - lo_min) + BLOCK;   // ring keys of the widest row, then the block's own
    for (int t0 = 0; t0 < total; t0 += DA_KT) {
        __syncthreads();
        for (int i = tid; i < DA_KT * (HD / 4); i += 256) {
            int j = i / (HD / 4), d4 = (i % (HD / 4)) * 4;
            int g = t0 + j;
            float4 kv = make_float4(0, 0, 0, 0), vv = kv;
            if (g < total) {
                int nring = clen - lo_min;
                if (g < nring) {
                    size_t o = ((size_t)((lo_min + g) & (RING - 1)) * KVH + kh) * HD + d4;
                    kv = *reinterpret_cast<const float4*>(ck + o);
                    vv = *reinterpret_cast<const float4*>(cv + o);
                } else {
                    size_t o = ((size_t)(g - nring) * KVH + kh) * HD + d4;
                    kv = *reinterpret_cast<const float4*>(bk + o);
                    vv = *reinterpret_cast<const float4*>(bv + o);
                }
            }
            *reinterpret_cast<float4*>(&Ks[j * DA_KS + d4]) = kv;
            *reinterpret_cast<float4*>(&Vs[j * HD + d4]) = vv;
        }
        __syncthreads();
        // scores: key j = jj * 8 + jc of the tile
        float sc[8];
        float tmax = -1e30f;
#pragma unroll
        for (int jj = 0; jj < 8; jj++) {
            const int j = jj * 8 + jc, g = t0 + j;
            float dot = 0.f;
            const float* kr = &Ks[j * DA_KS];
            const float* qr = &qs[u * HD];
#pragma unroll 16
            for (int d = 0; d < HD; d++) dot += qr[d] * kr[d];
            const int nring = clen - lo_min;
            bool vis = g < total && (g >= nring || lo_min + g >= my_lo);
            sc[jj] = vis ? dot : -1e30f;
            tmax = fmaxf(tmax, sc[jj]);
        }
        tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, 1));
        tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, 2));
        tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, 4));
        const float mn = fmaxf(m, tmax);
        const float corr = __expf(m - mn);
        float psum = 0.f;
#pragma unroll
        for (int jj = 0; jj < 8; jj++) {
            float p = sc[jj] > -1e29f ? __expf(sc[jj] - mn) : 0.f;
            Ps[u * DA_KT + jj * 8 + jc] = p;
            psum += p;
        }
        psum += __shfl_xor_sync(0xffffffffu, psum, 1);
        psum += __shfl_xor_sync(0xffffffffu, psum, 2);
        psum += __shfl_xor_sync(0xffffffffu, psum, 4);
        l = l * corr + psum;
        m = mn;
#pragma unroll
        for (int i = 0; i < 16; i++) acc[i] *= corr;
        __syncwarp();
        __syncthreads();
        const int d0 = jc * 16;
        for (int j = 0; j < DA_KT; j++) {
            const float p = Ps[u * DA_KT + j];
            const float4* vr = reinterpret_cast<const float4*>(&Vs[j * HD + d0]);
#pragma unroll
            for (int i = 0; i < 4; i++) {
                float4 v4 = vr[i];
                acc[4 * i] += p * v4.x; acc[4 * i + 1] += p * v4.y; acc[4 * i + 2] += p * v4.z; acc[4 * i + 3] += p * v4.w;
            }
        }
    }
    const float den = l + __expf(sink[h] - m);
    float* o = out + ((size_t)(b * BLOCK + row) * HEADS + h) * HD + jc * 16;
#pragma unroll
    for (int i = 0; i < 16; i++) o[i] = acc[i] / den;
}

__global__ void lse_k(const float* logits, float* lse) {
    const float* x = logits + (size_t)blockIdx.x * VOCAB;
    __shared__ float sm[32], mm[32];
    float m = -1e30f;
    for (int i = threadIdx.x; i < VOCAB; i += blockDim.x) m = fmaxf(m, x[i]);
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if ((threadIdx.x & 31) == 0) mm[threadIdx.x >> 5] = m;
    __syncthreads();
    m = mm[0];
    for (int i = 1; i < (int)(blockDim.x >> 5); i++) m = fmaxf(m, mm[i]);
    float s = 0.f;
    for (int i = threadIdx.x; i < VOCAB; i += blockDim.x) s += __expf(x[i] - m);
    s = block_sum(s, sm);
    if (threadIdx.x == 0) lse[blockIdx.x] = m + logf(s);
}

// features of candidate (slot, rank): [logp, p, logp - max, rank/7, is_top1], then LayerNorm over the five.
__global__ void features_k(const float* vals_all, const float* lse_all, const float* nw, const float* nb, float* feat_all) {
    const int b = blockIdx.x;
    const float* vals = vals_all + (size_t)(b * BLOCK + 1) * CAND;
    const float* lse = lse_all + b * BLOCK + 1;
    float* feat = feat_all + (size_t)b * NODES * 5;
    int node = threadIdx.x;
    if (node >= NODES) return;
    int s = node / TOPK, k = node % TOPK;
    float lp = vals[s * CAND + k] - lse[s];
    float mx = vals[s * CAND + 0] - lse[s];
    float f[5] = {lp, __expf(lp), lp - mx, (float)k / 7.f, k == 0 ? 1.f : 0.f};
    float mean = 0.f;
    for (int i = 0; i < 5; i++) mean += f[i];
    mean /= 5.f;
    float var = 0.f;
    for (int i = 0; i < 5; i++) var += (f[i] - mean) * (f[i] - mean);
    float inv = rsqrtf(var / 5.f + 1e-5f);
    for (int i = 0; i < 5; i++) feat[node * 5 + i] = (f[i] - mean) * inv * nw[i] + nb[i];
}

// feature MLP up_proj (5 -> 1024) with SiLU
__global__ void feat_up_k(const uint16_t* w, const float* b, const float* feat, float* y) {
    int node = blockIdx.x;
    for (int j = threadIdx.x; j < LH; j += blockDim.x) {
        float a = b[j];
        for (int i = 0; i < 5; i++) a += bf2f_dev(w[j * 5 + i]) * feat[node * 5 + i];
        y[(size_t)node * LH + j] = siluf(a);
    }
}

__global__ void nodes_k(const float* tok, const float* pass, const float* feat, const float* slot, const float* rank, float* nodes) {
    int i = blockIdx.x, b = i / NODES, li = i % NODES;
    int s = li / TOPK, k = li % TOPK;
    for (int c = threadIdx.x; c < LH; c += blockDim.x)
        nodes[(size_t)i * LH + c] = tok[(size_t)i * LH + c] + pass[(size_t)(b * BLOCK + 1 + s) * LH + c] + feat[(size_t)i * LH + c] +
                                    slot[(size_t)s * LH + c] + rank[(size_t)k * LH + c];
}

__global__ void split3_k(const float* qkv, float* q, float* k, float* v) {
    int row = blockIdx.x;
    for (int i = threadIdx.x; i < LH; i += blockDim.x) {
        q[(size_t)row * LH + i] = qkv[(size_t)row * 3 * LH + i];
        k[(size_t)row * LH + i] = qkv[(size_t)row * 3 * LH + LH + i];
        v[(size_t)row * LH + i] = qkv[(size_t)row * 3 * LH + 2 * LH + i];
    }
}

// Lattice self-attention over the 56 candidate nodes with a per-head bias matrix.
__global__ void lat_attn_k(const float* q, const float* k_all, const float* v_all, const float* bias, float* out) {
    int h = blockIdx.x, row = blockIdx.y;
    const int b = row / NODES, qi = row % NODES;
    const float* k = k_all + (size_t)b * NODES * LHEADS * LDIM;
    const float* v = v_all + (size_t)b * NODES * LHEADS * LDIM;
    __shared__ float qs[LDIM];
    __shared__ float sc[NODES];
    qs[threadIdx.x] = q[((size_t)row * LHEADS + h) * LDIM + threadIdx.x] * rsqrtf((float)LDIM);
    __syncthreads();
    if (threadIdx.x < NODES) {
        const float* kk = k + ((size_t)threadIdx.x * LHEADS + h) * LDIM;
        float dot = 0.f;
        for (int d = 0; d < LDIM; d++) dot += qs[d] * kk[d];
        sc[threadIdx.x] = dot + bias[((size_t)h * NODES + qi) * NODES + threadIdx.x];
    }
    __syncthreads();
    float m = -1e30f;
    for (int j = 0; j < NODES; j++) m = fmaxf(m, sc[j]);
    float den = 0.f, acc = 0.f;
    for (int j = 0; j < NODES; j++) {
        float e = __expf(sc[j] - m);
        den += e;
        acc += e * v[((size_t)j * LHEADS + h) * LDIM + threadIdx.x];
    }
    out[((size_t)row * LHEADS + h) * LDIM + threadIdx.x] = acc / den;
}

// [h, anchor, h * anchor] rows
__global__ void cat3_k(const float* h, const float* a_all, float* y) {
    int row = blockIdx.x;
    const float* a = a_all + (size_t)(row / NODES) * LH;
    for (int c = threadIdx.x; c < LH; c += blockDim.x) {
        float hv = h[(size_t)row * LH + c], av = a[c];
        float* o = y + (size_t)row * 3 * LH;
        o[c] = hv; o[LH + c] = av; o[2 * LH + c] = hv * av;
    }
}

// x / max(|x|, eps) on rows of LH
__global__ void l2norm_k(float* x, float eps) {
    __shared__ float sm[32];
    float* r = x + (size_t)blockIdx.x * LH;
    float ss = 0.f;
    for (int i = threadIdx.x; i < LH; i += blockDim.x) ss += r[i] * r[i];
    ss = block_sum(ss, sm);
    float inv = 1.f / fmaxf(sqrtf(ss), eps);
    for (int i = threadIdx.x; i < LH; i += blockDim.x) r[i] *= inv;
}

// Greedy walk through the lattice: slot 0 from the anchor, then each slot from the previous choice.
// BOLT_DRAFT_TOP1=1 takes rank 0 at every slot (plain DFlash) instead of the lattice's choice, to measure what the lattice adds
__global__ void walk_k(const float* anchor_out_all, const float* outv_all, const float* inv_all, const int* ids_all, int* result_all, int top1) {
    const int b = blockIdx.x;
    const float* anchor_out = anchor_out_all + (size_t)b * LH;
    const float* outv = outv_all + (size_t)b * NODES * LH;
    const float* inv = inv_all + (size_t)b * NODES * LH;
    const int* ids = ids_all + (size_t)(b * BLOCK + 1) * CAND;
    int* result = result_all + b * SLOTS;
    __shared__ float sc[TOPK];
    __shared__ int prev;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;  // 8 warps, one per rank
    if (threadIdx.x == 0) prev = 0;
    __syncthreads();
    for (int s = -1; s < SLOTS - 1; s++) {
        const float* a = s < 0 ? anchor_out : outv + ((size_t)s * TOPK + prev) * LH;
        const float* b = inv + ((size_t)(s + 1) * TOPK + warp) * LH;
        float acc = 0.f;
        for (int i = lane; i < LH; i += 32) acc += a[i] * b[i];
        acc = warp_sum(acc);
        if (lane == 0) sc[warp] = acc;
        __syncthreads();
        if (threadIdx.x == 0) {
            int best = 0;
            if (!top1) for (int k = 1; k < TOPK; k++) if (sc[k] > sc[best]) best = k;
            prev = best;
            result[s + 1] = ids[(s + 1) * CAND + best];
        }
        __syncthreads();
    }
}

// Rows of several sequences: row t goes to ring slot rowpos[t] of sequence rowseq[t] (skipped when negative).
__global__ void write_kv_multi_k(const float* k, const float* v, const float* const* ckp, const float* const* cvp,
                                 const int* rowseq, const int* rowpos) {
    int t = blockIdx.x;
    int q = rowseq[t];
    if (q < 0) return;
    float* ck = (float*)ckp[q];
    float* cv = (float*)cvp[q];
    for (int i = threadIdx.x; i < KVH * HD; i += blockDim.x) {
        size_t o = (size_t)(rowpos[t] & (RING - 1)) * KVH * HD + i;
        ck[o] = k[(size_t)t * KVH * HD + i];
        cv[o] = v[(size_t)t * KVH * HD + i];
    }
}

__global__ void write_kv_k(const float* k, const float* v, float* ck, float* cv, int pos0) {
    int t = blockIdx.x;
    for (int i = threadIdx.x; i < KVH * HD; i += blockDim.x) {
        size_t o = (size_t)((pos0 + t) & (RING - 1)) * KVH * HD + i;
        ck[o] = k[(size_t)t * KVH * HD + i];
        cv[o] = v[(size_t)t * KVH * HD + i];
    }
}

// ---------------------------------------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------------------------------------

cudaStream_t S;

bool dattn_tiled() { static bool on = getenv("BOLT_DATTN_OLD") == nullptr; return on; }

// BOLT_DPROF: events between the phases of draft_propose_multi, summed over calls and printed by draft_prof_report.
bool dprof_on() { static bool on = getenv("BOLT_DPROF") != nullptr; return on; }
std::vector<cudaEvent_t> g_dev;
std::vector<const char*> g_dname;
std::unordered_map<std::string, double> g_dacc;
void dmark(const char* name) {
    if (!dprof_on()) return;
    cudaEvent_t e;
    cudaEventCreate(&e);
    cudaEventRecord(e, S);
    g_dev.push_back(e);
    g_dname.push_back(name);
}
void dprof_collect() {
    for (size_t i = 1; i < g_dev.size(); i++) {
        float ms = 0;
        cudaEventElapsedTime(&ms, g_dev[i - 1], g_dev[i]);
        g_dacc[g_dname[i]] += ms;
    }
    for (auto e : g_dev) cudaEventDestroy(e);
    g_dev.clear();
    g_dname.clear();
}

void mm(const TBF& w, const float* x, int ldx, float* y, int ldy, int T, const float* bias = nullptr) {
    for (int base = 0; base < T; base += TMAX) {
        int nt = T - base < TMAX ? T - base : TMAX;
        mma_bf16(S, w, x + (size_t)base * ldx, ldx, y + (size_t)base * ldy, ldy, nt);
    }
    if (bias) bias_k<<<((size_t)T * w.n + 255) / 256, 256, 0, S>>>(y, bias, w.n, T);
}
void mm(const TNV4& w, const float* x, int ldx, float* y, int ldy, int T, const float* bias = nullptr) {
    for (int base = 0; base < T; base += TMAX) {
        int nt = T - base < TMAX ? T - base : TMAX;
        mma_nv4(S, w, x + (size_t)base * ldx, ldx, y + (size_t)base * ldy, ldy, nt);
    }
    if (bias) bias_k<<<((size_t)T * w.n + 255) / 256, 256, 0, S>>>(y, bias, w.n, T);
}
void mm(const Lin& w, const float* x, int ldx, float* y, int ldy, int T, const float* bias = nullptr) {
    if (!w.q8.q) { mm(w.bf, x, ldx, y, ldy, T, bias); return; }
    for (int base = 0; base < T; base += TMAX) {
        int nt = T - base < TMAX ? T - base : TMAX;
        mma_q8(S, w.q8, x + (size_t)base * ldx, ldx, y + (size_t)base * ldy, ldy, nt);
    }
    if (bias) bias_k<<<((size_t)T * w.q8.n + 255) / 256, 256, 0, S>>>(y, bias, w.q8.n, T);
}
void rms(const float* x, const float* w, float* y, int n, int rows, float eps = EPS) { rms_rows_k<<<rows, 256, 0, S>>>(x, w, y, n, eps); }
void addv(float* y, const float* a, size_t n) { add_k<<<(unsigned)((n + 255) / 256), 256, 0, S>>>(y, a, n); }
void silu_inplace(float* y, size_t n) { silu_k<<<(unsigned)((n + 255) / 256), 256, 0, S>>>(y, n); }

}  // namespace

Draft* draft_open(const char* dir, const uint16_t* embed, const TQ8& head, int nseq) {
    File f;
    std::string path = std::string(dir) + "/model.safetensors";
    if (!open_file(path.c_str(), f)) { fprintf(stderr, "bolt: LiLiCorr weights not found at %s\n", path.c_str()); return nullptr; }
    Draft* d = new Draft();
    d->embed = embed;
    d->head = head;
    d->fc = tensor_nv4(f, "fc");
    d->hid_norm = tensor_f32(f, "hidden_norm.weight", H);
    d->final_norm = tensor_f32(f, "norm.weight", H);
    for (int l = 0; l < DL; l++) {
        std::string p = "layers." + std::to_string(l);
        DLayer& a = d->L[l];
        a.in_norm = tensor_f32(f, p + ".input_layernorm.weight", H);
        a.post_norm = tensor_f32(f, p + ".post_attention_layernorm.weight", H);
        a.q = tensor_lin(f, p + ".self_attn.q_proj.weight");
        a.k = tensor_lin(f, p + ".self_attn.k_proj.weight");
        a.v = tensor_lin(f, p + ".self_attn.v_proj.weight");
        a.o = tensor_lin(f, p + ".self_attn.o_proj.weight");
        a.qn = tensor_f32(f, p + ".self_attn.q_norm.weight", HD);
        a.kn = tensor_f32(f, p + ".self_attn.k_norm.weight", HD);
        a.sink = tensor_f32(f, p + ".self_attn.attention_sink_bias", HEADS);
        a.gate = tensor_nv4(f, p + ".mlp.gate_proj");
        a.up = tensor_nv4(f, p + ".mlp.up_proj");
        a.down = tensor_nv4(f, p + ".mlp.down_proj");
        a.base_a = tensor_f32(f, p + ".attention_conv.base_kernel", 4 * H);
        a.base_m = tensor_f32(f, p + ".mlp_conv.base_kernel", 4 * H);
        a.att_kp = tensor_lin(f, p + ".attention_conv.kernel_projection.weight");
        a.mlp_kp = tensor_lin(f, p + ".mlp_conv.kernel_projection.weight");
    }
    d->sq.resize(nseq < 1 ? 1 : nseq);
    for (auto& q : d->sq) {
        for (int l = 0; l < DL; l++) {
            CK(cudaMalloc(&q.ck[l], (size_t)RING * KVH * HD * 4));
            CK(cudaMalloc(&q.cv[l], (size_t)RING * KVH * HD * 4));
            CK(cudaMemset(q.ck[l], 0, (size_t)RING * KVH * HD * 4));
            CK(cudaMemset(q.cv[l], 0, (size_t)RING * KVH * HD * 4));
        }
        CK(cudaMalloc(&q.anchor, H * 4));
        CK(cudaMemset(q.anchor, 0, H * 4));
    }
    d->nb = (int)d->sq.size();
    d->tok_p = tensor_nv4(f, "lilicorr.token_proj");
    d->pass_p = tensor_nv4(f, "lilicorr.pass_hidden_proj");
    d->ctx_p = tensor_nv4(f, "lilicorr.context_proj");
    d->tok_b = tensor_f32(f, "lilicorr.token_proj.bias", LH);
    d->pass_b = tensor_f32(f, "lilicorr.pass_hidden_proj.bias", LH);
    d->ctx_b = tensor_f32(f, "lilicorr.context_proj.bias", LH);
    d->out_norm = tensor_f32(f, "lilicorr.output_norm.weight", LH);
    d->anc_norm = tensor_f32(f, "lilicorr.anchor_norm.weight", LH);
    d->factor = tensor_bf16(f, "lilicorr.factor_input_proj.weight");
    d->factor_b = tensor_f32(f, "lilicorr.factor_input_proj.bias", LH);
    d->out_h = tensor_bf16(f, "lilicorr.out_head.weight");
    d->out_b = tensor_f32(f, "lilicorr.out_head.bias", LH);
    d->in_h = tensor_bf16(f, "lilicorr.in_head.weight");
    d->in_b = tensor_f32(f, "lilicorr.in_head.bias", LH);
    d->anc_h = tensor_bf16(f, "lilicorr.anchor_out_head.weight");
    d->anc_b = tensor_f32(f, "lilicorr.anchor_out_head.bias", LH);
    d->feat_nw = tensor_f32(f, "lilicorr.feature_norm.weight", 5);
    d->feat_nb = tensor_f32(f, "lilicorr.feature_norm.bias", 5);
    {
        const uint8_t* fu; Info fi;
        get(f, "lilicorr.feature_mlp.up_proj.weight", fu, fi);
        d->feat_up = (const uint16_t*)up_raw(fu, fi.len);
    }
    d->feat_up_b = tensor_f32(f, "lilicorr.feature_mlp.up_proj.bias", LH);
    d->feat_down = tensor_bf16(f, "lilicorr.feature_mlp.down_proj.weight");
    d->feat_down_b = tensor_f32(f, "lilicorr.feature_mlp.down_proj.bias", LH);
    for (int i = 0; i < 2; i++) {
        std::string p = "lilicorr.layers." + std::to_string(i);
        LLayer& a = d->ll[i];
        a.an = tensor_f32(f, p + ".attn_norm.weight", LH);
        a.mn = tensor_f32(f, p + ".mlp_norm.weight", LH);
        a.qkv = tensor_bf16(f, p + ".attn.in_proj_weight");
        a.qkv_b = tensor_f32(f, p + ".attn.in_proj_bias", 3 * LH);
        a.out = tensor_nv4(f, p + ".attn.out_proj");
        a.out_b = tensor_f32(f, p + ".attn.out_proj.bias", LH);
        a.up = tensor_nv4(f, p + ".mlp.up_proj");
        a.up_b = tensor_f32(f, p + ".mlp.up_proj.bias", 2 * LH);
        a.down = tensor_nv4(f, p + ".mlp.down_proj");
        a.down_b = tensor_f32(f, p + ".mlp.down_proj.bias", LH);
    }
    d->slot = tensor_f32(f, "lilicorr.slot_embedding", SLOTS * LH);
    d->rank = tensor_f32(f, "lilicorr.rank_embedding", TOPK * LH);
    {   // attention bias [heads][56][56] from the relative-slot and same-slot tables
        const uint8_t* rd; Info ri;
        get(f, "lilicorr.relative_slot_bias", rd, ri);
        const uint8_t* sd; Info si;
        get(f, "lilicorr.same_slot_bias", sd, si);
        const uint16_t* rb = (const uint16_t*)rd;
        const uint16_t* sb = (const uint16_t*)sd;
        std::vector<float> bias((size_t)LHEADS * NODES * NODES, 0.f);
        for (int i = 0; i < NODES; i++) for (int j = 0; j < NODES; j++) {
            int s_i = i / TOPK, s_j = j / TOPK;
            int rel = s_i - s_j; if (rel < -7) rel = -7; if (rel > 7) rel = 7;
            for (int h = 0; h < LHEADS; h++) {
                float b = bf2f(rb[h * 15 + rel + 7]);
                if (s_i == s_j) b += bf2f(sb[h]);
                bias[((size_t)h * NODES + i) * NODES + j] = b;
            }
        }
        CK(cudaMalloc(&d->abias, bias.size() * 4));
        CK(cudaMemcpy(d->abias, bias.data(), bias.size() * 4, cudaMemcpyHostToDevice));
    }
    munmap(f.map, f.size);

    const size_t NB = d->nb;
    auto fa = [&](float** p, size_t n) { CK(cudaMalloc(p, n * 4)); CK(cudaMemset(*p, 0, n * 4)); };
    const size_t R = NB * BLOCK;  // rows of every stream's block
    const size_t N = NB * NODES;  // lattice nodes of every stream
    fa(&d->x, R * H); fa(&d->r, R * H); fa(&d->h, R * H); fa(&d->z, R * H);
    fa(&d->y, R * 1024); fa(&d->q, R * H); fa(&d->k, R * 256); fa(&d->v, R * 256);
    fa(&d->g, R * INTER); fa(&d->u, R * INTER); fa(&d->hid, R * H);
    fa(&d->dlog, R * VOCAB);
    fa(&d->ictx, (size_t)TMAX * H); fa(&d->ik, TMAX * 256); fa(&d->iv, TMAX * 256);
    fa(&d->csv, R * 128 * CAND); fa(&d->lse, R);
    fa(&d->tokstate, N * LH); fa(&d->passstate, R * LH); fa(&d->feat, N * 8); fa(&d->fh, N * LH);
    fa(&d->nodes, N * LH); fa(&d->ln, N * LH); fa(&d->lqkv, N * 3 * LH);
    fa(&d->lqkv3, N * 3 * LH); fa(&d->lo, N * LH); fa(&d->cat, N * 3 * LH);
    fa(&d->edges_out, N * LH); fa(&d->edges_in, N * LH);
    fa(&d->aout, NB * LH); fa(&d->astate, NB * LH); fa(&d->acat, NB * H); fa(&d->gathered, N * H); fa(&d->tmp, N * 2 * LH);
    fa(&d->cvals, R * CAND);
    CK(cudaMalloc(&d->cids, R * CAND * 4));
    CK(cudaMalloc(&d->csi, R * 128 * CAND * 4));
    CK(cudaMalloc(&d->out_dev, NB * SLOTS * 4));
    CK(cudaMallocHost(&d->out_host, NB * SLOTS * 4));
    CK(cudaMalloc(&d->ckp, DL * NB * sizeof(void*)));
    CK(cudaMalloc(&d->cvp, DL * NB * sizeof(void*)));
    CK(cudaMalloc(&d->clen_d, NB * 4)); CK(cudaMalloc(&d->pos_d, NB * 4)); CK(cudaMalloc(&d->anc_d, NB * 4));
    CK(cudaMallocHost(&d->ckp_h, DL * NB * sizeof(void*)));
    CK(cudaMallocHost(&d->cvp_h, DL * NB * sizeof(void*)));
    CK(cudaMallocHost(&d->clen_h, NB * 4)); CK(cudaMallocHost(&d->pos_h, NB * 4)); CK(cudaMallocHost(&d->anc_h_tok, NB * 4));
    CK(cudaMalloc(&d->ancp, NB * sizeof(void*))); CK(cudaMallocHost(&d->ancp_h, NB * sizeof(void*)));
    {
        std::vector<const float*> rk(DL * NB), rv(DL * NB);
        for (int l = 0; l < DL; l++) for (size_t q = 0; q < NB; q++) { rk[l * NB + q] = d->sq[q].ck[l]; rv[l * NB + q] = d->sq[q].cv[l]; }
        CK(cudaMalloc(&d->ring_k, DL * NB * sizeof(void*)));
        CK(cudaMalloc(&d->ring_v, DL * NB * sizeof(void*)));
        CK(cudaMemcpy(d->ring_k, rk.data(), DL * NB * sizeof(void*), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d->ring_v, rv.data(), DL * NB * sizeof(void*), cudaMemcpyHostToDevice));
        CK(cudaMalloc(&d->irow_seq, TMAX * 4)); CK(cudaMalloc(&d->irow_pos, TMAX * 4));
        CK(cudaMallocHost(&d->irow_seq_h, TMAX * 4)); CK(cudaMallocHost(&d->irow_pos_h, TMAX * 4));
    }
    CK(cudaDeviceSynchronize());
    fprintf(stderr, "bolt: LiLiCorr resident, fc %d x %d, attention and kernel projections %s\n", d->fc.n, d->fc.k, draft_q8() ? "Q8_0" : "bf16");
    return d;
}

void draft_reset(Draft* d, int seq) {
    if (d) d->sq[seq].clen = 0;
}

size_t draft_blob_bytes() { return ((size_t)DL * 2 * RING * KVH * HD + H) * 4; }

void draft_save(Draft* d, int seq, void* dst, cudaStream_t s) {
    if (!d) return;
    Draft::SeqCtx& q = d->sq[seq];
    char* p = (char*)dst;
    const size_t ring = (size_t)RING * KVH * HD * 4;
    for (int l = 0; l < DL; l++) {
        CK(cudaMemcpyAsync(p, q.ck[l], ring, cudaMemcpyDeviceToDevice, s)); p += ring;
        CK(cudaMemcpyAsync(p, q.cv[l], ring, cudaMemcpyDeviceToDevice, s)); p += ring;
    }
    CK(cudaMemcpyAsync(p, q.anchor, H * 4, cudaMemcpyDeviceToDevice, s));
}

void draft_restore(Draft* d, int seq, const void* src, int clen, cudaStream_t s) {
    if (!d) return;
    Draft::SeqCtx& q = d->sq[seq];
    const char* p = (const char*)src;
    const size_t ring = (size_t)RING * KVH * HD * 4;
    for (int l = 0; l < DL; l++) {
        CK(cudaMemcpyAsync(q.ck[l], p, ring, cudaMemcpyDeviceToDevice, s)); p += ring;
        CK(cudaMemcpyAsync(q.cv[l], p, ring, cudaMemcpyDeviceToDevice, s)); p += ring;
    }
    CK(cudaMemcpyAsync(q.anchor, p, H * 4, cudaMemcpyDeviceToDevice, s));
    q.clen = clen;
}

int draft_clen(Draft* d, int seq) { return d ? d->sq[seq].clen : 0; }

void draft_skip(Draft* d, int seq, int upto) {
    if (d && upto > d->sq[seq].clen) d->sq[seq].clen = upto;
}

void draft_ingest(Draft* d, int seq, const float* cap, int pos0, int n, cudaStream_t s) {
    if (!d || n <= 0) return;
    S = s;
    Draft::SeqCtx& q = d->sq[seq];
    // cap rows are [row][6][4096]: exactly the concatenated feature vector the fc expects.
    mm(d->fc, cap, 6 * H, d->ictx, H, n);
    rms(d->ictx, d->hid_norm, d->ictx, H, n);
    CK(cudaMemcpyAsync(q.anchor, d->ictx + (size_t)(n - 1) * H, H * 4, cudaMemcpyDeviceToDevice, s));
    for (int l = 0; l < DL; l++) {
        DLayer& a = d->L[l];
        mm(a.k, d->ictx, H, d->ik, KVH * HD, n);
        mm(a.v, d->ictx, H, d->iv, KVH * HD, n);
        qk_norm_rope_k<<<dim3(n, KVH), 64, 0, s>>>(d->ik, a.kn, KVH, pos0);
        write_kv_k<<<n, 256, 0, s>>>(d->ik, d->iv, q.ck[l], q.cv[l], pos0);
    }
    q.clen = pos0 + n;
}

// The kept rows of several sequences' last block in one pass over the drafter's projections. cap holds the whole block
// (nrows rows); sequence seqs[i] keeps rows [row0[i], row0[i] + n[i]) at positions pos0[i]...; its anchor is the last.
void draft_ingest_multi(Draft* d, int nrows, const float* cap, int nseq, const int* seqs, const int* row0, const int* n, const int* pos0, cudaStream_t s) {
    if (!d || nseq < 1) return;
    S = s;
    for (int r = 0; r < nrows; r++) { d->irow_seq_h[r] = -1; d->irow_pos_h[r] = 0; }
    int last[SEG_MAX];
    for (int i = 0; i < nseq; i++) {
        for (int j = 0; j < n[i]; j++) { d->irow_seq_h[row0[i] + j] = seqs[i]; d->irow_pos_h[row0[i] + j] = pos0[i] + j; }
        last[i] = row0[i] + n[i] - 1;
    }
    CK(cudaMemcpyAsync(d->irow_seq, d->irow_seq_h, nrows * 4, cudaMemcpyHostToDevice, s));
    CK(cudaMemcpyAsync(d->irow_pos, d->irow_pos_h, nrows * 4, cudaMemcpyHostToDevice, s));
    mm(d->fc, cap, 6 * H, d->ictx, H, nrows);
    rms(d->ictx, d->hid_norm, d->ictx, H, nrows);
    for (int i = 0; i < nseq; i++)
        CK(cudaMemcpyAsync(d->sq[seqs[i]].anchor, d->ictx + (size_t)last[i] * H, H * 4, cudaMemcpyDeviceToDevice, s));
    for (int l = 0; l < DL; l++) {
        DLayer& a = d->L[l];
        mm(a.k, d->ictx, H, d->ik, KVH * HD, nrows);
        mm(a.v, d->ictx, H, d->iv, KVH * HD, nrows);
        qk_norm_rope_k<<<dim3(nrows, KVH), 64, 0, s>>>(d->ik, a.kn, KVH, 0, nullptr, d->irow_pos);
        write_kv_multi_k<<<nrows, 256, 0, s>>>(d->ik, d->iv, d->ring_k + (size_t)l * d->nb, d->ring_v + (size_t)l * d->nb, d->irow_seq, d->irow_pos);
    }
    for (int i = 0; i < nseq; i++) d->sq[seqs[i]].clen = pos0[i] + n[i];
}

// out[b] = *ptrs[b], n floats a row: the anchors of the call's streams, whose addresses a graph cannot hold.
__global__ void gather_rows_k(const float* const* __restrict__ ptrs, float* __restrict__ out, int n) {
    const float* src = ptrs[blockIdx.x];
    float* dst = out + (size_t)blockIdx.x * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = src[i];
}

// dattn2_k's shared memory, allowed once (outside any capture)
void dattn_opt() {
    static bool opted = false;
    if (!opted) { cudaFuncSetAttribute(dattn2_k, cudaFuncAttributeMaxDynamicSharedMemorySize, DA_SMEM); opted = true; }
}

// The proposal for B blocks, every launch on S: what one call varies (which streams, their rings, lengths, positions,
// anchors) comes from the pinned tables, copied in here, so the same sequence of launches serves any later call with B
// streams and can be captured as a graph. Ends with the slots copied to out_host; the caller waits for the stream.
void propose_body(Draft* d, int B, cudaStream_t s) {
    CK(cudaMemcpyAsync(d->ckp, d->ckp_h, DL * d->nb * sizeof(void*), cudaMemcpyHostToDevice, s));
    CK(cudaMemcpyAsync(d->cvp, d->cvp_h, DL * d->nb * sizeof(void*), cudaMemcpyHostToDevice, s));
    CK(cudaMemcpyAsync(d->clen_d, d->clen_h, d->nb * 4, cudaMemcpyHostToDevice, s));
    CK(cudaMemcpyAsync(d->pos_d, d->pos_h, d->nb * 4, cudaMemcpyHostToDevice, s));
    CK(cudaMemcpyAsync(d->anc_d, d->anc_h_tok, d->nb * 4, cudaMemcpyHostToDevice, s));
    CK(cudaMemcpyAsync(d->ancp, d->ancp_h, (size_t)B * sizeof(void*), cudaMemcpyHostToDevice, s));
    const int R = B * BLOCK;            // rows of all the blocks
    const int NN = B * NODES;           // lattice nodes of all the streams
    dmark("start");
    block_embed_k<<<R, 256, 0, s>>>(d->embed, d->anc_d, d->x);
    CK(cudaMemcpyAsync(d->r, d->x, (size_t)R * H * 4, cudaMemcpyDeviceToDevice, s));
    for (int l = 0; l < DL; l++) {
        DLayer& a = d->L[l];
        if (l > 0) {
            addv(d->x, d->r, (size_t)R * H);
            CK(cudaMemcpyAsync(d->r, d->x, (size_t)R * H * 4, cudaMemcpyDeviceToDevice, s));
        }
        rms(d->x, a.in_norm, d->h, H, R);
        mm(a.att_kp, d->h, H, d->y, 1024, R);
        gconv_k<<<(R * H + 255) / 256, 256, 0, s>>>(d->h, a.base_a, d->y, d->z, R, 0);
        mm(a.q, d->z, H, d->q, H, R);
        mm(a.k, d->z, H, d->k, 256, R);
        mm(a.v, d->z, H, d->v, 256, R);
        qk_norm_rope_k<<<dim3(R, HEADS), 64, 0, s>>>(d->q, a.qn, HEADS, 0, d->pos_d);
        qk_norm_rope_k<<<dim3(R, KVH), 64, 0, s>>>(d->k, a.kn, KVH, 0, d->pos_d);
        if (dattn_tiled()) {
            dattn2_k<<<dim3(4, KVH, B), 256, DA_SMEM, s>>>(d->q, d->k, d->v, d->ckp + (size_t)l * d->nb, d->cvp + (size_t)l * d->nb, a.sink, d->h, d->clen_d, d->pos_d);
        } else {
            dattn_k<<<dim3(HEADS, R), 128, 0, s>>>(d->q, d->k, d->v, d->ckp + (size_t)l * d->nb, d->cvp + (size_t)l * d->nb, a.sink, d->h, d->clen_d, d->pos_d);
        }
        dmark("layer.attn");
        mm(a.o, d->h, H, d->z, H, R);
        gconv_k<<<(R * H + 255) / 256, 256, 0, s>>>(d->z, a.base_a, d->y, d->x, R, 1);
        addv(d->x, d->r, (size_t)R * H);
        CK(cudaMemcpyAsync(d->r, d->x, (size_t)R * H * 4, cudaMemcpyDeviceToDevice, s));
        rms(d->x, a.post_norm, d->h, H, R);
        mm(a.mlp_kp, d->h, H, d->y, 1024, R);
        gconv_k<<<(R * H + 255) / 256, 256, 0, s>>>(d->h, a.base_m, d->y, d->z, R, 0);
        mm(a.gate, d->z, H, d->g, INTER, R);
        mm(a.up, d->z, H, d->u, INTER, R);
        silu_mul_k<<<((size_t)R * INTER + 255) / 256, 256, 0, s>>>(d->g, d->g, d->u, (size_t)R * INTER);
        mm(a.down, d->g, INTER, d->h, H, R);
        gconv_k<<<(R * H + 255) / 256, 256, 0, s>>>(d->h, a.base_m, d->y, d->x, R, 1);
        dmark("layer.rest");
    }
    addv(d->x, d->r, (size_t)R * H);
    rms(d->x, d->final_norm, d->hid, H, R);
    dmark("final");

    // Candidates of the seven draft slots (rows 1..7 of each block) from the target's head, with their log-probabilities.
    mma_q8(s, d->head, d->hid, H, d->dlog, VOCAB, R);
    dmark("lm_head");
    k_candidates(s, d->dlog, R, TOPK, d->cvals, d->cids, d->csv, d->csi);
    dmark("candidates");
    lse_k<<<R, 256, 0, s>>>(d->dlog, d->lse);
    dmark("lse");

    // The lattice head scores each stream's 56 (slot, rank) candidates.
    gather_cand_k<<<NN, 256, 0, s>>>(d->embed, d->cids, d->gathered);
    mm(d->tok_p, d->gathered, H, d->tokstate, LH, NN, d->tok_b);
    mm(d->pass_p, d->hid, H, d->passstate, LH, R, d->pass_b);
    features_k<<<B, 64, 0, s>>>(d->cvals, d->lse, d->feat_nw, d->feat_nb, d->feat);
    feat_up_k<<<NN, 256, 0, s>>>(d->feat_up, d->feat_up_b, d->feat, d->fh);
    mm(d->feat_down, d->fh, LH, d->tmp, LH, NN, d->feat_down_b);
    nodes_k<<<NN, 256, 0, s>>>(d->tokstate, d->passstate, d->tmp, d->slot, d->rank, d->nodes);
    for (int i = 0; i < 2; i++) {
        LLayer& a = d->ll[i];
        rms(d->nodes, a.an, d->ln, LH, NN);
        mm(a.qkv, d->ln, LH, d->lqkv, 3 * LH, NN, a.qkv_b);
        float* lq = d->lqkv3;
        split3_k<<<NN, 256, 0, s>>>(d->lqkv, lq, lq + (size_t)NN * LH, lq + (size_t)2 * NN * LH);
        lat_attn_k<<<dim3(LHEADS, NN), 128, 0, s>>>(lq, lq + (size_t)NN * LH, lq + (size_t)2 * NN * LH, d->abias, d->lo);
        mm(a.out, d->lo, LH, d->ln, LH, NN, a.out_b);
        addv(d->nodes, d->ln, (size_t)NN * LH);
        rms(d->nodes, a.mn, d->ln, LH, NN);
        mm(a.up, d->ln, LH, d->tmp, 2 * LH, NN, a.up_b);
        silu_inplace(d->tmp, (size_t)NN * 2 * LH);
        mm(a.down, d->tmp, 2 * LH, d->ln, LH, NN, a.down_b);
        addv(d->nodes, d->ln, (size_t)NN * LH);
    }
    rms(d->nodes, d->out_norm, d->ln, LH, NN);
    dmark("lattice");
    gather_rows_k<<<B, 256, 0, s>>>(d->ancp, d->acat, H);
    mm(d->ctx_p, d->acat, H, d->astate, LH, B, d->ctx_b);
    rms(d->astate, d->anc_norm, d->astate, LH, B);
    cat3_k<<<NN, 256, 0, s>>>(d->ln, d->astate, d->cat);
    mm(d->factor, d->cat, 3 * LH, d->fh, LH, NN, d->factor_b);
    silu_inplace(d->fh, (size_t)NN * LH);
    mm(d->out_h, d->fh, LH, d->edges_out, LH, NN, d->out_b);
    mm(d->in_h, d->fh, LH, d->edges_in, LH, NN, d->in_b);
    l2norm_k<<<NN, 256, 0, s>>>(d->edges_out, VEPS);
    l2norm_k<<<NN, 256, 0, s>>>(d->edges_in, VEPS);
    mm(d->anc_h, d->astate, LH, d->aout, LH, B, d->anc_b);
    l2norm_k<<<B, 256, 0, s>>>(d->aout, VEPS);
    dmark("heads");
    static const int top1 = getenv("BOLT_DRAFT_TOP1") && atoi(getenv("BOLT_DRAFT_TOP1")) != 0;
    walk_k<<<B, 256, 0, s>>>(d->aout, d->edges_out, d->edges_in, d->cids, d->out_dev, top1);
    dmark("walk");
    CK(cudaMemcpyAsync(d->out_host, d->out_dev, (size_t)B * SLOTS * 4, cudaMemcpyDeviceToHost, s));
}

// BOLT_DRAFT_GRAPH=0 launches the proposal eagerly (the way before); profiling (BOLT_DPROF) does too, for its events.
bool draft_graph_on() {
    static const bool on = !(getenv("BOLT_DRAFT_GRAPH") && atoi(getenv("BOLT_DRAFT_GRAPH")) == 0);
    return on && !dprof_on();
}

// Seven draft tokens each for the blocks that start with anchors[i] at positions pos[i] of sequences seqs[i]: the
// streams share every weight read. got[i] is how many tokens stream i drafted (0: its context is not at pos[i]).
// The launches for B streams are captured into a CUDA graph the first time B streams propose and replayed after that:
// the pinned tables are read at replay, so a graph serves every later call with B streams; a capture that fails
// turns the graph off and the call runs eagerly, with the same launches.
void draft_propose_multi(Draft* d, int n, const int* seqs, const int* anchors, const int* pos, int* out, int* got, cudaStream_t s) {
    for (int i = 0; i < n; i++) got[i] = 0;
    if (!d || n < 1) return;
    // streams the drafter can serve now; the rest report 0
    int idx[SEG_MAX], B = 0;
    for (int i = 0; i < n && B < d->nb; i++) {
        Draft::SeqCtx& q = d->sq[seqs[i]];
        if (q.clen <= 0 || pos[i] != q.clen) continue;
        idx[B++] = i;
    }
    if (B == 0) return;
    S = s;
    for (int b = 0; b < B; b++) {
        Draft::SeqCtx& q = d->sq[seqs[idx[b]]];
        for (int l = 0; l < DL; l++) { d->ckp_h[l * d->nb + b] = q.ck[l]; d->cvp_h[l * d->nb + b] = q.cv[l]; }
        d->clen_h[b] = q.clen;
        d->pos_h[b] = pos[idx[b]];
        d->anc_h_tok[b] = anchors[idx[b]];
        d->ancp_h[b] = q.anchor;
    }
    if (dattn_tiled()) dattn_opt();
    cudaGraphExec_t exec = nullptr;
    if (draft_graph_on() && !d->graph_off) {
        auto it = d->graphs.find(B);
        if (it != d->graphs.end()) {
            exec = it->second;
        } else {
            cudaGraph_t g = nullptr;
            cudaError_t e = cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal);
            if (e == cudaSuccess) {
                propose_body(d, B, s);
                e = cudaStreamEndCapture(s, &g);
            }
            if (e == cudaSuccess) e = cudaGraphInstantiate(&exec, g, 0);
            if (g) cudaGraphDestroy(g);
            if (e == cudaSuccess) {
                d->graphs[B] = exec;
            } else {
                (void)cudaGetLastError();
                exec = nullptr;
                d->graph_off = true;
                fprintf(stderr, "bolt draft: the proposal's graph for %d stream%s was not captured (%s): proposing eagerly\n", B, B == 1 ? "" : "s", cudaGetErrorString(e));
            }
        }
    }
    if (exec) CK(cudaGraphLaunch(exec, s));
    else propose_body(d, B, s);
    CK(cudaStreamSynchronize(s));
    dprof_collect();
    for (int b = 0; b < B; b++) {
        for (int j = 0; j < SLOTS; j++) out[idx[b] * 8 + j] = d->out_host[b * SLOTS + j];
        got[idx[b]] = SLOTS;
    }
}

int draft_propose(Draft* d, int seq, int anchor_tok, int pos, int* out, cudaStream_t s) {
    int got = 0;
    draft_propose_multi(d, 1, &seq, &anchor_tok, &pos, out, &got, s);
    return got;
}

void draft_prof_report() {
    if (!dprof_on()) return;
    double total = 0;
    for (auto& e : g_dacc) total += e.second;
    fprintf(stderr, "bolt draft prof: total %.2f ms\n", total);
    for (auto& e : g_dacc) fprintf(stderr, "  %-14s %8.2f ms\n", e.first.c_str(), e.second);
    g_dacc.clear();
}
