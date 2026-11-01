// Bolt 24GB host side: checkpoint loading, the block forward (latent MoE, Mamba-2, GQA attention) and the C API.
// Weights stay packed on the GPU. A block of up to TMAX rows is one launch sequence, captured into a CUDA
// graph when it repeats. Several sequences (BOLT_SEQS) keep their own state; one block can carry a segment of rows from
// each, so a shared decode round is one forward. Mamba state: two SSM slots per sequence (the block reads one and
// writes the other, so a verify window can be rolled back) and a ring of conv slots.

#include "bolt_common.h"
#include "bolt_draft.h"

#include <cuda.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <chrono>
#include "bolt_port.h"

#define CK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { fprintf(stderr, "bolt: %s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(_e)); } } while (0)

namespace {

constexpr int PROJ = 18560;
constexpr int ATT_SPLITS = 32;
constexpr int ATT_PART_FLOATS = 16 + 16 + 16 * 128;
constexpr int AUX_LAYERS[6] = {1, 18, 35, 51, 68, 85};

int aux_slot(int layer) {
    for (int i = 0; i < 6; i++) if (AUX_LAYERS[i] == layer) return i;
    return -1;
}

struct Layer {
    int type = 0;  // 0 mamba, 1 moe, 2 attention
    float* norm = nullptr;
    // mamba
    TNV4 in_proj{}, out_proj{};
    TNV4W in_projw{}, out_projw{};   // the same in the fp4 MMA's fragment order (BOLT_NV4_FP4)
    float *conv_w = nullptr, *conv_b = nullptr, *A_log = nullptr, *D = nullptr, *dt_bias = nullptr, *gnorm = nullptr;
    float* ssm = nullptr;   // [seq][2][SSM_ELEMS] (one slot a sequence with a lazy state)
    float *hist_x = nullptr, *hist_dt = nullptr;  // scan inputs of the last record block's rows [BATCH_MAX] (lazy: [seq][2][VERIFY_MAX])
    float* conv = nullptr;  // [seq][CONV_SLOTS][CONV_ELEMS]
    // moe
    TQ8 fc1{}, fc2{};
    TBF gate{};
    float* gbias = nullptr;
    int experts = 0, width = 0;
    TIQ2 up_s{}, down_s{};
    IQ2Slab up_ss{}, down_ss{};   // the same two tensors in the slab layout, for the int8 tile kernel (BOLT_SHARED_TILE)
    IQ2Slab up{}, down{};
    // attention
    TQ8 q{}, k{}, v{}, o{};
    void *kc = nullptr, *vc = nullptr;      // [seq][context][256] bf16, or e4m3 bytes (fp8 cache)
    float *ks = nullptr, *vs = nullptr;     // fp8 cache: a scale per (seq, token, head)
};

struct Info { std::string dtype; std::vector<long> shape; size_t off = 0, len = 0; };

struct Store {
    std::unordered_map<std::string, std::string> shard;
    std::unordered_map<std::string, std::unordered_map<std::string, Info>> headers;
    std::unordered_map<std::string, std::pair<const uint8_t*, size_t>> files;
    std::string dir;
    std::vector<std::pair<void*, size_t>> maps;
};

std::string slurp(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return {};
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::string s(n, 0);
    if (fread(s.data(), 1, n, f) != (size_t)n) s.clear();
    fclose(f);
    return s;
}

bool parse_header(const uint8_t* file, size_t n, std::unordered_map<std::string, Info>& out) {
    if (n < 8) return false;
    uint64_t hlen;
    memcpy(&hlen, file, 8);
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
    return true;
}

Store* open_store(const char* dir) {
    auto* s = new Store();
    s->dir = dir;
    std::string idx = slurp(std::string(dir) + "/model.safetensors.index.json");
    for (const char* prefix : {"\"language_model."}) {
        size_t p = 0;
        while (true) {
            size_t a = idx.find(prefix, p);
            if (a == std::string::npos) break;
            size_t b = idx.find('"', a + 1);
            std::string name = idx.substr(a + 1, b - a - 1);
            size_t c = idx.find('"', b + 1);
            size_t d = idx.find('"', c + 1);
            s->shard[name] = idx.substr(c + 1, d - c - 1);
            p = d + 1;
        }
    }
    fprintf(stderr, "bolt: index %zu tensors\n", s->shard.size());
    return s;
}

bool lookup(Store* s, const std::string& name, Info& info, const uint8_t*& data) {
    auto it = s->shard.find(name);
    if (it == s->shard.end()) return false;
    const std::string& shard = it->second;
    if (!s->headers.count(shard)) {
        std::string path = s->dir + "/" + shard;
        int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) return false;
        struct stat st;
        if (fstat(fd, &st) != 0) { close(fd); return false; }
        void* m = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        if (m == MAP_FAILED) return false;
        madvise(m, st.st_size, MADV_WILLNEED);
        s->maps.push_back({m, (size_t)st.st_size});
        s->files[shard] = {(const uint8_t*)m, (size_t)st.st_size};
        parse_header((const uint8_t*)m, st.st_size, s->headers[shard]);
    }
    auto jt = s->headers[shard].find(name);
    if (jt == s->headers[shard].end()) return false;
    info = jt->second;
    uint64_t hlen;
    memcpy(&hlen, s->files[shard].first, 8);
    data = s->files[shard].first + 8 + hlen + info.off;
    return true;
}

size_t numel(const Info& info) { size_t n = 1; for (long d : info.shape) n *= (size_t)d; return n; }

float* dev_f32(Store* s, const std::string& name) {
    Info info; const uint8_t* data;
    if (!lookup(s, name, info, data)) { fprintf(stderr, "bolt: missing %s\n", name.c_str()); return nullptr; }
    size_t n = numel(info);
    std::vector<float> h(n);
    if (info.dtype == "F32") memcpy(h.data(), data, n * 4);
    else if (info.dtype == "BF16") {
        for (size_t i = 0; i < n; i++) {
            uint16_t v; memcpy(&v, data + i * 2, 2);
            uint32_t u = (uint32_t)v << 16;
            memcpy(&h[i], &u, 4);
        }
    } else { fprintf(stderr, "bolt: %s dtype %s\n", name.c_str(), info.dtype.c_str()); return nullptr; }
    float* d = nullptr;
    CK(cudaMalloc(&d, n * 4));
    CK(cudaMemcpy(d, h.data(), n * 4, cudaMemcpyHostToDevice));
    return d;
}

void* dev_copy(const void* src, size_t n) {
    void* d = nullptr;
    CK(cudaMalloc(&d, n));
    CK(cudaMemcpy(d, src, n, cudaMemcpyHostToDevice));
    return d;
}

TQ8 load_q8(Store* s, const std::string& name) {
    TQ8 m{};
    Info w; const uint8_t* data;
    if (!lookup(s, name + ".weight", w, data) || w.shape.size() != 3 || w.shape[2] != 34) {
        fprintf(stderr, "bolt: %s is not a Q8_0 tensor\n", name.c_str());
        return m;
    }
    m.n = (int)w.shape[0];
    m.k = (int)w.shape[1] * 32;
    void* raw = dev_copy(data, w.len);
    void *qt, *dt;
    CK(cudaMalloc(&qt, tile_bytes_q8(m.n, m.k)));
    CK(cudaMalloc(&dt, tile_scale_bytes_q8(m.n, m.k)));
    tile_q8(0, (const uint8_t*)raw, qt, dt, m.n, m.k);
    CK(cudaDeviceSynchronize());
    cudaFree(raw);
    m.q = (const uint4*)qt; m.d = (const uint32_t*)dt;
    return m;
}

bool g_nv4_fp4 = false;   // BOLT_NV4_FP4: prompt chunks' NVFP4 projections on the fp4 tensor cores (rows quantized to NVFP4)
TNV4W load_nv4w(Store* s, const std::string& name) {
    TNV4W m{};
    Info w; const uint8_t* data;
    if (!lookup(s, name + ".weight", w, data) || w.shape.size() != 2) return m;
    m.n = (int)w.shape[0]; m.k = (int)w.shape[1] * 2;
    void* rq = dev_copy(data, w.len);
    Info sc; const uint8_t* sd;
    lookup(s, name + ".weight_scale", sc, sd);
    void* rs = dev_copy(sd, sc.len);
    Info g; const uint8_t* gd;
    lookup(s, name + ".weight_scale_2", g, gd);
    memcpy(&m.s2, gd, 4);
    void *qw, *sfw;
    CK(cudaMalloc(&qw, tile_bytes_nv4(m.n, m.k)));
    CK(cudaMalloc(&sfw, tile_scale_bytes_nv4(m.n, m.k)));
    tile_nv4w(0, (const uint8_t*)rq, (const uint8_t*)rs, qw, sfw, m.n, m.k);
    CK(cudaDeviceSynchronize());
    cudaFree(rq); cudaFree(rs);
    m.q = (const uint4*)qw; m.sf = (const uint32_t*)sfw;
    return m;
}
TNV4 load_nv4(Store* s, const std::string& name) {
    TNV4 m{};
    Info w; const uint8_t* data;
    if (!lookup(s, name + ".weight", w, data) || w.shape.size() != 2) {
        fprintf(stderr, "bolt: %s is not an NVFP4 tensor\n", name.c_str());
        return m;
    }
    m.n = (int)w.shape[0];
    m.k = (int)w.shape[1] * 2;
    void* rq = dev_copy(data, w.len);
    Info sc; const uint8_t* sd;
    lookup(s, name + ".weight_scale", sc, sd);
    void* rs = dev_copy(sd, sc.len);
    Info g; const uint8_t* gd;
    lookup(s, name + ".weight_scale_2", g, gd);
    memcpy(&m.s2, gd, 4);
    void *qt, *sct;
    CK(cudaMalloc(&qt, tile_bytes_nv4(m.n, m.k)));
    CK(cudaMalloc(&sct, tile_scale_bytes_nv4(m.n, m.k)));
    tile_nv4(0, (const uint8_t*)rq, (const uint8_t*)rs, qt, sct, m.n, m.k);
    CK(cudaDeviceSynchronize());
    cudaFree(rq); cudaFree(rs);
    m.q = (const uint4*)qt; m.sc = (const uint2*)sct;
    return m;
}

TIQ2 load_iq2(Store* s, const std::string& name) {
    TIQ2 m{};
    Info w; const uint8_t* data;
    if (!lookup(s, name + ".weight", w, data) || w.shape.size() != 3 || w.shape[2] != 66) {
        fprintf(stderr, "bolt: %s is not an IQ2_XXS tensor\n", name.c_str());
        return m;
    }
    m.n = (int)w.shape[0];
    m.k = (int)w.shape[1] * 256;
    void* raw = dev_copy(data, w.len);
    void *qt, *dt;
    CK(cudaMalloc(&qt, tile_bytes_iq2(m.n, m.k)));
    CK(cudaMalloc(&dt, tile_scale_bytes_iq2(m.n, m.k)));
    tile_iq2(0, (const uint8_t*)raw, qt, dt, m.n, m.k);
    CK(cudaDeviceSynchronize());
    cudaFree(raw);
    m.q = (const uint4*)qt; m.d = (const uint16_t*)dt;
    return m;
}

TBF load_bf16(Store* s, const std::string& name) {
    TBF m{};
    Info w; const uint8_t* data;
    if (!lookup(s, name, w, data) || w.shape.size() != 2) {
        fprintf(stderr, "bolt: %s is not a bf16 matrix\n", name.c_str());
        return m;
    }
    m.n = (int)w.shape[0];
    m.k = (int)w.shape[1];
    void* raw = dev_copy(data, w.len);
    void* wt;
    CK(cudaMalloc(&wt, tile_bytes_bf16(m.n, m.k)));
    tile_bf16(0, (const uint16_t*)raw, wt, m.n, m.k);
    CK(cudaDeviceSynchronize());
    cudaFree(raw);
    m.w = (const uint4*)wt;
    return m;
}

// All experts of one projection into a slab. Returns false if a tensor is missing or oddly shaped.
bool load_slab(Store* s, const std::string& layer_prefix, const char* proj, int experts, int rows, int k, IQ2Slab& slab) {
    int nb = k / 256;
    size_t raw_per = (size_t)rows * nb * 66;
    size_t qs_per = (size_t)rows * nb * 64, d_per = (size_t)rows * nb;
    uint8_t* raw = nullptr; uint8_t* qs = nullptr; void* d = nullptr;
    CK(cudaMalloc(&raw, raw_per * experts));
    CK(cudaMalloc(&qs, qs_per * experts));
    CK(cudaMalloc(&d, d_per * experts * 2));
    for (int e = 0; e < experts; e++) {
        Info w; const uint8_t* data;
        std::string name = layer_prefix + ".mixer.experts." + std::to_string(e) + "." + proj + ".weight";
        if (!lookup(s, name, w, data) || w.len != raw_per) {
            fprintf(stderr, "bolt: expert tensor %s missing or the wrong size\n", name.c_str());
            return false;
        }
        CK(cudaMemcpy(raw + e * raw_per, data, raw_per, cudaMemcpyHostToDevice));
    }
    repack_iq2(0, raw, qs, d, (size_t)rows * experts, k);
    CK(cudaDeviceSynchronize());
    cudaFree(raw);
    slab.qs = qs; slab.d = d; slab.qs_stride = qs_per; slab.d_stride = d_per;
    return true;
}

// one IQ2_XXS tensor as a slab (the shared expert for the int8 tile kernel)
bool load_slab_one(Store* s, const std::string& name, IQ2Slab& slab) {
    Info w; const uint8_t* data;
    if (!lookup(s, name + ".weight", w, data) || w.shape.size() != 3 || w.shape[2] != 66) return false;
    const int rows = (int)w.shape[0], k = (int)w.shape[1] * 256, nb = k / 256;
    uint8_t* raw = nullptr; uint8_t* qs = nullptr; void* d = nullptr;
    CK(cudaMalloc(&raw, w.len));
    CK(cudaMalloc(&qs, (size_t)rows * nb * 64));
    CK(cudaMalloc(&d, (size_t)rows * nb * 2));
    CK(cudaMemcpy(raw, data, w.len, cudaMemcpyHostToDevice));
    repack_iq2(0, raw, qs, d, (size_t)rows, k);
    CK(cudaDeviceSynchronize());
    cudaFree(raw);
    slab.qs = qs; slab.d = d; slab.qs_stride = (size_t)rows * nb * 64; slab.d_stride = (size_t)rows * nb;
    return true;
}

std::vector<std::string> block_types(const std::string& cfg) {
    std::vector<std::string> t;
    size_t p = cfg.find("\"block_configs\"");
    size_t end = cfg.find("\"blockwise_members\"", p);
    size_t i = p;
    while (i < end) {
        size_t a = cfg.find("\"block_type\"", i);
        if (a == std::string::npos || a > end) break;
        size_t q1 = cfg.find('"', cfg.find(':', a) + 1);
        size_t q2 = cfg.find('"', q1 + 1);
        t.push_back(cfg.substr(q1 + 1, q2 - q1 - 1));
        i = q2 + 1;
    }
    return t;
}

void moe_shape(const std::string& cfg, int index, int& experts, int& width) {
    size_t p = cfg.find("\"block_configs\"");
    int seen = 0;
    size_t i = p;
    while (true) {
        size_t a = cfg.find("\"block_type\"", i);
        if (a == std::string::npos) break;
        size_t q1 = cfg.find('"', cfg.find(':', a) + 1);
        size_t q2 = cfg.find('"', q1 + 1);
        std::string ty = cfg.substr(q1 + 1, q2 - q1 - 1);
        if (ty == "moe") {
            if (seen == index) {
                size_t e = cfg.find("\"n_routed_experts\"", a);
                size_t w = cfg.find("\"moe_intermediate_size\"", a);
                experts = atoi(cfg.c_str() + cfg.find(':', e) + 1);
                width = atoi(cfg.c_str() + cfg.find(':', w) + 1);
                return;
            }
            seen++;
        }
        i = q2 + 1;
    }
}

}  // namespace

// Attention rows of all sequences, [seq][kv_stride tokens][256] bf16 per layer and for K and V. The address range is
// reserved at the full window for every sequence, but physical memory is mapped a granule (2 MiB, 4096 tokens) at a
// time as a sequence grows, so sixteen streams can each have the native 262,144-token window while the memory held
// follows what they have actually used. If the driver cannot do that, everything is allocated up front.
struct KvPool {
    bool lazy = false;
    size_t gran = 0;                       // bytes a granule maps
    int tokens_per_gran = 0;               // tokens of one buffer in a granule
    int gran_max = 0;                      // granules a sequence can use
    size_t stride_bytes = 0;               // address range of one sequence in one buffer
    std::vector<CUdeviceptr> base;         // one per K/V buffer of every attention layer
    std::vector<CUmemGenericAllocationHandle> handle;   // [buffer][seq][granule]
    std::vector<int> mapped;               // granules mapped per sequence
    CUmemAllocationProp prop{};
    size_t reserve_free = 0;               // device bytes left free for everything else
};

struct Bolt;
extern "C" void bolt_prof_report(Bolt* b);

struct Bolt {
    int context = 32768;
    bool kv_pin = false;     // BOLT_KV_PIN=1: every sequence's whole window is mapped at load and kept
    bool kv_fp8 = false;     // BOLT_KV=fp8: K and V as e4m3 with a scale per token and head
    bool ssm_lazy = false;   // BOLT_SSM_LAZY=1: one SSM slot a sequence; a window's kept rows are folded in by its next block
    int kv_row = 512;        // bytes of one token's 256 K (or V) values
    int kv_stride = 32768;   // tokens between sequences in the KV buffers (the window rounded up to a granule)
    KvPool kv;
    float eps = 1e-5f, route_scale = 5.f;
    uint16_t* embed = nullptr;
    uint16_t* xemb = nullptr;   // [TMAX][HID] bf16: image rows for the next block (placeholder tokens read them)
    float* norm_f = nullptr;
    TQ8 head{};
    std::vector<Layer> layers;

    // activations, TMAX rows each
    float *x = nullptr, *xn = nullptr, *r1 = nullptr, *r2 = nullptr, *proj = nullptr, *xbc = nullptr, *ybuf = nullptr, *gout = nullptr;
    float *qb = nullptr, *kb = nullptr, *vb = nullptr, *apart = nullptr, *aout = nullptr;
    float *glog = nullptr, *lat = nullptr, *lat2 = nullptr, *rw = nullptr, *act = nullptr, *mo = nullptr, *sact = nullptr;
    bool shared_tile = false;     // BOLT_SHARED_TILE: prompt chunks' shared expert on the int8 tile kernel
    bool shared_decode = false;   // BOLT_SHARED_DECODE: verify blocks' shared expert on the int8 deep-prefetch kernel too
    int prompt_min_rows = 256;    // BOLT_PROMPT_MIN_ROWS: rows from which a block takes the prompt-chunk kernels (the perplexity harness lowers it)
    int8_t* xq_s = nullptr; float* xs_s = nullptr;   // its int8 rows (up to 5376 wide) and scales
    int *s_order = nullptr, *s_first = nullptr, *s_count = nullptr, *s_exp = nullptr, *s_nwork = nullptr;
    int *ridx = nullptr, *order = nullptr, *wfirst = nullptr, *wcount = nullptr, *wexp = nullptr, *nwork = nullptr;
    int* moe_stat = nullptr;      // BOLT_PROF: distinct experts and work items, summed over the MoE launches since the last report
    uint16_t* xpack[3] = {nullptr, nullptr, nullptr};   // the GEMMs' packed rows, one scratch a stream
    uint8_t* xq4 = nullptr; uint8_t* xsf4 = nullptr;      // rows as NVFP4 for the fp4 projections (main stream only)
    int8_t *xq_t = nullptr, *xq_p = nullptr;   // int8 inputs of the expert kernels: per token (up) and per pair (down)
    float *xs_t = nullptr, *xs_p = nullptr;
    float* logits = nullptr;
    float *cap = nullptr;  // [TMAX][6][4096]
    float *cvals = nullptr, *csv = nullptr;
    int *cids = nullptr, *csi = nullptr;
    std::vector<float> cvals_h;
    std::vector<int> cids_h;
    int* msel = nullptr;                  // structured output: the rows to redraw, their masks, and the candidates found
    uint32_t* mmask = nullptr;
    float* mvals = nullptr;
    int* mids = nullptr;

    Ctl* ctl_d = nullptr;
    Ctl* ctl_h = nullptr;
    cudaStream_t s0 = nullptr, s1 = nullptr, s2 = nullptr;
    cudaEvent_t ev_fork = nullptr, ev_j1 = nullptr, ev_j2 = nullptr;
    std::unordered_map<int, cudaGraphExec_t> graphs;
    std::unordered_map<int, int> uses;

    // sequences
    struct Seq {
        bool used = false;
        int len = 0;                    // tokens in the cache
        int ssm = 0, conv = 0;          // current SSM slot (0/1) and conv ring slot
        int last_rows = 0, last_row0 = 0, last_ssm = 0, last_conv = 0, last_len = 0, last_record = 0;
        int pending = 0, hist = 0;      // lazy state: kept rows of the last window not folded in yet, and their saved-input slot
        int ing_pos0 = 0, ing_n = 0, ing_row0 = 0;  // rows of the last block the drafter has not absorbed yet
        int ingest_floor = 0;           // positions below this never reach the drafter's window
    };
    int nseq = 1;
    std::vector<Seq> seqs;
    int last_cand_k = 0;
    ReplayLayer* replay = nullptr;  // device table, one entry per Mamba layer
    int mamba_layers = 0;
    Draft* draft = nullptr;
    double cost_ms[VERIFY_MAX + 1] = {};
    std::vector<std::pair<int, double>> shared_ms;  // forward ms of a block of that many rows over several sequences
    bool prof = false;  // BOLT_PROF: eager launches with an event after every stage; a report every BOLT_PROF_EVERY (100) blocks
    bool prof_serial = false;   // BOLT_PROF=2: the shared expert and fc1 on the main stream too, so every stage's time is its own
    int prof_blocks = 0, prof_every = 100;
    long prof_rows = 0; int prof_rows_n = 0;   // rows of the blocks since the last report
    std::vector<cudaEvent_t> prof_ev;
    std::vector<const char*> prof_name;
    std::unordered_map<std::string, std::pair<double, int>> prof_acc;
    double draft_ms = 0, draft_multi_ms = 0;
};

namespace {

// Bytes of one token's K and V over the 8 attention layers (fp8: the bytes plus a scale per head).
uint64_t kv_token_bytes(const Bolt* b) { return 8ull * 2 * (b->kv_row + (b->kv_fp8 ? 2 * 4 : 0)); }

// ---- lazily mapped KV (see KvPool) ----

constexpr const char* kv_err(CUresult r) { return r == CUDA_SUCCESS ? "ok" : "failed"; }

bool kv_open(Bolt* b, int nbuf) {
    KvPool& k = b->kv;
    if (getenv("BOLT_KV_EAGER") || nbuf < 1) return false;
    CUdevice dev;
    if (cuCtxGetDevice(&dev) != CUDA_SUCCESS) return false;
    int vmm = 0;
    if (cuDeviceGetAttribute(&vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, dev) != CUDA_SUCCESS || !vmm) return false;
    k.prop = {};
    k.prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    k.prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    k.prop.location.id = dev;
    if (cuMemGetAllocationGranularity(&k.gran, &k.prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS || k.gran % b->kv_row) return false;
    k.tokens_per_gran = (int)(k.gran / b->kv_row);
    b->kv_stride = (b->context + k.tokens_per_gran - 1) / k.tokens_per_gran * k.tokens_per_gran;
    k.gran_max = b->kv_stride / k.tokens_per_gran;
    k.stride_bytes = (size_t)b->kv_stride * b->kv_row;
    k.base.assign(nbuf, 0);
    for (int i = 0; i < nbuf; i++) {
        if (cuMemAddressReserve(&k.base[i], (size_t)b->nseq * k.stride_bytes, k.gran, 0, 0) != CUDA_SUCCESS) {
            for (int j = 0; j < i; j++) cuMemAddressFree(k.base[j], (size_t)b->nseq * k.stride_bytes);
            k.base.clear();
            return false;
        }
    }
    k.handle.assign((size_t)nbuf * b->nseq * k.gran_max, 0);
    k.mapped.assign(b->nseq, 0);
    const char* r = getenv("BOLT_KV_RESERVE_GIB");
    k.reserve_free = (size_t)((r ? atof(r) : 3.0) * 1073741824.0);
    k.lazy = true;
    return true;
}

void kv_unmap_granule(Bolt* b, int seq, int g, int nbufs) {
    KvPool& k = b->kv;
    for (int i = 0; i < nbufs; i++) {
        CUmemGenericAllocationHandle& h = k.handle[((size_t)i * b->nseq + seq) * k.gran_max + g];
        if (!h) continue;
        cuMemUnmap(k.base[i] + (size_t)seq * k.stride_bytes + (size_t)g * k.gran, k.gran);
        cuMemRelease(h);
        h = 0;
    }
}

// Map physical memory under the first `tokens` rows of the sequence. False when the driver or the free memory refuses.
bool kv_ensure(Bolt* b, int seq, int tokens) {
    KvPool& k = b->kv;
    if (!k.lazy) return true;
    const int need = (tokens + k.tokens_per_gran - 1) / k.tokens_per_gran;
    if (need > k.gran_max) return false;
    if (k.mapped[seq] >= need) return true;
    cudaFree(0);   // the context is current on this thread
    const int nbuf = (int)k.base.size();
    CUmemAccessDesc acc = {};
    acc.location = k.prop.location;
    acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    while (k.mapped[seq] < need) {
        const int g = k.mapped[seq];
        size_t fr = 0, tot = 0;
        cudaMemGetInfo(&fr, &tot);
        if (fr < k.reserve_free + (size_t)nbuf * k.gran) {
            fprintf(stderr, "bolt: no memory to grow sequence %d past %d tokens (%.1f GiB free)\n", seq, g * k.tokens_per_gran, fr / 1073741824.0);
            return false;
        }
        int done = 0;
        for (; done < nbuf; done++) {
            CUmemGenericAllocationHandle h;
            if (cuMemCreate(&h, k.gran, &k.prop, 0) != CUDA_SUCCESS) break;
            CUdeviceptr addr = k.base[done] + (size_t)seq * k.stride_bytes + (size_t)g * k.gran;
            if (cuMemMap(addr, k.gran, 0, h, 0) != CUDA_SUCCESS) { cuMemRelease(h); break; }
            if (cuMemSetAccess(addr, k.gran, &acc, 1) != CUDA_SUCCESS) { cuMemUnmap(addr, k.gran); cuMemRelease(h); break; }
            k.handle[((size_t)done * b->nseq + seq) * k.gran_max + g] = h;
            cuMemsetD8Async(addr, 0, k.gran, b->s0);
        }
        if (done < nbuf) {
            kv_unmap_granule(b, seq, g, done);
            return false;
        }
        k.mapped[seq]++;
    }
    return true;
}

// Give back the memory a finished sequence mapped beyond its first granule.
void kv_trim(Bolt* b, int seq) {
    KvPool& k = b->kv;
    if (!k.lazy || b->kv_pin) return;
    cudaFree(0);
    CK(cudaStreamSynchronize(b->s0));
    const int nbuf = (int)k.base.size();
    while (k.mapped[seq] > 1) {
        kv_unmap_granule(b, seq, k.mapped[seq] - 1, nbuf);
        k.mapped[seq]--;
    }
}

// Debug: BOLT_SKIP=scan,route,shared,... leaves stages out (the output is garbage) to see what each costs in a graph.
bool skip(const char* name) {
    static const char* e = getenv("BOLT_SKIP");
    return e && strstr(e, name);
}

void mark(Bolt* b, const char* name) {
    if (!b->prof) return;
    cudaEvent_t e;
    CK(cudaEventCreate(&e));
    CK(cudaEventRecord(e, b->s0));
    b->prof_ev.push_back(e);
    b->prof_name.push_back(name);
}

void prof_collect(Bolt* b) {
    for (size_t i = 1; i < b->prof_ev.size(); i++) {
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, b->prof_ev[i - 1], b->prof_ev[i]));
        auto& a = b->prof_acc[b->prof_name[i]];
        a.first += ms;
        a.second++;
    }
    for (auto e : b->prof_ev) cudaEventDestroy(e);
    b->prof_ev.clear();
    b->prof_name.clear();
}

void enqueue_block(Bolt* b, int T, int record, int cand_k, int last_only, int nseg, int split_log) {
    mma_decode_rows(record != 0);
    cudaStream_t s0 = b->s0, s1 = b->prof_serial ? b->s0 : b->s1, s2 = b->prof_serial ? b->s0 : b->s2;
    const Ctl* ctl = b->ctl_d;
    mark(b, "start");
    k_embed(s0, b->embed, b->xemb, ctl, b->x, T);
    const int L = (int)b->layers.size();
    for (int l = 0; l < L; l++) {
        Layer& ly = b->layers[l];
        const float* r1 = l == 0 ? nullptr : b->r1;
        const float* r2 = (l > 0 && b->layers[l - 1].type == 1) ? b->r2 : nullptr;
        float* cap = nullptr;
        if (l >= 1 && aux_slot(l - 1) >= 0) cap = b->cap + (size_t)aux_slot(l - 1) * HID;
        k_add_rms(s0, b->x, r1, r2, ly.norm, b->xn, cap, 6 * HID, b->eps, T);
        mark(b, "add_rms");
        if (ly.type == 0) {
            if (!skip("inproj")) {
                if (g_nv4_fp4 && !record && T >= b->prompt_min_rows) { quant_nv4_rows(s0, b->xn, HID, b->xq4, b->xsf4, T, HID); mma_nv4_fp4(s0, ly.in_projw, b->xq4, b->xsf4, b->proj, PROJ, T); }
                else mma_nv4(s0, ly.in_proj, b->xn, HID, b->proj, PROJ, T);
            }
            mark(b, "mamba.in_proj");
            k_conv(s0, ctl, ly.conv, ly.conv_w, ly.conv_b, b->proj, PROJ, b->xbc, T, record, record ? ly.hist_x : nullptr, record ? ly.hist_dt : nullptr, nseg, b->ssm_lazy);
            if (!skip("scan")) k_scan(s0, ctl, ly.ssm, b->xbc, b->proj, PROJ, ly.A_log, ly.dt_bias, ly.D, b->ybuf, T, record, nseg, b->ssm_lazy, ly.hist_x, ly.hist_dt);
            mark(b, "mamba.conv+scan");
            k_gate_rms(s0, b->ybuf, b->proj, PROJ, ly.gnorm, b->gout, b->eps, T);
            mark(b, "mamba.gate_rms");
            if (!skip("outproj")) {
                if (g_nv4_fp4 && !record && T >= b->prompt_min_rows) { quant_nv4_rows(s0, b->gout, 8192, b->xq4, b->xsf4, T, 8192); mma_nv4_fp4(s0, ly.out_projw, b->xq4, b->xsf4, b->r1, HID, T); }
                else mma_nv4(s0, ly.out_proj, b->gout, 8192, b->r1, HID, T);
            }
            mark(b, "mamba.out_proj");
        } else if (ly.type == 2) {
            TQ8Multi m{};
            m.w[0] = ly.q; m.w[1] = ly.k; m.w[2] = ly.v;
            m.y[0] = b->qb; m.y[1] = b->kb; m.y[2] = b->vb;
            m.ldy[0] = 4096; m.ldy[1] = 256; m.ldy[2] = 256;
            mma_q8_multi(s0, m, b->xn, HID, T);
            mark(b, "attn.qkv");
            const KvView kvv{ly.kc, ly.vc, ly.ks, ly.vs, b->kv_stride, b->kv_fp8};
            k_kv_store(s0, ctl, b->kb, b->vb, kvv, T);
            static const bool old_attn = getenv("BOLT_ATTN_OLD") != nullptr;
            if (record && !old_attn) k_attn_mma_windows(s0, ctl, b->qb, kvv, b->apart, b->aout, T, nseg, 1 << split_log);
            else if (T <= 8 || record) k_attn(s0, ctl, b->qb, kvv, b->apart, b->aout, T, T <= 16 ? ATT_SPLITS : (T <= 64 ? 16 : 8), 1);
            else k_attn_mma(s0, ctl, b->qb, kvv, b->apart, b->aout, T, T >= 256 ? 4 : 16);
            mark(b, "attn.core");
            mma_q8(s0, ly.o, b->aout, 4096, b->r1, HID, T);
            mark(b, "attn.o_proj");
        } else {
            CK(cudaEventRecord(b->ev_fork, s0));
            CK(cudaStreamWaitEvent(s1, b->ev_fork));
            CK(cudaStreamWaitEvent(s2, b->ev_fork));
            // shared expert
            if (!skip("shared")) {
                if (b->shared_tile && ly.up_ss.qs && (T >= b->prompt_min_rows || b->shared_decode)) {
                    // the shared expert as one expert over the block's rows: the int8 tile kernel for prompt chunks, the
                    // deep-prefetch kernel for verify blocks (BOLT_SHARED_DECODE=0 keeps the bf16 path for those)
                    const MoeWork sw{b->s_order, b->s_first, b->s_count, b->s_exp, b->s_nwork};
                    auto dense = T >= 256 ? moe_tile_dense : moe_coop_dense;
                    k_quant_rows(s1, b->xn, HID, b->xq_s, HID, b->xs_s, HID / 32, T, HID);
                    dense(s1, ly.up_ss, sw, T, b->xq_s, b->xs_s, b->sact, ly.up_s.n, ly.up_s.n, HID, 1);
                    k_quant_rows(s1, b->sact, ly.up_s.n, b->xq_s, ly.up_s.n, b->xs_s, ly.up_s.n / 32, T, ly.up_s.n);
                    dense(s1, ly.down_ss, sw, T, b->xq_s, b->xs_s, b->r2, HID, HID, ly.up_s.n, 0);
                } else {
                    mma_iq2(s1, ly.up_s, b->xn, HID, b->sact, ly.up_s.n, T, 1);
                    mma_iq2(s1, ly.down_s, b->sact, ly.up_s.n, b->r2, HID, T);
                }
            }
            CK(cudaEventRecord(b->ev_j1, s1));
            if (b->prof_serial) mark(b, "moe.shared");
            // latent projection
            if (!skip("fc")) mma_q8(s2, ly.fc1, b->xn, HID, b->lat, 1024, T);
            CK(cudaEventRecord(b->ev_j2, s2));
            if (b->prof_serial) mark(b, "moe.fc1");
            // routing
            if (!skip("gate")) mma_bf16(s0, ly.gate, b->xn, HID, b->glog, 512, T);
            mark(b, "moe.gate");
            if (!skip("route")) k_route(s0, b->glog, 512, ly.gbias, b->ridx, b->rw, ly.experts, b->route_scale, T);
            mark(b, "moe.route");
            const int pairs = T * TOPN;
            k_moe_group(s0, b->ridx, pairs, b->order, b->wfirst, b->wcount, b->wexp, b->nwork, b->prof ? b->moe_stat : nullptr);
            mark(b, "moe.group");
            CK(cudaStreamWaitEvent(s0, b->ev_j2));
            mark(b, "moe.wait_fc1");
            MoeWork mw{b->order, b->wfirst, b->wcount, b->wexp, b->nwork};
            if (!skip("experts")) {
                static const bool bf = moe_bf16();
                if (bf) {
                    moe_up_bf(s0, ly.up, mw, b->lat, 1024, b->act, ly.width, pairs);
                    mark(b, "moe.up");
                    moe_down_bf(s0, ly.down, mw, b->act, ly.width, b->mo, ly.width, pairs);
                    mark(b, "moe.down");
                } else {
                    k_quant_rows(s0, b->lat, 1024, b->xq_t, 1024, b->xs_t, 32, T, 1024);
                    moe_up(s0, ly.up, mw, b->xq_t, b->xs_t, b->act, ly.width, pairs);
                    mark(b, "moe.up");
                    k_quant_rows(s0, b->act, ly.width, b->xq_p, ly.width, b->xs_p, ly.width / 32, pairs, ly.width);
                    moe_down(s0, ly.down, mw, b->xq_p, b->xs_p, b->mo, ly.width, pairs);
                    mark(b, "moe.down");
                }
            }
            k_combine(s0, b->mo, b->rw, b->lat2, T);
            if (!skip("fc")) mma_q8(s0, ly.fc2, b->lat2, 1024, b->r1, HID, T);
            mark(b, "moe.combine+fc2");
            CK(cudaStreamWaitEvent(s0, b->ev_j1));
            mark(b, "moe.wait_shared");
        }
    }
    k_add_rms(s0, b->x, b->r1, b->r2, b->norm_f, b->xn, nullptr, 0, b->eps, T);
    mark(b, "final_norm");
    if (cand_k > 0) {
        // last_only: only the final row is read back (the prompt pass), so the head runs on that row alone
        const int rows = last_only ? 1 : T;
        const float* hx = last_only ? b->xn + (size_t)(T - 1) * HID : b->xn;
        mma_q8(s0, b->head, hx, HID, b->logits, VOCAB, rows);
        mark(b, "lm_head");
        k_candidates(s0, b->logits, rows, cand_k, b->cvals, b->cids, b->csv, b->csi);
    }
    mark(b, "candidates");
}

int bucket_of(int nseg) { return nseg <= 1 ? 0 : nseg <= 2 ? 1 : nseg <= 4 ? 2 : nseg <= 8 ? 3 : 4; }
constexpr int BUCKET_SEGS[5] = {1, 2, 4, 8, 16};

int graph_key(int T, int record, int cand_k, int last_only, int bucket, int split_log) {
    return (((((T * 2 + record) * 2 + last_only) * 5 + bucket) * 6 + split_log) * 128) + cand_k;
}

// Hand the drafter the target features of the rows it has not seen. Old prompt rows outside its window are skipped.
void flush_ingest(Bolt* b, int id) {
    Bolt::Seq& q = b->seqs[id];
    if (q.ing_n <= 0) return;
    int pos0 = q.ing_pos0, n = q.ing_n, row0 = q.ing_row0;
    q.ing_n = 0;
    if (!b->draft) return;
    int skip = 0;
    if (pos0 < q.ingest_floor) skip = q.ingest_floor - pos0 < n ? q.ingest_floor - pos0 : n;
    if (skip >= n) {
        draft_skip(b->draft, id, pos0 + n);
        return;
    }
    if (skip > 0) draft_skip(b->draft, id, pos0 + skip);
    draft_ingest(b->draft, id, b->cap + (size_t)(row0 + skip) * 6 * HID, pos0 + skip, n - skip, b->s0);
}

// Hand the drafter every sequence's pending rows. Sequences whose rows come from the last record block go together.
void flush_all_ingest(Bolt* b) {
    int seqs[SEG_MAX], row0[SEG_MAX], n[SEG_MAX], pos0[SEG_MAX], k = 0, nrows = 0;
    for (int id = 0; id < b->nseq; id++) {
        Bolt::Seq& q = b->seqs[id];
        if (q.ing_n <= 0) continue;
        if (b->draft && q.last_record && q.ing_pos0 >= q.ingest_floor) {
            seqs[k] = id; row0[k] = q.ing_row0; n[k] = q.ing_n; pos0[k] = q.ing_pos0;
            nrows = std::max(nrows, q.ing_row0 + q.ing_n);
            k++;
            q.ing_n = 0;
        } else {
            flush_ingest(b, id);
        }
    }
    if (k > 0) draft_ingest_multi(b->draft, nrows, b->cap, k, seqs, row0, n, pos0, b->s0);
}

struct SegIn { int seq, rows; const int* tokens; };

// CTAs a verify window's attention launches per stream and KV head, a power of two: the most any sequence needs at its
// own win_chunk (the kernel gives each sequence the same key chunks whatever this is, so it changes no rounding).
int split_log_of(const SegIn* sg, int nseg, const Bolt* b) {
    int need = 1, lg = 0;
    for (int i = 0; i < nseg; i++) {
        const int base0 = b->seqs[sg[i].seq].len, chunk = win_chunk(base0);
        need = std::max(need, (base0 + sg[i].rows + chunk - 1) / chunk);
    }
    while ((1 << lg) < need && lg < 5) lg++;
    return lg;
}

// Run one block: a segment of rows for each of nseg distinct sequences. A record block (verify windows, up to
// VERIFY_MAX rows a segment) can be cut back per sequence by keep; any other block is a single segment. Returns 0
// on success.
struct EmbedPatch { int row; const uint16_t* src; int n; };   // rows [row, row + n) of the block come from src (device bf16)

int run_blocks(Bolt* b, const SegIn* sg, int nseg, int record, int cand_k, int last_only = 0, const EmbedPatch* patches = nullptr, int npatch = 0) {
    if (nseg < 1 || nseg > SEG_MAX || (!record && nseg != 1)) return -1;
    int T = 0;
    for (int i = 0; i < nseg; i++) {
        if (sg[i].seq < 0 || sg[i].seq >= b->nseq || sg[i].rows < 1) return -1;
        for (int j = 0; j < i; j++) if (sg[j].seq == sg[i].seq) return -1;
        if (record && sg[i].rows > VERIFY_MAX) return -1;
        if (b->seqs[sg[i].seq].len + sg[i].rows > b->context) return -2;
        if (!kv_ensure(b, sg[i].seq, b->seqs[sg[i].seq].len + sg[i].rows)) return -4;
        T += sg[i].rows;
    }
    if (T > TMAX || (record && T > BATCH_MAX)) return -1;
    if (cand_k > 0 && !last_only && T > LOGIT_ROWS) return -1;
    flush_all_ingest(b);
    Ctl* h = b->ctl_h;
    memset(h, 0, sizeof(Ctl));
    int row = 0;
    for (int i = 0; i < nseg; i++) {
        Bolt::Seq& q = b->seqs[sg[i].seq];
        h->seg_row0[i] = row;
        h->seg_base[i] = q.len;
        h->seg_seq[i] = sg[i].seq;
        h->seg_ssm[i] = b->ssm_lazy ? 0 : q.ssm;
        h->seg_conv[i] = q.conv;
        h->seg_replay[i] = b->ssm_lazy ? q.pending : 0;
        h->seg_hist[i] = q.hist;
        for (int t = 0; t < sg[i].rows; t++) h->tokens[row + t] = sg[i].tokens[t];
        row += sg[i].rows;
    }
    h->seg_row0[nseg] = row;
    h->nrows = T;
    h->nseg = nseg;
    CK(cudaMemcpyAsync(b->ctl_d, h, sizeof(Ctl), cudaMemcpyHostToDevice, b->s0));
    for (int i = 0; i < npatch; i++)
        CK(cudaMemcpyAsync(b->xemb + (size_t)patches[i].row * HID, patches[i].src, (size_t)patches[i].n * HID * 2, cudaMemcpyDeviceToDevice, b->s0));
    const int bucket = bucket_of(nseg);
    const int split_log = record ? split_log_of(sg, nseg, b) : 0;
    // Shared blocks come in many sizes: round the rows up so a handful of captured graphs serve them all. The extra rows
    // are embedded from token 0 and touch no sequence's state (the kernels stop at ctl->nrows).
    const int real_rows = T;
    if (record && (nseg > 1 || T > VERIFY_MAX)) T = T <= 64 ? (T + 7) / 8 * 8 : (T + 15) / 16 * 16;
    int key = graph_key(T, record, cand_k, last_only, bucket, split_log);
    auto it = b->graphs.find(key);
    if (b->prof) {
        enqueue_block(b, T, record, cand_k, last_only, BUCKET_SEGS[bucket], split_log);
    } else if (it != b->graphs.end()) {
        CK(cudaGraphLaunch(it->second, b->s0));
    } else if (++b->uses[key] >= 2) {
        cudaGraph_t g;
        CK(cudaStreamBeginCapture(b->s0, cudaStreamCaptureModeGlobal));
        enqueue_block(b, T, record, cand_k, last_only, BUCKET_SEGS[bucket], split_log);
        CK(cudaStreamEndCapture(b->s0, &g));
        cudaGraphExec_t exec;
        CK(cudaGraphInstantiate(&exec, g, 0));
        cudaGraphDestroy(g);
        b->graphs[key] = exec;
        CK(cudaGraphLaunch(exec, b->s0));
    } else {
        enqueue_block(b, T, record, cand_k, last_only, BUCKET_SEGS[bucket], split_log);
    }
    if (cand_k > 0) {
        const int rows = last_only ? 1 : real_rows;
        CK(cudaMemcpyAsync(b->cvals_h.data(), b->cvals, (size_t)rows * CAND * 4, cudaMemcpyDeviceToHost, b->s0));
        CK(cudaMemcpyAsync(b->cids_h.data(), b->cids, (size_t)rows * CAND * 4, cudaMemcpyDeviceToHost, b->s0));
    }
    CK(cudaStreamSynchronize(b->s0));
    if (b->prof) { prof_collect(b); b->prof_rows += real_rows; b->prof_rows_n++; if (++b->prof_blocks % b->prof_every == 0) bolt_prof_report(b); }
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "bolt: forward T=%d: %s\n", T, cudaGetErrorString(e));
        return -3;
    }
    b->last_cand_k = cand_k;
    for (int i = 0; i < nseg; i++) {
        Bolt::Seq& q = b->seqs[sg[i].seq];
        const int rows = sg[i].rows;
        q.last_rows = rows;
        q.last_row0 = h->seg_row0[i];
        q.last_ssm = q.ssm;
        q.last_conv = q.conv;
        q.last_len = q.len;
        q.last_record = record;
        q.ing_pos0 = q.len;
        q.ing_n = rows;
        q.ing_row0 = h->seg_row0[i];
        // the block advanced every segment as if all its rows were kept; keep() cuts it back
        q.conv = record ? (q.conv + rows) % CONV_SLOTS : (q.conv + 1) % CONV_SLOTS;
        if (b->ssm_lazy) {
            q.pending = 0;              // folded into the state by this block's scan
            if (record) q.hist ^= 1;    // the block saved its rows' inputs in the other slot
        } else q.ssm ^= 1;
        q.len += rows;
    }
    return 0;
}

int run_block(Bolt* b, int seq, const int* tokens, int T, int record, int cand_k, int last_only = 0, const EmbedPatch* patches = nullptr, int npatch = 0) {
    SegIn s{seq, T, tokens};
    return run_blocks(b, &s, 1, record, cand_k, last_only, patches, npatch);
}

}  // namespace

static void bolt_calibrate(Bolt* b);

extern "C" Bolt* bolt_open(const char* dir, int context) {
    {
        // BOLT_SPIN=0 leaves the driver's default wait; spinning on the host while a forward runs skips the wake-up
        // latency of a yielding wait after every sync (llama.cpp measured a large decode gain from this on GB10)
        const char* sp = getenv("BOLT_SPIN");
        if (!sp || atoi(sp) != 0) CK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
    }
    CK(cudaFree(0));
    fprintf(stderr, "bolt: loading %s\n", dir);
    std::string cfg = slurp(std::string(dir) + "/config.json");
    auto types = block_types(cfg);
    if (types.size() != 88) { fprintf(stderr, "bolt: expected 88 blocks, got %zu\n", types.size()); return nullptr; }
    Store* store = open_store(dir);
    Bolt* b = new Bolt();
    b->context = context > 0 ? context : 32768;
    if (b->context > 262144) b->context = 262144;
    {
        const char* ns = getenv("BOLT_SEQS");
        b->nseq = ns ? atoi(ns) : 1;
        if (b->nseq < 1) b->nseq = 1;
        if (b->nseq > SEG_MAX) b->nseq = SEG_MAX;
        b->seqs.assign(b->nseq, Bolt::Seq());
    }
    Info ei; const uint8_t* ed;
    if (!lookup(store, "language_model.backbone.embeddings.weight", ei, ed)) { fprintf(stderr, "bolt: no embeddings\n"); return nullptr; }
    b->embed = (uint16_t*)dev_copy(ed, ei.len);
    b->norm_f = dev_f32(store, "language_model.backbone.norm_f.weight");
    b->head = load_q8(store, "language_model.lm_head");
    b->layers.resize(types.size());
    {
        int n_att = 0;
        for (auto& t : types) n_att += t == "attention";
        const char* lz = getenv("BOLT_SSM_LAZY");
        b->ssm_lazy = lz && atoi(lz) != 0;
        const char* st = getenv("BOLT_SHARED_TILE");   // 0 keeps the shared expert's bf16 path for prompt chunks
        b->shared_tile = !(st && atoi(st) == 0);
        const char* sd = getenv("BOLT_SHARED_DECODE");
        b->shared_decode = sd && atoi(sd) != 0;
        if (const char* pm = getenv("BOLT_PROMPT_MIN_ROWS")) b->prompt_min_rows = std::max(1, atoi(pm));
        const char* f4 = getenv("BOLT_NV4_FP4");   // 0 keeps the fp16 path for prompt chunks (perplexity 3.0418 against 3.0657 on the harness)
        g_nv4_fp4 = !(f4 && atoi(f4) == 0);
        prompt_min_rows_set(b->prompt_min_rows);
        const char* kvf = getenv("BOLT_KV");
        b->kv_fp8 = kvf && (!strcmp(kvf, "fp8") || !strcmp(kvf, "e4m3"));
        b->kv_row = b->kv_fp8 ? 256 : 512;
        b->kv_stride = b->context;
        kv_open(b, 2 * n_att);
        if (b->kv.lazy && !kv_ensure(b, 0, 1)) { fprintf(stderr, "bolt: cannot map the first KV granule\n"); return nullptr; }
        for (int q = 1; b->kv.lazy && q < b->nseq; q++) kv_ensure(b, q, 1);
    }
    int mi = 0, ai = 0, oi = 0;
    for (size_t li = 0; li < types.size(); li++) {
        Layer& ly = b->layers[li];
        std::string p = "language_model.backbone.layers." + std::to_string(li);
        ly.norm = dev_f32(store, p + ".norm.weight");
        if (types[li] == "mamba") {
            ly.type = 0;
            ly.in_proj = load_nv4(store, p + ".mixer.in_proj");
            ly.out_proj = load_nv4(store, p + ".mixer.out_proj");
            if (g_nv4_fp4) { ly.in_projw = load_nv4w(store, p + ".mixer.in_proj"); ly.out_projw = load_nv4w(store, p + ".mixer.out_proj"); }
            ly.conv_w = dev_f32(store, p + ".mixer.conv1d.weight");
            ly.conv_b = dev_f32(store, p + ".mixer.conv1d.bias");
            ly.A_log = dev_f32(store, p + ".mixer.A_log");
            ly.D = dev_f32(store, p + ".mixer.D");
            ly.dt_bias = dev_f32(store, p + ".mixer.dt_bias");
            ly.gnorm = dev_f32(store, p + ".mixer.norm.weight");
            const size_t slots = b->ssm_lazy ? 1 : 2, hist_rows = b->ssm_lazy ? (size_t)b->nseq * 2 * VERIFY_MAX : BATCH_MAX;
            CK(cudaMalloc(&ly.ssm, (size_t)b->nseq * slots * SSM_ELEMS * 4));
            CK(cudaMalloc(&ly.conv, (size_t)b->nseq * CONV_SLOTS * CONV_ELEMS * 4));
            CK(cudaMemset(ly.ssm, 0, (size_t)b->nseq * slots * SSM_ELEMS * 4));
            CK(cudaMemset(ly.conv, 0, (size_t)b->nseq * CONV_SLOTS * CONV_ELEMS * 4));
            CK(cudaMalloc(&ly.hist_x, hist_rows * HIST_ROW * 4));
            CK(cudaMalloc(&ly.hist_dt, hist_rows * 128 * 4));
            CK(cudaMemset(ly.hist_x, 0, hist_rows * HIST_ROW * 4));
            CK(cudaMemset(ly.hist_dt, 0, hist_rows * 128 * 4));
            mi++;
        } else if (types[li] == "moe") {
            ly.type = 1;
            int experts = 0, width = 0;
            moe_shape(cfg, oi, experts, width);
            ly.experts = experts;
            ly.width = width;
            if (experts < 9 || experts > 512 || width % 256 != 0) { fprintf(stderr, "bolt: layer %zu odd moe shape %d x %d\n", li, experts, width); return nullptr; }
            ly.fc1 = load_q8(store, p + ".mixer.fc1_latent_proj");
            ly.fc2 = load_q8(store, p + ".mixer.fc2_latent_proj");
            ly.gate = load_bf16(store, p + ".mixer.gate.weight");
            if (ly.gate.n != experts) { fprintf(stderr, "bolt: layer %zu router has %d rows, config says %d experts\n", li, ly.gate.n, experts); return nullptr; }
            ly.gbias = dev_f32(store, p + ".mixer.gate.e_score_correction_bias");
            ly.up_s = load_iq2(store, p + ".mixer.shared_experts.up_proj");
            ly.down_s = load_iq2(store, p + ".mixer.shared_experts.down_proj");
            if (b->shared_tile) {
                if (!load_slab_one(store, p + ".mixer.shared_experts.up_proj", ly.up_ss) || !load_slab_one(store, p + ".mixer.shared_experts.down_proj", ly.down_ss)) {
                    fprintf(stderr, "bolt: shared expert slab of layer %zu failed; BOLT_SHARED_TILE off for it\n", li);
                    ly.up_ss.qs = nullptr;
                }
            }
            if (!load_slab(store, p, "up_proj", experts, width, 1024, ly.up) ||
                !load_slab(store, p, "down_proj", experts, 1024, width, ly.down)) return nullptr;
            oi++;
            if (oi % 5 == 0) fprintf(stderr, "bolt: moe layers loaded %d\n", oi);
        } else {
            ly.type = 2;
            ly.q = load_q8(store, p + ".mixer.q_proj");
            ly.k = load_q8(store, p + ".mixer.k_proj");
            ly.v = load_q8(store, p + ".mixer.v_proj");
            ly.o = load_q8(store, p + ".mixer.o_proj");
            if (b->kv.lazy) {
                ly.kc = (void*)(uintptr_t)b->kv.base[2 * ai];
                ly.vc = (void*)(uintptr_t)b->kv.base[2 * ai + 1];
            } else {
                const size_t bytes = (size_t)b->nseq * b->context * b->kv_row;
                CK(cudaMalloc(&ly.kc, bytes));
                CK(cudaMalloc(&ly.vc, bytes));
                CK(cudaMemset(ly.kc, 0, bytes));
                CK(cudaMemset(ly.vc, 0, bytes));
            }
            if (b->kv_fp8) {
                const size_t n = (size_t)b->nseq * b->kv_stride * 2;
                CK(cudaMalloc(&ly.ks, n * 4));
                CK(cudaMalloc(&ly.vs, n * 4));
                CK(cudaMemset(ly.ks, 0, n * 4));
                CK(cudaMemset(ly.vs, 0, n * 4));
            }
            ai++;
        }
        if (li % 8 == 0) fprintf(stderr, "bolt: layer %zu\n", li);
    }
    b->mamba_layers = mi;
    {
        std::vector<ReplayLayer> tab;
        for (Layer& ly : b->layers) if (ly.type == 0) tab.push_back({ly.ssm, ly.A_log, ly.dt_bias, ly.hist_x, ly.hist_dt});
        CK(cudaMalloc(&b->replay, tab.size() * sizeof(ReplayLayer)));
        CK(cudaMemcpy(b->replay, tab.data(), tab.size() * sizeof(ReplayLayer), cudaMemcpyHostToDevice));
    }
    b->prof = getenv("BOLT_PROF") != nullptr;
    b->prof_serial = b->prof && atoi(getenv("BOLT_PROF")) == 2;
    if (const char* pe = getenv("BOLT_PROF_EVERY")) b->prof_every = std::max(1, atoi(pe));
    auto zalloc = [&](float** p, size_t floats) { CK(cudaMalloc(p, floats * 4)); CK(cudaMemset(*p, 0, floats * 4)); };
    CK(cudaMalloc(&b->xemb, (size_t)TMAX * HID * 2));
    zalloc(&b->x, (size_t)TMAX * HID);
    zalloc(&b->xn, (size_t)TMAX * HID);
    zalloc(&b->r1, (size_t)TMAX * HID);
    zalloc(&b->r2, (size_t)TMAX * HID);
    zalloc(&b->proj, (size_t)TMAX * PROJ);
    zalloc(&b->xbc, (size_t)TMAX * 10240);
    zalloc(&b->ybuf, (size_t)TMAX * 8192);
    zalloc(&b->gout, (size_t)TMAX * 8192);
    zalloc(&b->qb, (size_t)TMAX * 4096);
    zalloc(&b->kb, (size_t)TMAX * 256);
    zalloc(&b->vb, (size_t)TMAX * 256);
    zalloc(&b->apart, std::max({(size_t)16 * 2 * ATT_SPLITS * ATT_PART_FLOATS, (size_t)64 * 2 * 16 * ATT_PART_FLOATS, (size_t)BATCH_MAX * 2 * 8 * ATT_PART_FLOATS, attn_mma_part_floats(255, 16), attn_mma_part_floats(TMAX, 4), (size_t)SEG_MAX * 2 * 32 * (8 * (32 + 16 * 128))}));  // the blocked kernel's partials fit: (T/4)*2*16*(64*130) floats
    zalloc(&b->aout, (size_t)TMAX * 4096);
    zalloc(&b->glog, (size_t)TMAX * 512);
    zalloc(&b->lat, (size_t)TMAX * 1024);
    zalloc(&b->lat2, (size_t)TMAX * 1024);
    zalloc(&b->rw, (size_t)TMAX * TOPN);
    zalloc(&b->act, (size_t)TMAX * TOPN * 2560);
    zalloc(&b->mo, (size_t)TMAX * TOPN * 1024);
    zalloc(&b->sact, (size_t)TMAX * 5376);
    if (b->shared_tile) {
        CK(cudaMalloc(&b->xq_s, (size_t)TMAX * 5376)); CK(cudaMalloc(&b->xs_s, (size_t)TMAX * (5376 / 32) * 4));
        CK(cudaMalloc(&b->s_order, (size_t)TMAX * 4)); CK(cudaMalloc(&b->s_first, (size_t)TMAX * 4)); CK(cudaMalloc(&b->s_count, (size_t)TMAX * 4));
        CK(cudaMalloc(&b->s_exp, (size_t)TMAX * 4)); CK(cudaMalloc(&b->s_nwork, 4));
    }
    zalloc(&b->logits, (size_t)LOGIT_ROWS * VOCAB);
    zalloc(&b->cap, (size_t)TMAX * 6 * HID);
    zalloc(&b->cvals, (size_t)LOGIT_ROWS * CAND);
    zalloc(&b->csv, (size_t)LOGIT_ROWS * 128 * CAND);
    CK(cudaMalloc(&b->ridx, (size_t)TMAX * TOPN * 4));
    CK(cudaMalloc(&b->order, (size_t)TMAX * TOPN * 4));
    CK(cudaMalloc(&b->wfirst, (size_t)TMAX * TOPN * 4)); CK(cudaMalloc(&b->wcount, (size_t)TMAX * TOPN * 4)); CK(cudaMalloc(&b->wexp, (size_t)TMAX * TOPN * 4)); CK(cudaMalloc(&b->nwork, 4));
    CK(cudaMalloc(&b->moe_stat, 8)); CK(cudaMemset(b->moe_stat, 0, 8));
    CK(cudaMalloc(&b->xq_t, (size_t)TMAX * 1024)); CK(cudaMalloc(&b->xs_t, (size_t)TMAX * 32 * 4));
    CK(cudaMalloc(&b->xq_p, (size_t)TMAX * TOPN * 2560)); CK(cudaMalloc(&b->xs_p, (size_t)TMAX * TOPN * 80 * 4));
    CK(cudaMalloc(&b->cids, (size_t)LOGIT_ROWS * CAND * 4));
    CK(cudaMalloc(&b->csi, (size_t)LOGIT_ROWS * 128 * CAND * 4));
    CK(cudaMalloc(&b->msel, (size_t)LOGIT_ROWS * 4));
    CK(cudaMalloc(&b->mmask, (size_t)LOGIT_ROWS * (VOCAB / 32) * 4));
    CK(cudaMalloc(&b->mvals, (size_t)LOGIT_ROWS * CAND * 4));
    CK(cudaMalloc(&b->mids, (size_t)LOGIT_ROWS * CAND * 4));
    b->cvals_h.assign((size_t)LOGIT_ROWS * CAND, 0.f);
    b->cids_h.assign((size_t)LOGIT_ROWS * CAND, 0);
    CK(cudaMalloc(&b->ctl_d, sizeof(Ctl)));
    CK(cudaMallocHost(&b->ctl_h, sizeof(Ctl)));
    CK(cudaStreamCreateWithFlags(&b->s0, cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&b->s1, cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&b->s2, cudaStreamNonBlocking));
    {
        // packed rows for the GEMMs: TMAX rows (a prompt chunk) of the widest K (out_proj's 8192), one scratch a stream (48 MB)
        const size_t elems = (size_t)TMAX * 8192;
        cudaStream_t ss[3] = {b->s0, b->s1, b->s2};
        for (int i = 0; i < 3; i++) { CK(cudaMalloc(&b->xpack[i], elems * 2)); mma_pack_buffer(ss[i], b->xpack[i], elems); }
        if (g_nv4_fp4) { CK(cudaMalloc(&b->xq4, (size_t)TMAX * 8192 / 2)); CK(cudaMalloc(&b->xsf4, (size_t)TMAX * 8192 / 16)); }
    }
    CK(cudaEventCreateWithFlags(&b->ev_fork, cudaEventDisableTiming));
    CK(cudaEventCreateWithFlags(&b->ev_j1, cudaEventDisableTiming));
    CK(cudaEventCreateWithFlags(&b->ev_j2, cudaEventDisableTiming));
    for (auto& m : store->maps) munmap(m.first, m.second);
    store->maps.clear();
    delete store;

    const char* ddir = getenv("BOLT_DRAFT");
    std::string dd = ddir ? ddir : "/root/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr/snapshots/4c6a5613b6d887194a3a5493e3a18a5b3930bb86";
    b->draft = getenv("BOLT_DRAFT_OFF") ? nullptr : draft_open(dd.c_str(), b->embed, b->head, b->nseq);
    CK(cudaDeviceSynchronize());
    size_t fr = 0, tot = 0;
    cudaMemGetInfo(&fr, &tot);
    fprintf(stderr, "bolt: weights resident, %d mamba %d moe %d attention, %d sequence%s of context %d (%s KV, %s SSM state), %.1f GiB free\n", mi, oi, ai, b->nseq, b->nseq == 1 ? "" : "s", b->context, b->kv_fp8 ? "fp8" : "bf16", b->ssm_lazy ? "one-slot lazy" : "two-slot", fr / 1073741824.0);
    if (b->kv.lazy && getenv("BOLT_KV_PIN") && atoi(getenv("BOLT_KV_PIN")) != 0) {
        // map every sequence's whole window now, so the memory is certain (a start that cannot have it fails here)
        for (int q = 0; q < b->nseq; q++) {
            if (!kv_ensure(b, q, b->context)) {
                fprintf(stderr, "bolt: cannot pin %d sequences of %d tokens of KV cache (mapped %d); lower PARALLEL or CONTEXT, or use KV_DTYPE=fp8\n", b->nseq, b->context, q);
                return nullptr;
            }
        }
        CK(cudaStreamSynchronize(b->s0));
        b->kv_pin = true;
        size_t fr2 = 0, tot2 = 0;
        cudaMemGetInfo(&fr2, &tot2);
        fprintf(stderr, "bolt: KV cache pinned for %d x %d tokens, %.1f GiB free\n", b->nseq, b->context, fr2 / 1073741824.0);
    }
    if (!getenv("BOLT_NOCAL")) bolt_calibrate(b);
    return b;
}

extern "C" Bolt* bolt_load(const char* dir) {
    const char* c = getenv("BOLT_CONTEXT");
    return bolt_open(dir, c ? atoi(c) : 32768);
}


// The C ABI version bolt_native.zig checks at load: bump it with any change to the calls below.
extern "C" int bolt_abi() { return 3; }

extern "C" int bolt_seqs(Bolt* b) { return b->nseq; }
extern "C" int bolt_context(Bolt* b) { return b->context; }

// Device bytes one more sequence holds at this context: both SSM slots and the conv ring of every Mamba layer, the
// KV rows of every attention layer, the drafter's rings.
extern "C" uint64_t bolt_seq_bytes(Bolt* b) {
    const uint64_t ssm = b->ssm_lazy ? (uint64_t)SSM_ELEMS + 2ull * VERIFY_MAX * (HIST_ROW + 128) : 2 * (uint64_t)SSM_ELEMS;
    return (uint64_t)b->mamba_layers * (ssm + (uint64_t)CONV_SLOTS * CONV_ELEMS) * 4 + (uint64_t)(b->kv.lazy && !b->kv_pin ? b->kv.tokens_per_gran : b->context) * kv_token_bytes(b) +
           (b->draft ? draft_blob_bytes() : 0);
}

// Seven draft tokens for the block that starts with `anchor` at the sequence's current position (0 when no drafter).
extern "C" int bolt_propose_seq(Bolt* b, int seq, int anchor, int* out) {
    if (!b->draft) return 0;
    flush_ingest(b, seq);
    return draft_propose(b->draft, seq, anchor, b->seqs[seq].len, out, b->s0);
}

// Drafts for several sequences in one pass over the drafter's weights: out is [n][8], got[i] is 7 or 0.
extern "C" void bolt_propose_multi(Bolt* b, int n, const int* seqs, const int* anchors, int* out, int* got) {
    int pos[SEG_MAX];
    if (!b->draft || n < 1 || n > SEG_MAX) { for (int i = 0; i < n; i++) got[i] = 0; return; }
    flush_all_ingest(b);
    for (int i = 0; i < n; i++) pos[i] = b->seqs[seqs[i]].len;
    draft_propose_multi(b->draft, n, seqs, anchors, pos, out, got, b->s0);
}

extern "C" int bolt_has_draft(Bolt* b) { return b->draft != nullptr; }
// Measured forward time in ms for a verify window of `width` rows, and for one proposal (width 0).
extern "C" double bolt_cost(Bolt* b, int width) {
    if (width == 0) return b->draft_ms;
    if (width < 1 || width > VERIFY_MAX) return 0;
    return b->cost_ms[width];
}
// Proposal ms for a call over all the sequences (0 when there is one).
extern "C" double bolt_draft_multi_cost(Bolt* b) { return b->draft_multi_ms; }
// Forward ms of shared blocks (several sequences' windows): fills rows[i], ms[i] and returns how many.
extern "C" int bolt_shared_costs(Bolt* b, int* rows, double* ms, int cap) {
    int n = 0;
    for (auto& e : b->shared_ms) {
        if (n >= cap) break;
        rows[n] = e.first;
        ms[n] = e.second;
        n++;
    }
    return n;
}

extern "C" void bolt_free(Bolt* b) { (void)b; /* the process owns the model until it exits */ }

// Start a sequence over: empty caches and state. Only the slots the first block reads are cleared.
extern "C" void bolt_reset_seq(Bolt* b, int seq) {
    Bolt::Seq& q = b->seqs[seq];
    const bool used = q.used;
    q = Bolt::Seq();
    q.used = used;
    for (Layer& ly : b->layers) if (ly.type == 0) {
        CK(cudaMemsetAsync(ly.ssm + (size_t)seq * (b->ssm_lazy ? 1 : 2) * SSM_ELEMS, 0, (size_t)SSM_ELEMS * 4, b->s0));
        CK(cudaMemsetAsync(ly.conv + (size_t)seq * CONV_SLOTS * CONV_ELEMS, 0, (size_t)CONV_ELEMS * 4, b->s0));
    }
    CK(cudaStreamSynchronize(b->s0));
    draft_reset(b->draft, seq);
}

// Sequence slots for callers that serve several streams: the lowest free one, started over (-1 when none).
extern "C" int bolt_seq_alloc(Bolt* b) {
    for (int i = 0; i < b->nseq; i++) {
        if (b->seqs[i].used) continue;
        bolt_reset_seq(b, i);
        b->seqs[i].used = true;
        return i;
    }
    return -1;
}
extern "C" void bolt_seq_release(Bolt* b, int seq) {
    if (seq < 0 || seq >= b->nseq) return;
    b->seqs[seq].used = false;
    kv_trim(b, seq);
}

extern "C" int bolt_len_seq(Bolt* b, int seq) { return b->seqs[seq].len; }

// One block of rows for each of nseg sequences, all verify windows (record): their state is advanced as if every row
// is kept; bolt_keep_seq cuts a sequence back. Candidates of every row are read back (rows in segment order).
extern "C" int bolt_forward_multi(Bolt* b, int nseg, const int* seqs, const int* rows, const int* tokens, int cand_k) {
    SegIn sg[SEG_MAX];
    if (nseg < 1 || nseg > SEG_MAX) return -1;
    int at = 0;
    for (int i = 0; i < nseg; i++) {
        sg[i] = {seqs[i], rows[i], tokens + at};
        at += rows[i];
    }
    return run_blocks(b, sg, nseg, 1, cand_k);
}

// Structured output: redraw n rows of the last forward among the tokens their masks allow (row rows[i] reads masks + i * words,
// bit t set = token t allowed) and replace those rows' candidates. A row with fewer than k allowed tokens ends in -3.4e38 values.
extern "C" int bolt_mask_rows(Bolt* b, int n, const int* rows, const uint32_t* masks, int words, int k) {
    if (n < 1 || n > LOGIT_ROWS || words < 1 || words > VOCAB / 32 || k < 1) return -1;
    if (k > CAND) k = CAND;
    for (int i = 0; i < n; i++) if (rows[i] < 0 || rows[i] >= LOGIT_ROWS) return -2;
    CK(cudaMemcpyAsync(b->msel, rows, (size_t)n * 4, cudaMemcpyHostToDevice, b->s0));
    CK(cudaMemcpyAsync(b->mmask, masks, (size_t)n * words * 4, cudaMemcpyHostToDevice, b->s0));
    k_candidates_masked(b->s0, b->logits, b->msel, b->mmask, words, n, k, b->mvals, b->mids, b->csv, b->csi);
    std::vector<float> v((size_t)n * CAND);
    std::vector<int> id((size_t)n * CAND);
    CK(cudaMemcpyAsync(v.data(), b->mvals, v.size() * 4, cudaMemcpyDeviceToHost, b->s0));
    CK(cudaMemcpyAsync(id.data(), b->mids, id.size() * 4, cudaMemcpyDeviceToHost, b->s0));
    CK(cudaStreamSynchronize(b->s0));
    for (int i = 0; i < n; i++) {
        memcpy(&b->cvals_h[(size_t)rows[i] * CAND], &v[(size_t)i * CAND], (size_t)k * 4);
        memcpy(&b->cids_h[(size_t)rows[i] * CAND], &id[(size_t)i * CAND], (size_t)k * 4);
    }
    return 0;
}

// Candidates of a row of the last forward, best first.
extern "C" void bolt_candidates(Bolt* b, int row, int k, float* vals, int* ids) {
    if (k > CAND) k = CAND;
    memcpy(vals, &b->cvals_h[(size_t)row * CAND], (size_t)k * 4);
    memcpy(ids, &b->cids_h[(size_t)row * CAND], (size_t)k * 4);
}

// Keep the first n rows of the sequence's last record block (idempotent for the same n). Returns whether a replay was
// launched on s0 (the caller synchronizes).
static bool keep_rows(Bolt* b, int seq, int n) {
    Bolt::Seq& q = b->seqs[seq];
    if (n < 0) n = 0;
    if (n > q.last_rows) n = q.last_rows;
    if (!q.last_record) return false;
    bool launched = false;
    if (b->ssm_lazy) {
        // the window wrote no state: its first n rows are folded in by the sequence's next block
        q.pending = n;
        q.last_rows = n;
    } else if (n < q.last_rows) {
        if (n > 0 && b->replay) {
            // the window kept only its final state: rebuild the one after row n from the pre-window state
            k_scan_replay(b->s0, b->replay, b->mamba_layers, seq, q.last_ssm, q.last_row0, n);
            launched = true;
        }
        q.ssm = n == 0 ? q.last_ssm : (q.last_ssm ^ 1);
        q.last_rows = n;
    }
    q.conv = (q.last_conv + n) % CONV_SLOTS;
    q.len = q.last_len + n;
    q.ing_n = n;
    return launched;
}

extern "C" void bolt_keep_seq(Bolt* b, int seq, int n) {
    if (keep_rows(b, seq, n)) CK(cudaStreamSynchronize(b->s0));
}

// The same for several sequences at once: their replays go out together and the host waits once.
extern "C" void bolt_keep_multi(Bolt* b, int n, const int* seqs, const int* counts) {
    bool launched = false;
    for (int i = 0; i < n; i++) {
        if (seqs[i] < 0 || seqs[i] >= b->nseq) continue;
        launched |= keep_rows(b, seqs[i], counts[i]);
    }
    if (launched) CK(cudaStreamSynchronize(b->s0));
}

// Lazy state: fold the pending rows into the sequence's state now (a snapshot copies the state after them).
static void fold_pending(Bolt* b, int seq) {
    Bolt::Seq& q = b->seqs[seq];
    if (!b->ssm_lazy || q.pending <= 0) return;
    k_scan_fold(b->s0, b->replay, b->mamba_layers, seq, q.hist, q.pending);
    q.pending = 0;
}

// Tell the drafter where the sequence's whole prompt ends: rows more than its window before that are never ingested.
extern "C" void bolt_prompt_end(Bolt* b, int seq, int total) {
    b->seqs[seq].ingest_floor = total > 1100 ? total - 1100 : 0;
}

// The prompt: chunks of TMAX rows. Afterwards row 0 of bolt_candidates is the last prompt row.
extern "C" int bolt_prefill_seq(Bolt* b, int seq, const int* tokens, int n, int cand_k) {
    int done = 0;
    while (done < n) {
        int t = n - done < TMAX ? n - done : TMAX;
        int last = done + t == n;
        int rc = run_block(b, seq, tokens + done, t, 0, last ? cand_k : 0, 1);
        if (rc) return rc;
        done += t;
    }
    return 0;
}


// A run of prompt tokens whose embeddings are rows already on the device: tokens[pos .. pos + count) of the call take
// the bf16 [count][4096] rows at `rows`.
struct BoltRowRun { int pos, count; const void* rows; };

// bolt_prefill_seq with image rows: the tokens in the runs are placeholders (any id); their rows are replaced.
extern "C" int bolt_prefill_seq_rows(Bolt* b, int seq, const int* tokens, int n, int cand_k, const BoltRowRun* runs, int nruns) {
    int done = 0;
    std::vector<int> toks(tokens, tokens + n);
    std::vector<EmbedPatch> patches;
    while (done < n) {
        const int t = n - done < TMAX ? n - done : TMAX;
        patches.clear();
        for (int i = 0; i < nruns; i++) {
            const int lo = std::max(runs[i].pos, done), hi = std::min(runs[i].pos + runs[i].count, done + t);
            if (hi <= lo) continue;
            patches.push_back({lo - done, (const uint16_t*)runs[i].rows + (size_t)(lo - runs[i].pos) * HID, hi - lo});
            for (int p = lo; p < hi; p++) toks[p] = -1;
        }
        const int last = done + t == n;
        const int rc = run_block(b, seq, toks.data() + done, t, 0, last ? cand_k : 0, 1, patches.data(), (int)patches.size());
        if (rc) return rc;
        done += t;
    }
    CK(cudaStreamSynchronize(b->s0));   // the caller reuses its rows buffer for the next piece
    return 0;
}

// The residual stream after the six drafter layers for one row of the last forward.
extern "C" void bolt_aux(Bolt* b, int row, float* out) {
    CK(cudaMemcpy(out, b->cap + (size_t)row * 6 * HID, 6 * HID * 4, cudaMemcpyDeviceToHost));
}

// ---- single-sequence API (sequence 0) for the bench tools ----

extern "C" void bolt_reset(Bolt* b) { bolt_reset_seq(b, 0); }
extern "C" int bolt_len(Bolt* b) { return b->seqs[0].len; }
extern "C" int bolt_propose(Bolt* b, int anchor, int* out) { return bolt_propose_seq(b, 0, anchor, out); }
extern "C" void bolt_keep(Bolt* b, int n) { bolt_keep_seq(b, 0, n); }
extern "C" int bolt_prefill(Bolt* b, const int* tokens, int n, int cand_k) {
    bolt_prompt_end(b, 0, b->seqs[0].len + n);
    return bolt_prefill_seq(b, 0, tokens, n, cand_k);
}
// One sequence's block of up to TMAX rows (record: a verify window of up to VERIFY_MAX rows).
extern "C" int bolt_forward_seq(Bolt* b, int seq, const int* tokens, int T, int record, int cand_k) { return run_block(b, seq, tokens, T, record, cand_k); }
// Run T rows. record keeps per-row mamba state so bolt_keep can roll back (T <= 8).
extern "C" int bolt_forward(Bolt* b, const int* tokens, int T, int record, int cand_k) { return run_block(b, 0, tokens, T, record, cand_k); }

extern "C" void bolt_push(Bolt* b, uint32_t token) {
    int t = (int)token;
    run_block(b, 0, &t, 1, 0, 1);
}

extern "C" int bolt_sample(Bolt* b) { return b->cids_h[0]; }

extern "C" float bolt_logprob(Bolt* b, int target) {
    std::vector<float> h(VOCAB);
    CK(cudaMemcpy(h.data(), b->logits, (size_t)VOCAB * 4, cudaMemcpyDeviceToHost));  // row 0 of the last block
    float m = -1e30f;
    for (float v : h) m = fmaxf(m, v);
    double s = 0;
    for (float v : h) s += exp((double)v - m);
    return (float)(h[target] - m - log(s));
}

// Log-probability of `target` under a row of the last block's logits.
extern "C" float bolt_logprob_row(Bolt* b, int row, int target) {
    std::vector<float> h(VOCAB);
    CK(cudaMemcpy(h.data(), b->logits + (size_t)row * VOCAB, (size_t)VOCAB * 4, cudaMemcpyDeviceToHost));
    float m = -1e30f;
    for (float v : h) m = fmaxf(m, v);
    double s = 0;
    for (float v : h) s += exp((double)v - m);
    return (float)(h[target] - m - log(s));
}

// Test hook: teacher-forced drafter accuracy over a known sequence. stats[j] counts slot j hits,
// stats[8 + a] counts blocks whose accepted prefix was a, stats[20] the blocks scored.
extern "C" void bolt_probe(Bolt* b, const int* toks, int n, int* stats) {
    for (int i = 0; i < n; i++) {
        if (i > 0 && i + 8 <= n) {
            int out[8];
            int r = bolt_propose(b, toks[i], out);
            if (r == 7) {
                int a = 0;
                for (int j = 0; j < 7; j++) if (out[j] == toks[i + 1 + j]) stats[j]++;
                while (a < 7 && out[a] == toks[i + 1 + a]) a++;
                stats[8 + a]++;
                stats[20]++;
            }
        }
        bolt_push(b, (uint32_t)toks[i]);
    }
}

// Time each verify width and one proposal so the round loop's cost model starts from measurements, and shared
// blocks of several sequences' windows.
static void bolt_calibrate(Bolt* b) {
    auto now = [] { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
    int toks[BATCH_MAX];
    for (int i = 0; i < BATCH_MAX; i++) toks[i] = 1000 + 37 * (i % 61);
    for (int w = 1; w <= VERIFY_MAX; w++) {
        double best = 1e30;
        for (int rep = 0; rep < 5; rep++) {
            double t0 = now();
            if (run_block(b, 0, toks, w, 1, 20)) break;
            double t1 = now();
            bolt_keep_seq(b, 0, 0);
            b->seqs[0].ing_n = 0;
            if (rep >= 2 && t1 - t0 < best) best = t1 - t0;
        }
        b->cost_ms[w] = best;
    }
    if (b->nseq > 1) {
        // total rows R as segments of up to VERIFY_MAX rows over distinct sequences, each with tokens of its own so
        // the rows route to different experts like real streams do
        int mix[BATCH_MAX];
        for (int i = 0; i < BATCH_MAX; i++) mix[i] = 1000 + (int)((i * 2654435761u + 12345u) % 90000u);
        auto shared = [&](int nseg, const int* rows) {
            int seqs[SEG_MAX];
            for (int i = 0; i < nseg; i++) seqs[i] = i;
            double best = 1e30;
            for (int rep = 0; rep < 5; rep++) {
                double t0 = now();
                if (bolt_forward_multi(b, nseg, seqs, rows, mix, 64)) break;
                double t1 = now();
                for (int i = 0; i < nseg; i++) { bolt_keep_seq(b, seqs[i], 0); b->seqs[seqs[i]].ing_n = 0; }
                if (rep >= 2 && t1 - t0 < best) best = t1 - t0;
            }
            return best;
        };
        const char* cal = getenv("BOLT_CAL");
        if (cal && !strcmp(cal, "spread")) {
            // R rows over as many sequences as there are (at most one row each below nseq), the costliest spread: the
            // recurrent state of every sequence in the block is read and written whatever its rows
            for (int R : {2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128}) {
                int nseg = R < b->nseq ? R : b->nseq;
                if (R > nseg * VERIFY_MAX) continue;
                int rows[SEG_MAX];
                for (int i = 0; i < nseg; i++) rows[i] = R / nseg + (i < R % nseg ? 1 : 0);
                double best = shared(nseg, rows);
                if (best < 1e29) b->shared_ms.push_back({R, best});
            }
        } else {
            // The round loop prices a shared round as this table at its total rows plus an overhead it measures per
            // stream count. So the table holds what rows cost, not streams: one stream's windows, then the cost of a
            // row measured with every sequence in the block (1 row each against 8 each: the same states, more rows),
            // and the state of each further stream lands in the per-count overhead the loop learns.
            double ms1 = 0, ms8 = 0;
            int r1 = 0, r8 = 0;
            for (int per : {1, 8}) {
                const int R = b->nseq * per;
                if (R > BATCH_MAX) break;
                int rows[SEG_MAX];
                for (int i = 0; i < b->nseq; i++) rows[i] = per;
                const double best = shared(b->nseq, rows);
                if (best >= 1e29) break;
                if (per == 1) { ms1 = best; r1 = R; } else { ms8 = best; r8 = R; }
            }
            double slope = r8 > r1 && ms8 > ms1 ? (ms8 - ms1) / (r8 - r1) : 0;
            if (slope <= 0) slope = (b->cost_ms[VERIFY_MAX] - b->cost_ms[1]) / (VERIFY_MAX - 1);
            if (slope < 0.1) slope = 0.1;
            fprintf(stderr, "bolt: shared rounds of %d streams: %d rows %.1f ms, %d rows %.1f ms; %.2f ms a row\n", b->nseq, r1, ms1, r8, ms8, slope);
            for (int R : {2, 4, 8}) b->shared_ms.push_back({R, b->cost_ms[R]});
            for (int R : {12, 16, 24, 32, 48, 64, 96, 128}) {
                if (R > b->nseq * VERIFY_MAX || R > BATCH_MAX) break;
                b->shared_ms.push_back({R, b->cost_ms[VERIFY_MAX] + slope * (R - VERIFY_MAX)});
            }
        }
    }
    if (b->draft) {
        // a short fake context for the drafter
        bolt_forward(b, toks, 8, 1, 1);
        bolt_keep(b, 8);
        double best = 1e30;
        for (int rep = 0; rep < 4; rep++) {
            int out[8];
            double t0 = now();
            bolt_propose(b, toks[0], out);
            double t1 = now();
            if (rep >= 1 && t1 - t0 < best) best = t1 - t0;
        }
        b->draft_ms = best;
        if (b->nseq > 1) {
            // every sequence needs a context of its own to propose from
            int seqs[SEG_MAX], anchors[SEG_MAX], out[SEG_MAX * 8], got[SEG_MAX];
            for (int i = 0; i < b->nseq; i++) {
                seqs[i] = i;
                anchors[i] = toks[0];
                if (i > 0) { bolt_forward_seq(b, i, toks, 8, 1, 1); bolt_keep_seq(b, i, 8); }
            }
            double bestn = 1e30;
            for (int rep = 0; rep < 4; rep++) {
                double t0 = now();
                bolt_propose_multi(b, b->nseq, seqs, anchors, out, got);
                double t1 = now();
                if (rep >= 1 && t1 - t0 < bestn) bestn = t1 - t0;
            }
            b->draft_multi_ms = bestn;
        }
    }
    for (int i = 0; i < b->nseq; i++) bolt_reset_seq(b, i);
    fprintf(stderr, "bolt: window ms");
    for (int w = 1; w <= VERIFY_MAX; w++) fprintf(stderr, " %d:%.1f", w, b->cost_ms[w]);
    fprintf(stderr, "; proposal %.1f ms", b->draft_ms);
    if (b->draft_multi_ms > 0) fprintf(stderr, " (%d streams %.1f ms)", b->nseq, b->draft_multi_ms);
    if (!b->shared_ms.empty()) {
        fprintf(stderr, "; shared rows");
        for (auto& e : b->shared_ms) fprintf(stderr, " %d:%.1f", e.first, e.second);
    }
    fprintf(stderr, "\n");
}

extern "C" void bolt_prof_report(Bolt* b) {
    draft_prof_report();
    std::vector<std::pair<std::string, std::pair<double, int>>> v(b->prof_acc.begin(), b->prof_acc.end());
    std::sort(v.begin(), v.end(), [](auto& x, auto& y) { return x.second.first > y.second.first; });
    double total = 0;
    for (auto& e : v) total += e.second.first;
    fprintf(stderr, "bolt prof: total %.2f ms over %d blocks of %.1f rows\n", total, b->prof_rows_n, b->prof_rows_n ? (double)b->prof_rows / b->prof_rows_n : 0.0);
    b->prof_rows = 0; b->prof_rows_n = 0;
    if (b->moe_stat) {
        int st[2] = {0, 0};
        CK(cudaMemcpy(st, b->moe_stat, 8, cudaMemcpyDeviceToHost));
        CK(cudaMemset(b->moe_stat, 0, 8));
        auto it = b->prof_acc.find("moe.up");
        const int launches = it == b->prof_acc.end() ? 0 : it->second.second;
        if (launches > 0) fprintf(stderr, "  moe: %.1f distinct experts and %.1f work items a layer\n", (double)st[0] / launches, (double)st[1] / launches);
    }
    for (auto& e : v) fprintf(stderr, "  %-18s %9.2f ms %5.1f%%  %6d calls  %.1f us each\n", e.first.c_str(), e.second.first, 100 * e.second.first / total, e.second.second, 1000 * e.second.first / e.second.second);
    b->prof_acc.clear();
}

// ---- prompt-cache snapshots: the state after `at` prompt tokens, copied out and back ----

struct Snap {
    int at = 0, draft_clen = 0;
    float *ssm = nullptr, *conv = nullptr;
    uint8_t* kv = nullptr;   // per attention layer: K rows [0, at), V rows, then (fp8) their scales
    void* draft = nullptr;
    size_t bytes = 0;
};

extern "C" uint64_t bolt_snap_bytes(Bolt* b, int at) {
    return (uint64_t)b->mamba_layers * (SSM_ELEMS + CONV_ELEMS) * 4 + (uint64_t)at * kv_token_bytes(b) + (b->draft ? draft_blob_bytes() : 0);
}

extern "C" void* bolt_snap_save_seq(Bolt* b, int seq, int at) {
    Bolt::Seq& q = b->seqs[seq];
    if (at != q.len || at <= 0) return nullptr;
    flush_ingest(b, seq);
    fold_pending(b, seq);
    const size_t slots = b->ssm_lazy ? 1 : 2;
    Snap* sn = new Snap();
    sn->at = at;
    sn->draft_clen = draft_clen(b->draft, seq);
    sn->bytes = bolt_snap_bytes(b, at);
    const size_t kvb = (size_t)at * b->kv_row, scb = b->kv_fp8 ? (size_t)at * 2 * 4 : 0, lay = 2 * (kvb + scb);
    CK(cudaMalloc(&sn->ssm, (size_t)b->mamba_layers * SSM_ELEMS * 4));
    CK(cudaMalloc(&sn->conv, (size_t)b->mamba_layers * CONV_ELEMS * 4));
    CK(cudaMalloc(&sn->kv, lay * 8));
    if (b->draft) CK(cudaMalloc(&sn->draft, draft_blob_bytes()));
    int mi = 0, ai = 0;
    const size_t kv0 = (size_t)seq * b->kv_stride;   // tokens
    for (Layer& ly : b->layers) {
        if (ly.type == 0) {
            CK(cudaMemcpyAsync(sn->ssm + (size_t)mi * SSM_ELEMS, ly.ssm + ((size_t)seq * slots + q.ssm) * SSM_ELEMS, (size_t)SSM_ELEMS * 4, cudaMemcpyDeviceToDevice, b->s0));
            CK(cudaMemcpyAsync(sn->conv + (size_t)mi * CONV_ELEMS, ly.conv + ((size_t)seq * CONV_SLOTS + q.conv) * CONV_ELEMS, (size_t)CONV_ELEMS * 4, cudaMemcpyDeviceToDevice, b->s0));
            mi++;
        } else if (ly.type == 2) {
            uint8_t* dst = sn->kv + (size_t)ai * lay;
            CK(cudaMemcpyAsync(dst, (uint8_t*)ly.kc + kv0 * b->kv_row, kvb, cudaMemcpyDeviceToDevice, b->s0));
            CK(cudaMemcpyAsync(dst + kvb, (uint8_t*)ly.vc + kv0 * b->kv_row, kvb, cudaMemcpyDeviceToDevice, b->s0));
            if (b->kv_fp8) {
                CK(cudaMemcpyAsync(dst + 2 * kvb, ly.ks + kv0 * 2, scb, cudaMemcpyDeviceToDevice, b->s0));
                CK(cudaMemcpyAsync(dst + 2 * kvb + scb, ly.vs + kv0 * 2, scb, cudaMemcpyDeviceToDevice, b->s0));
            }
            ai++;
        }
    }
    draft_save(b->draft, seq, sn->draft, b->s0);
    CK(cudaStreamSynchronize(b->s0));
    return sn;
}

// Make the sequence's live state the snapshot's. Returns 0 on success.
extern "C" int bolt_snap_restore_seq(Bolt* b, int seq, void* handle) {
    Snap* sn = (Snap*)handle;
    if (!sn || sn->at > b->context) return -1;
    Bolt::Seq& q = b->seqs[seq];
    if (!kv_ensure(b, seq, sn->at)) return -1;
    const size_t kvb = (size_t)sn->at * b->kv_row, scb = b->kv_fp8 ? (size_t)sn->at * 2 * 4 : 0, lay = 2 * (kvb + scb);
    int mi = 0, ai = 0;
    const size_t kv0 = (size_t)seq * b->kv_stride;
    const size_t slots = b->ssm_lazy ? 1 : 2;
    q.ssm = 0;
    q.conv = 0;
    q.pending = 0;
    for (Layer& ly : b->layers) {
        if (ly.type == 0) {
            CK(cudaMemcpyAsync(ly.ssm + (size_t)seq * slots * SSM_ELEMS, sn->ssm + (size_t)mi * SSM_ELEMS, (size_t)SSM_ELEMS * 4, cudaMemcpyDeviceToDevice, b->s0));
            CK(cudaMemcpyAsync(ly.conv + (size_t)seq * CONV_SLOTS * CONV_ELEMS, sn->conv + (size_t)mi * CONV_ELEMS, (size_t)CONV_ELEMS * 4, cudaMemcpyDeviceToDevice, b->s0));
            mi++;
        } else if (ly.type == 2) {
            const uint8_t* src = sn->kv + (size_t)ai * lay;
            CK(cudaMemcpyAsync((uint8_t*)ly.kc + kv0 * b->kv_row, src, kvb, cudaMemcpyDeviceToDevice, b->s0));
            CK(cudaMemcpyAsync((uint8_t*)ly.vc + kv0 * b->kv_row, src + kvb, kvb, cudaMemcpyDeviceToDevice, b->s0));
            if (b->kv_fp8) {
                CK(cudaMemcpyAsync(ly.ks + kv0 * 2, src + 2 * kvb, scb, cudaMemcpyDeviceToDevice, b->s0));
                CK(cudaMemcpyAsync(ly.vs + kv0 * 2, src + 2 * kvb + scb, scb, cudaMemcpyDeviceToDevice, b->s0));
            }
            ai++;
        }
    }
    draft_restore(b->draft, seq, sn->draft, sn->draft_clen, b->s0);
    CK(cudaStreamSynchronize(b->s0));
    q.len = sn->at;
    q.last_rows = 0;
    q.ing_n = 0;
    return 0;
}

extern "C" void* bolt_snap_save(Bolt* b, int at) { return bolt_snap_save_seq(b, 0, at); }
extern "C" int bolt_snap_restore(Bolt* b, void* handle) { return bolt_snap_restore_seq(b, 0, handle); }

extern "C" void bolt_snap_drop(Bolt*, void* handle) {
    Snap* sn = (Snap*)handle;
    if (!sn) return;
    cudaFree(sn->ssm); cudaFree(sn->conv); cudaFree(sn->kv); cudaFree(sn->draft);
    delete sn;
}
