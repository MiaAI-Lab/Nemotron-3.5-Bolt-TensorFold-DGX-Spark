# Shared settings for start.sh, stop.sh and scripts/*.sh. A setting's value comes from the first of these that sets it:
#   1. the environment: `PORT=9000 ./start.sh`, `PULL=0 scripts/prepare.sh`
#   2. ./.env (KEY=value lines, read, never run)
if [[ -f .env ]]; then
  while IFS= read -r _line || [[ -n "$_line" ]]; do
    [[ "$_line" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
    _key=${BASH_REMATCH[2]}; _value=${BASH_REMATCH[3]}
    if [[ "$_value" =~ ^\"([^\"]*)\"[[:space:]]*(#.*)?$ || "$_value" =~ ^\'([^\']*)\'[[:space:]]*(#.*)?$ ]]; then
      _value=${BASH_REMATCH[1]}
    else
      _value=${_value%%#*}; _value=${_value%"${_value##*[![:space:]]}"}
    fi
    [[ -n "${!_key+set}" ]] || export "$_key=$_value"
  done < .env
fi
unset _line _key _value

# NVIDIA's packed 24GB export. Revision a326822a is the Hub commit this contract was read from.
MODEL_ID="${MODEL_ID:-nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB}"
MODEL_REVISION="${MODEL_REVISION:-a326822af111290f2af5ac59ef4b5040347dabc4}"
CKPT_GIB=23
QUANT_LABEL="IQ2_XXS+NVFP4+Q8_0"
# NVIDIA's LiLiCorr drafter for this target (DFlash backbone plus a candidate-lattice head, 1.3 GB). DRAFTS=1 loads it.
DRAFT_ID="${DRAFT_ID:-nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr}"
DRAFT_REVISION="${DRAFT_REVISION:-4c6a5613b6d887194a3a5493e3a18a5b3930bb86}"

# TensorFold v1.1.4 (its RADIO tower, image and video input, structured output for omni checkpoints). patches/
# backport upstream CUDA build fixes and register libbolt. Zig ZIG_VERSION builds the native CUDA server.
TF_REPO="${TF_REPO:-https://github.com/ashhart/TensorFold.git}"
TF_REF="${TF_REF:-eab8582819688325e371907e4706e1c602b8a273}"
ZIG_VERSION="${ZIG_VERSION:-0.17.0}"
ZIG_SHA256="${ZIG_SHA256:-9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8}"
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
TF_TAG="zig-${TF_REF:0:7}"
IMAGE="${IMAGE:-tensorfold-nemotron-bolt:${TF_TAG}}"
image_hash() {
  ( cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
    cat patches/*.patch engine/*.cu engine/*.h engine/*.inc engine/*.zig 2>/dev/null
    sed -n "/<<'DOCKERFILE'/,/^DOCKERFILE\$/p" scripts/prepare.sh
    echo "$TF_REPO $TF_REF zig-$ZIG_VERSION $ZIG_SHA256 $BASE_IMAGE nemotron-bolt"
  ) | sha256sum | cut -c1-12
}
GHCR_IMAGE="${GHCR_IMAGE:-ghcr.io/miaai-lab/nemotron-3.5-bolt-tensorfold-dgx-spark}"
IMAGE_TAG="${IMAGE_TAG:-}"
IMAGE_DIGEST="${IMAGE_DIGEST:-}"
prebuilt_image() {
  local tag="${TF_TAG}-$(image_hash)"
  if [[ "$tag" == "$IMAGE_TAG" && -n "$IMAGE_DIGEST" ]]; then echo "$GHCR_IMAGE@$IMAGE_DIGEST"; else echo "$GHCR_IMAGE:$tag"; fi
}
CONTAINER_NAME="${CONTAINER_NAME:-nemotron35-bolt-tf}"

SERVED_NAME="${SERVED_NAME:-Nemotron-3.5-Bolt}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8888}"
# Native window from config.json max_position_embeddings, for every stream. A stream's bf16 attention cache is mapped as
# it grows (8 KB a token), so sixteen streams at the full window cost what they have used, not 2 GiB each.
CONTEXT="${CONTEXT:-262144}"
# Requests decoded together, 1 to 16. Each holds 0.4 GiB of recurrent state from the start; their windows go into one
# forward, so the weights are read once for all of them (docs/engine.md).
PARALLEL="${PARALLEL:-16}"
# Prompt cache (TensorFold's prefix reuse): GiB of kept prompt states. A state is the Mamba state (165 MB), 8 KB a token
# of attention rows and the drafter's context (21 MB); a repeated system prompt or the history of a chat is not prefilled
# again. 0 turns it off. This memory is outside TensorFold's own accounting: it comes from the free memory below.
PROMPT_CACHE_GIB="${PROMPT_CACHE_GIB:-8}"
# The checkpoint has no MTP head. Drafts are the separate LiLiCorr checkpoint, run inside libbolt and driven by
# TensorFold's round loop: windows of up to eight rows, rolled back to the accepted prefix. DRAFTS=0 serves one row at a time.
DRAFTS="${DRAFTS:-1}"
# bf16, or fp8: K and V as e4m3 bytes with an fp32 scale per token and head (264 B instead of 512 B a layer and token).
KV_DTYPE="${KV_DTYPE:-bf16}"
# 1 maps every stream's whole CONTEXT window at load and keeps it (PARALLEL x CONTEXT x 8 KB in bf16, 4.1 KB in fp8), so the
# memory is certain and a start that cannot have it fails at once. 0 maps it as streams grow and gives it back.
KV_PIN="${KV_PIN:-0}"
# 1 keeps one copy of each stream's Mamba state instead of two: a verify window writes no state, and the rows it keeps are
# folded in by the stream's next forward from the scan inputs it saved, so a partial accept costs no replay pass and a
# stream holds 0.28 GiB from load instead of 0.41. Measured on a Spark: the same replies token for token, 3 to 8% more
# tokens a second at 4 to 16 streams, no change alone (BOLT_SSM_LAZY). 0 is the two-copy path.
SSM_LAZY="${SSM_LAZY:-1}"
# 1 stores the LiLiCorr drafter's attention and kernel projections as Q8_0 (half the bytes a proposal reads); the target
# checks every draft, so this changes what is proposed, never the reply. 0 keeps them bf16 (BOLT_DRAFT_Q8). Measured: the
# same replies and the same speed within noise either way; Q8_0 keeps 200 MB of memory.
# BOLT_CAL=spread (passed through when set) calibrates the planner's shared-round table the 1.0.12 way, by spread rows.
DRAFT_Q8="${DRAFT_Q8:-1}"
# 1 runs the prompt pass's NVFP4 projections (in_proj, out_proj) on the fp4 tensor cores with the rows quantized to NVFP4 as
# vLLM's NVFP4 path does: prefill +17 to 20 %, perplexity +0.8 % on the harness (prose +1.2 %, code +0.4 %); the first
# token of a prompt was unchanged in every test, later greedy tokens can differ. 0 keeps the fp16 rows (BOLT_NV4_FP4).
NV4_FP4="${NV4_FP4:-1}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
THINKING="${THINKING:-1}"
# generation_config.json: temperature 1.0, top_p 0.95. TOP_K empty omits --top-k.
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-}"
# Images and video: TensorFold's RADIO tower (1.7 GB of weights, loaded when VISION=1). Pictures arrive as base64 data URLs
# (VISION_URLS=1 also fetches public HTTPS URLs). A request holds up to VISION_MAX_IMAGES images and VISION_MAX_VIDEOS videos;
# the token budgets are shared by a request's images (VISION_IMAGE_TOKENS) and its videos (VISION_VIDEO_TOKENS).
# VISION=0 serves text only and skips the weights.
VISION="${VISION:-1}"
VISION_URLS="${VISION_URLS:-0}"
VISION_MAX_IMAGES="${VISION_MAX_IMAGES:-64}"
VISION_MAX_VIDEOS="${VISION_MAX_VIDEOS:-4}"
VISION_IMAGE_TOKENS="${VISION_IMAGE_TOKENS:-262144}"
VISION_VIDEO_TOKENS="${VISION_VIDEO_TOKENS:-131072}"
export TENSORFOLD_MEMORY_RESERVE_GIB="${TENSORFOLD_MEMORY_RESERVE_GIB:-10}"
export TENSORFOLD_NO_UPDATE_CHECK="${TENSORFOLD_NO_UPDATE_CHECK:-1}"
# SERVE=0 stops after the checkpoint check.
SERVE="${SERVE:-1}"

# Weights on the Hub at MODEL_REVISION: usedStorage 24,102,448,875 bytes (22.45 GiB). CKPT_GIB rounds that up.
# One token's bf16 attention cache is 8 layers * 2 kv heads * 128 * 2 bytes * 2 (K and V) = 8192 bytes.
# Mamba state is per sequence: 160 MiB of SSM state in fp32, two copies (a verify window reads one and writes the other),
# plus the conv ring, 0.41 GiB a stream (0.28 with SSM_LAZY=1), allocated for all PARALLEL streams at load.
MEM_NEED_GIB="${MEM_NEED_GIB:-}"
MEM_FLOOR_GIB="${MEM_FLOOR_GIB:-$TENSORFOLD_MEMORY_RESERVE_GIB}"
kv_gib() {
  local ctx=${1:-$CONTEXT} bytes=8192
  [[ "$KV_DTYPE" == fp8 ]] && bytes=4224
  [[ "$ctx" =~ ^[0-9]+$ ]] || ctx=$CONTEXT
  echo $(( (bytes * ctx + 1073741823) / 1073741824 ))
}
# What a start needs free: the checkpoint, every stream's recurrent state, a working set of attention rows (the rest is
# mapped as streams grow, down to BOLT_KV_RESERVE_GIB left free), TensorFold's reserve and the prompt cache.
# Vision adds the tower and projector (1.7 GB) and a scratch that grows with the largest picture (about 1 GiB at 3,000 px).
mem_need_gib() {
  local ctx=${1:-$CONTEXT} vis=0 per=45
  [[ "$VISION" == 1 ]] && vis=3
  [[ "$SSM_LAZY" == 1 ]] && per=30   # hundredths of a GiB a stream holds from load
  (( ctx > 65536 )) && ctx=65536
  if [[ "$KV_PIN" == 1 ]]; then ctx=${1:-$CONTEXT}; echo $(( vis + CKPT_GIB + (PARALLEL * per + 99) / 100 + PARALLEL * $(kv_gib "$ctx") + MEM_FLOOR_GIB + 2 + PROMPT_CACHE_GIB )); return; fi
  echo $(( vis + CKPT_GIB + (PARALLEL * per + 99) / 100 + $(kv_gib "$ctx") + MEM_FLOOR_GIB + 2 + PROMPT_CACHE_GIB ))
}

HF_CACHE="${HF_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}}"
KERNEL_CACHE="${KERNEL_CACHE:-$HOME/.cache/tensorfold-nemotron-bolt}"
STATE_DIR="${STATE_DIR:-$HOME/.local/state/nemotron35-bolt-tf}"
LOG_DIR="${LOG_DIR:-$HOME/.cache/tensorfold-nemotron-bolt/logs}"
LOG_KEEP="${LOG_KEEP:-10}"
MIN_FREE_GB="${MIN_FREE_GB:-40}"
IMAGE_FREE_GB="${IMAGE_FREE_GB:-40}"

_c() { [[ -t "$1" ]] && printf '\033[%sm' "$2" || true; }
log()  { printf '%s[%s]%s %s\n' "$(_c 1 '1;36')" "$(basename "$0")" "$(_c 1 0)" "$*"; }
warn() { printf '%s[%s] WARN:%s %s\n' "$(_c 2 '1;33')" "$(basename "$0")" "$(_c 2 0)" "$*" >&2; }
die()  { printf '%s[%s] ERROR:%s %s\n' "$(_c 2 '1;31')" "$(basename "$0")" "$(_c 2 0)" "$*" >&2; exit 1; }

[[ -z "${HF_TOKEN:-}" ]] || export HF_TOKEN
model_cache_dir() { echo "$HF_CACHE/hub/models--${MODEL_ID//\//--}"; }
snapshot_rev() {
  if [[ -n "$MODEL_REVISION" ]]; then echo "$MODEL_REVISION"
  else cat "$(model_cache_dir)/refs/main" 2>/dev/null; fi
}

# 0 when the checkpoint (every shard its index names) and, with DRAFTS=1, the drafter are in the Hugging Face cache.
# start.sh runs prepare.sh, which downloads what is missing, when this fails.
weights_ready() {
  local snap rev
  rev=$(snapshot_rev); [[ -n "$rev" ]] || return 1
  snap="$(model_cache_dir)/snapshots/$rev"
  python3 -I -c '
import json, os, sys
d = sys.argv[1]
for f in ("config.json", "tokenizer.json", "model.safetensors.index.json"):
    if not os.path.exists(os.path.join(d, f)): sys.exit(1)
shards = set(json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"].values())
sys.exit(0 if shards and all(os.path.exists(os.path.join(d, f)) for f in shards) else 1)
' "$snap" || return 1
  [[ "${DRAFTS:-1}" != 1 ]] || [[ -f "$HF_CACHE/hub/models--${DRAFT_ID//\//--}/snapshots/$DRAFT_REVISION/model.safetensors" ]]
}

PREPARED_MARKER="$STATE_DIR/prepared"
prepared_state() {
  local hash label rev
  hash=$(image_hash)
  label=$(docker image inspect -f '{{.Id}}' "$IMAGE" 2>/dev/null || echo missing)
  rev=$(snapshot_rev)
  echo "model=$MODEL_ID@$rev image=$label patches=$hash"
}
