#!/usr/bin/env bash
# Serve Nemotron 3.5 Bolt 74B-A9B 24GB with TensorFold's Zig engine on one DGX Spark.
# ./start.sh prepares the image and the checkpoints, checks the snapshot, launches tensorfold-native (the Bolt backend in
# engine/bolt_native.zig over libbolt.so) and sends a smoke request. SERVE=0 stops after the check.
#
# Usage: ./start.sh [restart] [extra tensorfold-native serve args]
#   ./start.sh                         # prepare if needed, check, launch
#   SERVE=0 ./start.sh                 # prepare and check only
#   ./start.sh restart --context 131072
#   DRY_RUN=1 ./start.sh               # print the docker command and exit
# Extra arguments go to tensorfold-native serve after the defaults, so they win.
# Settings, from the environment or ./.env (scripts/config.sh):
#   serving  CONTEXT, PARALLEL, MAX_TOKENS, THINKING, DRAFTS, TEMPERATURE, TOP_P, TOP_K, SERVED_NAME, HOST, PORT, SERVE
#   files    MODEL_ID, MODEL_REVISION, HF_CACHE, KERNEL_CACHE, STATE_DIR
#   image    IMAGE, TF_REPO, TF_REF, ZIG_VERSION, BASE_IMAGE, GHCR_IMAGE, CONTAINER_NAME
#   setup    PREPARE (auto | 1 | 0), PULL, FOREGROUND=1, WAIT_TIMEOUT, DRY_RUN=1, MEM_NEED_GIB, MEM_CHECK=0
#   engine   every TENSORFOLD_* variable goes into the container (TENSORFOLD_API_KEY by name only)
#            KV_DTYPE is bf16 (default) or fp8 (e4m3 with a scale per token and head: half the KV memory). VISION=1 (default) loads the image and video tower; VISION_MAX_IMAGES / _MAX_VIDEOS / _IMAGE_TOKENS / _VIDEO_TOKENS size it.
#            SSM_LAZY=0 keeps two Mamba state copies a stream (see scripts/config.sh); DRAFT_Q8=0 keeps the drafter's projections bf16; NV4_FP4=0 keeps the prompt pass's projections on fp16 rows.
#            WINDOW is 32 (default), 16 or 8: the rows a verify window carries at most, so a copy of earlier text is checked in one longer forward.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
source ./scripts/config.sh

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; }
WAIT_TIMEOUT="${WAIT_TIMEOUT:-1800}"
MODE=start
case "${1:-}" in
  restart) MODE=restart; shift ;;
  help) usage; exit 0 ;;
esac
for arg in "$@"; do [[ "$arg" == -h || "$arg" == --help ]] && { usage; exit 0; }; done

[[ "$CONTEXT" =~ ^[1-9][0-9]*$ && "$CONTEXT" -le 262144 ]] || die "CONTEXT is a token count up to 262144, not $CONTEXT"
[[ "$PARALLEL" =~ ^([1-9]|1[0-6])$ ]] || die "PARALLEL is 1 to 16, not $PARALLEL"
[[ "$MAX_TOKENS" =~ ^[1-9][0-9]*$ ]] || die "MAX_TOKENS is a token count, not $MAX_TOKENS"
for v in THINKING DRAFTS VISION VISION_URLS SERVE KV_PIN SSM_LAZY DRAFT_Q8 NV4_FP4; do [[ "${!v}" =~ ^[01]$ ]] || die "$v is 0 or 1, not ${!v}"; done
[[ "$KV_DTYPE" =~ ^(bf16|fp8)$ ]] || die "KV_DTYPE is bf16 or fp8, not $KV_DTYPE (the engine has no int8 or int4 cache)"
[[ "$WINDOW" =~ ^(8|16|32)$ ]] || die "WINDOW is 8, 16 or 32, not $WINDOW"
DRY=0; [[ "${DRY_RUN:-0}" == 1 ]] && DRY=1

# The native server refuses --kv-dtype ("not served by the native engine yet"). KV stays the engine's own.
SERVE_ARGS=(--context "$CONTEXT" --parallel "$PARALLEL" --max-tokens "$MAX_TOKENS"
            --temperature "$TEMPERATURE" --top-p "$TOP_P" --prompt-cache-gib "$PROMPT_CACHE_GIB")
[[ -n "$TOP_K" ]] && SERVE_ARGS+=(--top-k "$TOP_K")
if [[ "$THINKING" == 1 ]]; then SERVE_ARGS+=(--thinking); else SERVE_ARGS+=(--no-thinking); fi
[[ "$DRAFTS" == 1 ]] || SERVE_ARGS+=(--no-drafts)
if [[ "$VISION" == 1 ]]; then
  SERVE_ARGS+=(--vision --vision-max-images "$VISION_MAX_IMAGES" --vision-max-videos "$VISION_MAX_VIDEOS"
               --vision-image-tokens "$VISION_IMAGE_TOKENS" --vision-video-tokens "$VISION_VIDEO_TOKENS")
  [[ "$VISION_URLS" == 1 ]] && SERVE_ARGS+=(--vision-urls)
fi
ENV_DRAFT=(); [[ "$DRAFTS" == 1 ]] && ENV_DRAFT=(-e "BOLT_DRAFT=/root/.cache/huggingface/hub/models--${DRAFT_ID//\//--}/snapshots/$DRAFT_REVISION")
USER_ARGS=("$@")
arg_value() {
  local flag=$1 value="" i
  local -a all=("${SERVE_ARGS[@]}" "${USER_ARGS[@]}")
  for (( i = 0; i < ${#all[@]}; i++ )); do
    case "${all[i]}" in
      "$flag") value="${all[i + 1]:-}" ;;
      "$flag="*) value="${all[i]#*=}" ;;
    esac
  done
  echo "$value"
}
# libbolt keeps one sequence of state per stream (--parallel, which the backend passes on); its cache type and pinning,
# the Mamba state's copies and the drafter's weight format
ENV_SEQS=(-e "BOLT_KV=$KV_DTYPE" -e "BOLT_KV_PIN=$KV_PIN" -e "BOLT_SSM_LAZY=$SSM_LAZY" -e "BOLT_DRAFT_Q8=$DRAFT_Q8" -e "BOLT_NV4_FP4=$NV4_FP4" -e "BOLT_WINDOW=$WINDOW")
API_HOST="$HOST"; [[ "$HOST" == 0.0.0.0 || "$HOST" == "::" ]] && API_HOST=127.0.0.1
[[ "$API_HOST" == *:* ]] && API_HOST="[$API_HOST]"
URL="http://$API_HOST:$PORT"

running() { [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" == true ]]; }
served_name() {
  curl -s --max-time 5 "$URL/v1/models" 2>/dev/null |
    python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null
}

B=$'\033[1m'; M=$'\033[1;35m'; G=$'\033[1;32m'; D=$'\033[2m'; R=$'\033[0m'
[[ -t 1 ]] || { B=; M=; G=; D=; R=; }
source ./scripts/banner.sh
echo
banner
printf '\n%s  Mia'"'"'s TensorFold Start Script%s\n' "$M" "$R"
printf '%s  %s · %s at once · %s-token window · %s KV · vision %s · drafts %s · port %s%s\n\n' "$D" "$MODEL_ID" \
  "$(arg_value --parallel)" "$(arg_value --context)" "$KV_DTYPE" \
  "$( [[ "$VISION" == 1 ]] && echo on || echo off)" \
  "$( [[ "$DRAFTS" == 1 ]] && echo on || echo off)" "$PORT" "$R"
STEPS=5
step() { printf '%s[%s/%s]%s %s%s%s\n' "$M" "$1" "$STEPS" "$R" "$B" "$2" "$R"; }

command -v docker >/dev/null || die "docker is not installed"
mkdir -p "$KERNEL_CACHE" "$STATE_DIR"
exec 8>"$STATE_DIR/start.lock"
flock -n 8 || die "another ./start.sh is already running; wait for it to finish"
(( DRY )) && log "DRY_RUN=1: printing the docker command; nothing is stopped or started"

if (( ! DRY )) && [[ "$MODE" == start ]] && running && [[ "$SERVE" == 1 ]]; then
  log "$CONTAINER_NAME is already running (model: $(served_name || echo "not answering yet"), port $PORT): nothing to do."
  log "Use ./start.sh restart to restart it, or ./stop.sh to stop it."
  exit 0
fi

step 1 "Setup: image and checkpoint"
need_prepare=0
if [[ "${PREPARE:-auto}" == 1 ]]; then need_prepare=1
elif [[ "${PREPARE:-auto}" != 0 ]]; then
  [[ "$(prepared_state 2>/dev/null)" == "$(cat "$PREPARED_MARKER" 2>/dev/null)" ]] || need_prepare=1
  weights_ready || need_prepare=1
fi
if (( need_prepare )); then
  if (( DRY )); then log "DRY_RUN: scripts/prepare.sh would run now"
  else
    weights_ready || log "The weights are not all in $HF_CACHE/hub: prepare.sh downloads them (about 23 GiB, resumable)"
    log "Not ready yet: running scripts/prepare.sh"
    ./scripts/prepare.sh
  fi
else
  log "Ready: $IMAGE (patches $(image_hash)) and $MODEL_ID"
fi
why="scripts/prepare.sh did not"; [[ "${PREPARE:-auto}" == 0 ]] && why="PREPARE=0 skipped scripts/prepare.sh, which would"
(( DRY )) && why="DRY_RUN: scripts/prepare.sh would"
missing() { if (( DRY )); then warn "$1"; else die "$1"; fi; }
have_image=1
docker image inspect "$IMAGE" >/dev/null 2>&1 || { have_image=0; missing "image $IMAGE missing: $why build it"; }
if (( have_image )); then
  gate=$(docker image inspect -f '{{index .Config.Labels "tf.gate"}}' "$IMAGE" 2>/dev/null || true)
  [[ "$gate" == bolt ]] || missing "$IMAGE has no Bolt gate (tf.gate=${gate:-?}): scripts/prepare.sh --rebuild"
fi
KCACHE=$(docker image inspect -f '{{index .Config.Labels "tf.patches"}}' "$IMAGE" 2>/dev/null || true)
[[ "$KCACHE" =~ ^[0-9a-f]{12}$ ]] || KCACHE=$(image_hash)
rev=$(snapshot_rev)
sub="hub/models--${MODEL_ID//\//--}/snapshots/$rev"
if [[ ! -f "$HF_CACHE/$sub/config.json" ]]; then missing "$MODEL_ID @ ${rev:0:8} not in $HF_CACHE: $why download it"; fi
MODEL_ARG="/root/.cache/huggingface/$sub"
if [[ -f "$HF_CACHE/$sub/config.json" ]]; then
  info=$(python3 -I scripts/ckptcheck.py "$HF_CACHE/$sub" 2>&1) || missing "checkpoint check failed: $info"
  log "Checkpoint OK: $info"
fi

step 2 "Checks: arguments, previous server, port, memory"
if (( have_image )); then
  code=0
  docker run --rm --network none --entrypoint /opt/tensorfold/native/bin/tensorfold-native "$IMAGE" \
    serve /nonexistent/argument-check --name "$SERVED_NAME" --host "$HOST" --port "$PORT" "${SERVE_ARGS[@]}" "${USER_ARGS[@]}" \
    >/dev/null 2>"$STATE_DIR/args.err" || code=$?
  if (( code == 2 )) || ! grep -q 'neither a directory' "$STATE_DIR/args.err"; then
    if (( DRY )); then warn "DRY_RUN: tensorfold-native rejects these arguments: $(grep -v '^usage:' "$STATE_DIR/args.err" | tail -1)"
    else cat "$STATE_DIR/args.err" >&2; die "tensorfold-native serve rejects these arguments; nothing was changed"; fi
  fi
fi
if (( DRY )); then
  running && log "DRY_RUN: $CONTAINER_NAME is running; a real start would stop it first"
elif [[ "$MODE" == restart ]] && running; then
  ./stop.sh
elif [[ "$MODE" == start ]] && running; then
  log "A server was left running: stopping it, then starting"
  ./stop.sh
fi
if (( ! DRY )) && docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  log "Removing the previous container $CONTAINER_NAME"
  mkdir -p "$LOG_DIR"
  docker logs "$CONTAINER_NAME" > "$LOG_DIR/$(date +%Y%m%d-%H%M%S).log" 2>&1 || true
  docker rm -f "$CONTAINER_NAME" >/dev/null
fi
if ss -ltn "sport = :$PORT" 2>/dev/null | grep -q LISTEN; then
  (( DRY )) && log "DRY_RUN: port $PORT is in use" ||
    die "port $PORT is already in use: $(ss -ltnp "sport = :$PORT" 2>/dev/null | tail -n +2)"
fi
need_gib=${MEM_NEED_GIB:-$(mem_need_gib "$(arg_value --context)")}
avail_gb=$(awk '/^MemAvailable:/ {print int($2 / 1048576)}' /proc/meminfo)
if (( avail_gb >= need_gib )); then
  log "Arguments OK, port $PORT free, ${avail_gb} GiB memory available (this window needs ~$need_gib at start)"
else
  msg="only ${avail_gb} GiB memory available; a $(arg_value --context)-token window needs ~$need_gib (MemAvailable)"
  if (( DRY )) || [[ "${MEM_CHECK:-1}" == 0 ]]; then warn "$msg"; else die "$msg (MEM_CHECK=0 starts anyway)"; fi
fi

if [[ "$SERVE" != 1 ]]; then
  printf '\n%s  Checkpoint matches the contract. ./start.sh (SERVE=1) launches tensorfold-native.%s\n\n' "$B" "$R"
  exit 0
fi

# every TENSORFOLD_* and BOLT_* variable goes into the container (libbolt's own switches, e.g. BOLT_CAL, BOLT_SPIN,
# BOLT_MOE_BD, BOLT_STATS; the ones start.sh sets itself come later on the command line and win)
ENV_ARGS=(-e HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}")
while IFS='=' read -r name _; do
  [[ "$name" == TENSORFOLD_API_KEY ]] && continue
  ENV_ARGS+=(-e "$name=${!name}")
done < <(env | grep -E '^(TENSORFOLD|BOLT)_[A-Z0-9_]+=' || true)
KEY_ARGS=(); [[ -z "${TENSORFOLD_API_KEY:-}" ]] || { export TENSORFOLD_API_KEY; KEY_ARGS=(-e TENSORFOLD_API_KEY); }

step 3 "Launch: container $CONTAINER_NAME"
log "tensorfold-native serve $MODEL_ARG --name $SERVED_NAME --host $HOST --port $PORT ${SERVE_ARGS[*]} ${USER_ARGS[*]}"
here_cmd=(docker run -d --name "$CONTAINER_NAME"
          --gpus all --ipc=host --network host
          --ulimit memlock=-1 --ulimit stack=67108864
          "${ENV_ARGS[@]}" "${KEY_ARGS[@]}" "${ENV_DRAFT[@]}" "${ENV_SEQS[@]}"
          -v "$HF_CACHE":/root/.cache/huggingface:ro
          -v "$KERNEL_CACHE/$KCACHE":/cache
          "$IMAGE"
          tensorfold-native serve "$MODEL_ARG" --name "$SERVED_NAME" --host "$HOST" --port "$PORT"
          "${SERVE_ARGS[@]}" "${USER_ARGS[@]}")
if (( DRY )); then
  printf '[dry-run]\n  %s\n' "$(printf '%q ' "${here_cmd[@]}" | sed 's/ $//')"
  exit 0
fi
"${here_cmd[@]}" >/dev/null

if [[ "${FOREGROUND:-0}" == 1 ]]; then
  trap './stop.sh; exit 130' INT TERM
  docker logs -f "$CONTAINER_NAME" || true
  exit "$(docker inspect -f '{{.State.ExitCode}}' "$CONTAINER_NAME" 2>/dev/null || echo 1)"
fi

step 4 "Loading: ~${CKPT_GIB} GiB of weights"
NOISE='^\s*$|^=+$|^== PyTorch ==|^NVIDIA Release|Copyright|All rights reserved|PyTorch Version|Various files include|NOTE: CUDA Forward|Using CUDA|cuda-compatibility|Container image|^\[tensorfold\] [^ ]+ "GET /(health|v1/models|models|metrics|stats|slots)[ ?]'
docker logs -f "$CONTAINER_NAME" > >(grep --line-buffered -v -E "$NOISE" | sed -u "s/^/  ${D}│${R} /") 2>&1 &
LOGS_PID=$!
trap 'kill $LOGS_PID 2>/dev/null || true' EXIT
fail() {
  kill $LOGS_PID 2>/dev/null || true
  sleep 0.5
  printf '\n%s── last server log lines ──%s\n' "$D" "$R"
  docker logs --tail 40 "$CONTAINER_NAME" 2>&1 | sed 's/^/  │ /'
  die "$1"
}
start=$SECONDS
next_beat=15
until curl -sf --max-time 5 "$URL/health" >/dev/null 2>&1; do
  if ! running; then
    fail "the server exited (code $(docker inspect -f '{{.State.ExitCode}}' "$CONTAINER_NAME")) before it was ready"
  fi
  (( SECONDS - start < WAIT_TIMEOUT )) || fail "not ready after ${WAIT_TIMEOUT}s; it is still running: docker logs -f $CONTAINER_NAME"
  a=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
  if (( a < MEM_FLOOR_GIB )); then
    docker stop -t 5 "$CONTAINER_NAME" >/dev/null 2>&1 || true
    fail "only $a GiB memory left while loading (MEM_FLOOR_GIB $MEM_FLOOR_GIB): stopped"
  fi
  if (( SECONDS - start >= next_beat )); then
    printf '  %s⋯ %ss elapsed, %s GiB free%s\n' "$D" "$((SECONDS - start))" "$a" "$R"
    next_beat=$((next_beat + 15))
  fi
  sleep 3
done
kill $LOGS_PID 2>/dev/null || true
sleep 0.3
LOADED_S=$((SECONDS - start))
log "Server answered after ${LOADED_S}s"

step 5 "Smoke test: one chat completion"
SERVED=$(served_name || echo "$SERVED_NAME")
body="{\"model\": \"$SERVED\", \"max_tokens\": 32, \"temperature\": 0, \"chat_template_kwargs\": {\"enable_thinking\": false}, \"messages\": [{\"role\": \"user\", \"content\": \"Reply with OK.\"}]}"
if smoke=$(curl -s --max-time 180 "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d "$body" |
           python3 -c 'import json,sys; r = json.load(sys.stdin); c = r["choices"][0]["message"].get("content") or ""; assert c.strip(); print(repr(c.strip()[:40]) + ",", r["usage"]["completion_tokens"], "tokens")' 2>/dev/null); then
  log "OK: $smoke"
else
  fail "the smoke test request failed; the server is still running"
fi
IP=$(hostname -I 2>/dev/null | awk '{print $1}')
[[ "$HOST" == 0.0.0.0 || "$HOST" == "::" ]] || IP="$HOST"
printf '\n%s  ✔ %s is now LIVE! on port %s%s\n\n' "$G" "$SERVED" "$PORT" "$R"
cat <<EOF
    API      http://${IP:-<spark-address>}:$PORT/v1   (model: $SERVED)
    Loaded   in ${LOADED_S}s · ${avail_gb} GiB was free
    Model    $MODEL_ID @ ${MODEL_REVISION:0:8} ($QUANT_LABEL)
    Window   $(arg_value --context) tokens · $(arg_value --parallel) at once · $KV_DTYPE KV · drafts $( [[ "$DRAFTS" == 1 ]] && echo on || echo off)
    Logs     docker logs -f $CONTAINER_NAME
    Restart  SERVE=1 ./start.sh restart
    Stop     ./stop.sh

EOF
