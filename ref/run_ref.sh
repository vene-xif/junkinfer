#!/usr/bin/env bash
# JunkInfer one-time reference run on VENE (RTX 3090, 128 GB RAM).
#
# Produces, once, everything the AUX-1 engine work is judged against:
#   - BF16 GGUF of Qwen3.6-35B-A3B (from the official safetensors)
#   - fixed tokenized eval corpus (wiki, code, model-generated chat incl. tool calls, 32k/128k/262k long docs)
#   - BF16 reference top-512 log-probs for every scored position (JREF) + per-token MoE routing traces (JRTR)
#   - llama-perplexity KLD base file (full logits, wiki c=2048) for cross-checks / ik_llama
#   - imatrix; GGUF quants (mainline + ik trellis KT); EXL3 3.5/4.0/3.0 bpw
#   - KLD of every quant and of q8_0/q4_0 KV against the BF16 reference -> the quality bars
#
# usage: run_ref.sh --dry-run          verify every assumption, change nothing (fast)
#        run_ref.sh                    run all stages (skips stages already marked done)
#        run_ref.sh --only <stage>     run one stage
#        run_ref.sh --list             list stages and their state
set -uo pipefail

P=/mnt/chonker/projects/junkinfer
SRC=$P/src
D=$P/data
LOGS=$D/logs
STATE=$D/state
RAW=$D/raw
HF=$D/hf/Qwen3.6-35B-A3B
GG=$D/gguf
BF16=$GG/Qwen3.6-35B-A3B-BF16.gguf
CORP=$D/corpus
REFS=$D/refs
EXL=$D/exl3
EVAL=$D/eval
KLD=$EVAL/kld.jsonl
VENV=$P/.venv-ref
TOOLS=$P/ref/tools
PROMPTS=$P/ref/prompts/chat_prompts.json
LCPP=$SRC/llama.cpp
LB=$LCPP/build/bin
IK=$SRC/ik_llama.cpp
IKB=$IK/build/bin
EXL3SRC=$SRC/exllamav3
ANCHORS=/mnt/chonker/ai/models
CHAT_MODEL=$ANCHORS/Qwen3.6-35B-A3B-UD-IQ4_NL.gguf

LCPP_PIN=b76b13379
EXL3_PIN=058b863
IK_PIN=8313388a
HF_REPO=Qwen/Qwen3.6-35B-A3B
HF_SHARDS=26
WIKI_URL=https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip
PROSE_URL=https://www.gutenberg.org/cache/epub/2600/pg2600.txt
TORCH_INDEX=https://download.pytorch.org/whl/cu130
SERVER_PORT=18950
NT=12              # VENE logical CPUs
NT_Q=8             # quantize threads while the GPU lane also needs CPU

export PATH=/usr/local/cuda/bin:$PATH
export CUDACXX=/usr/local/cuda/bin/nvcc
export HF_HUB_ENABLE_HF_TRANSFER=0
export EXL3_SRC=$EXL3SRC

DRY=0; ONLY=""; LIST=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --only) ONLY="$2"; shift ;;
    --list) LIST=1 ;;
    *) echo "unknown arg $1"; exit 2 ;;
  esac
  shift
done

ts() { date '+%F %T'; }
say() { echo "[$(ts)] $*"; }
die() { say "FATAL: $*"; exit 1; }
done_mark() { [ -f "$STATE/$1.done" ]; }
mark() { mkdir -p "$STATE"; date '+%F %T' > "$STATE/$1.done"; }
failmark() { mkdir -p "$STATE"; echo "$(ts) $2" > "$STATE/$1.failed"; }

# run_stage NAME essential(1|0) function
run_stage() {
  local name=$1 essential=$2 fn=$3
  if [ -n "$ONLY" ] && [ "$ONLY" != "$name" ]; then return 0; fi
  if done_mark "$name"; then say "skip $name (done $(cat $STATE/$name.done))"; return 0; fi
  local rf="$STATE/$name.running" opid
  opid=$(cat "$rf" 2>/dev/null || true)
  if [ -n "$opid" ] && kill -0 "$opid" 2>/dev/null; then
    # an earlier instance of this script is still running this stage: wait for it, never run it twice
    say "=== $name already running in pid $opid, waiting"
    while kill -0 "$opid" 2>/dev/null; do sleep 10; done
    sleep 2
    if done_mark "$name"; then say "<<< $name OK (other instance)"; return 0; fi
    say "<<< $name did not finish in the other instance"
    [ "$essential" = 1 ] && die "essential stage $name failed (other instance)"
    return 0
  fi
  say ">>> $name"
  rm -f "$STATE/$name.failed"
  local t0=$SECONDS rc
  echo $BASHPID > "$rf"
  # NOT inside an `if`: errexit is ignored in a subshell that is an if-condition
  ( set -e; $fn ) > "$LOGS/$name.log" 2>&1
  rc=$?
  if [ $rc = 0 ]; then
    mark "$name"; say "<<< $name OK ($(( (SECONDS - t0) / 60 )) min)"
  else
    failmark "$name" "exit $rc"
    say "<<< $name FAILED ($(( (SECONDS - t0) / 60 )) min) - see $LOGS/$name.log"; tail -n 15 "$LOGS/$name.log" | sed 's/^/    | /'
  fi
  rm -f "$rf"
  [ $rc != 0 ] && [ "$essential" = 1 ] && die "essential stage $name failed"
  return 0
}

wait_for() {  # wait_for STAGE [timeout_s]
  local s=$1 to=${2:-36000} t=0
  while ! done_mark "$s"; do
    [ -f "$STATE/$s.failed" ] && { echo "dependency $s failed"; return 1; }
    sleep 20; t=$((t + 20)); [ $t -lt $to ] || { echo "timeout waiting for $s"; return 1; }
  done
  return 0   # a while loop returns its last body command's status: never let that leak under errexit
}

py() { "$VENV/bin/python" "$@"; }

gpu_free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }

# ------------------------------------------------------------------ preflight / dry run
preflight() {
  local bad=0
  chk() { if ( eval "$2" ) >/dev/null 2>&1; then echo "  ok   $1"; else echo "  FAIL $1"; bad=1; fi; }
  say "preflight"
  chk "nvidia-smi"              "nvidia-smi -L"
  chk "GPU is an RTX 3090"      "nvidia-smi -L | grep -q 3090"
  chk "GPU idle (>= 23000 MiB free)" "[ \$(gpu_free_mib) -ge 23000 ]"
  chk "no GPU compute apps"     "[ -z \"\$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)\" ]"
  chk "nvcc 13.x"               "nvcc --version | grep -q 'release 13'"
  chk "cmake/git/gcc/g++/curl/python3/unzip" "command -v cmake && command -v git && command -v gcc && command -v g++ && command -v curl && command -v python3 && command -v unzip"
  chk "python3 >= 3.10 with venv" "python3 -c 'import sys, venv; assert sys.version_info >= (3,10)'"
  chk "system huggingface_hub + hf_xet" "python3 -c 'import huggingface_hub, hf_xet'"
  chk "RAM >= 100 GiB"          "[ \$(awk '/MemTotal/{print int(\$2/1048576)}' /proc/meminfo) -ge 100 ]"
  chk "disk free >= 450 GB at $P" "[ \$(df -BG --output=avail $P | tail -1 | tr -dc 0-9) -ge 450 ]"
  chk "llama.cpp pinned $LCPP_PIN" "git -C $LCPP rev-parse HEAD | grep -q ^$LCPP_PIN"
  chk "exllamav3 pinned $EXL3_PIN" "git -C $EXL3SRC rev-parse HEAD | grep -q ^$EXL3_PIN"
  chk "ik_llama pinned $IK_PIN" "git -C $IK rev-parse HEAD | grep -q ^$IK_PIN"
  chk "convert: Qwen3_5MoeForConditionalGeneration" "grep -q 'Qwen3_5MoeForConditionalGeneration' $LCPP/conversion/qwen.py"
  chk "graph names ffn_moe_topk / ffn_moe_weights_norm" "grep -q '\"ffn_moe_topk\"' $LCPP/src/llama-graph.cpp && grep -q '\"ffn_moe_weights_norm\"' $LCPP/src/llama-graph.cpp"
  chk "qwen35moe uses norm_w=true" "grep -A8 'build_moe_ffn(cur,' $LCPP/src/models/qwen35moe.cpp | grep -q 'LLM_FFN_SILU, true'"
  chk "imatrix --output-format"  "grep -q '\"--output-format\"}, \"{gguf,dat}\"' $LCPP/common/arg.cpp"
  chk "quantize knows IQ4_XS IQ3_S IQ3_XXS Q4_K_M Q6_K Q8_0" "for q in IQ4_XS IQ3_S IQ3_XXS Q4_K_M Q6_K Q8_0; do grep -q \"\\\"\$q\\\"\" $LCPP/tools/quantize/quantize.cpp || exit 1; done"
  chk "ik: qwen35moe + IQ4_KT/IQ3_KT" "grep -q LLM_ARCH_QWEN35MOE $IK/src/llama-arch.cpp && grep -q '\"IQ4_KT\"' $IK/examples/quantize/quantize.cpp && grep -q '\"IQ3_KT\"' $IK/examples/quantize/quantize.cpp"
  chk "exllamav3: qwen3_5 arch + convert.py" "grep -q Qwen3_5Moe $EXL3SRC/exllamav3/architecture/architectures.py && [ -f $EXL3SRC/convert.py ]"
  chk "server: return_tokens"    "grep -q return_tokens $LCPP/tools/server/README.md"
  chk "junk-ref.cpp syntax"      "g++ -std=c++17 -fsyntax-only -I$LCPP/include -I$LCPP/common -I$LCPP/ggml/include -I$LCPP/vendor $TOOLS/junk-ref/junk-ref.cpp"
  chk "python tools compile"     "for f in $TOOLS/*.py; do python3 -c \"import ast,sys; ast.parse(open('\$f').read())\" || exit 1; done"
  chk "prompts json valid"       "python3 -c \"import json; d=json.load(open('$PROMPTS')); assert len(d)>=24 and all(s['split'] in ('eval','calib') for s in d)\""
  chk "this script parses"       "bash -n $0"
  chk "port $SERVER_PORT free"   "! (exec 3<>/dev/tcp/127.0.0.1/$SERVER_PORT) 2>/dev/null"
  chk "HF repo: $HF_SHARDS shards, ~72 GB" "python3 -c \"
from huggingface_hub import HfApi
i=HfApi().model_info('$HF_REPO', files_metadata=True)
sh=[s for s in i.siblings if s.rfilename.endswith('.safetensors')]
tot=sum(s.size for s in i.siblings)
assert len(sh)==$HF_SHARDS and 65e9<tot<80e9, (len(sh), tot)\""
  chk "wikitext zip reachable"   "curl -sfIL --max-time 20 $WIKI_URL"
  chk "gutenberg prose reachable" "curl -sfIL --max-time 20 $PROSE_URL"
  chk "pytorch cu130 index has cp312 x86_64 wheels" "[ \$(curl -sfL --max-time 20 $TORCH_INDEX/torch/ | grep -c 'cu130-cp312-cp312-manylinux_2_28_x86_64') -gt 0 ]"
  for a in Qwen3.6-35B-A3B-UD-IQ4_NL.gguf Qwen_Qwen3.6-35B-A3B-IQ4_NL.gguf Qwen3.6-35B-A3B-APEX-I-Balanced.gguf Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf; do
    chk "anchor $a" "[ -s $ANCHORS/$a ]"
  done
  echo "  state: $(ls $STATE 2>/dev/null | tr '\n' ' ')"
  [ $bad = 0 ] || return 1
}

# ------------------------------------------------------------------ stages
st_venv() {
  [ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
  py -m pip install -q --upgrade pip
  py -m pip install -q torch --index-url $TORCH_INDEX
  py -m pip install -q numpy safetensors "tokenizers>=0.21.1" transformers sentencepiece protobuf pyyaml rich ninja \
      pydantic marisa_trie "llguidance>=1.7.0" pillow typing_extensions huggingface_hub jinja2
  py -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
}

st_download_small() {
  mkdir -p "$HF" "$RAW"
  python3 - <<EOF
from huggingface_hub import snapshot_download
snapshot_download("$HF_REPO", local_dir="$HF", allow_patterns=["*.json", "*.txt", "*.jinja", "LICENSE", "README.md"])
EOF
  [ -s "$HF/tokenizer.json" ] && [ -s "$HF/config.json" ]
  [ -s "$RAW/wikitext-2-raw/wiki.test.raw" ] || { curl -sfL -o "$RAW/wikitext-2-raw-v1.zip" $WIKI_URL && unzip -o -q "$RAW/wikitext-2-raw-v1.zip" -d "$RAW"; }
  [ -s "$RAW/pg2600.txt" ] || curl -sfL -o "$RAW/pg2600.txt" $PROSE_URL
  ls -la "$RAW" "$RAW/wikitext-2-raw"
  [ -s "$RAW/wikitext-2-raw/wiki.test.raw" ] && [ -s "$RAW/wikitext-2-raw/wiki.train.raw" ] && [ -s "$RAW/pg2600.txt" ]
}

st_download_model() {
  python3 - <<EOF
from huggingface_hub import snapshot_download
snapshot_download("$HF_REPO", local_dir="$HF", max_workers=8)
EOF
  local n; n=$(ls "$HF"/model-*-of-000$HF_SHARDS.safetensors | wc -l)
  [ "$n" = "$HF_SHARDS" ] || { echo "only $n shards"; return 1; }
  du -sh "$HF"
}

st_build_lcpp() {
  mkdir -p "$LCPP/examples/junk-ref"
  cp "$TOOLS/junk-ref/junk-ref.cpp" "$TOOLS/junk-ref/CMakeLists.txt" "$LCPP/examples/junk-ref/"
  grep -q 'add_subdirectory(junk-ref)' "$LCPP/examples/CMakeLists.txt" || \
    sed -i 's/^    add_subdirectory(eval-callback)$/    add_subdirectory(eval-callback)\n    add_subdirectory(junk-ref)/' "$LCPP/examples/CMakeLists.txt"
  grep -q 'add_subdirectory(junk-ref)' "$LCPP/examples/CMakeLists.txt"
  cmake -S "$LCPP" -B "$LCPP/build" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF
  cmake --build "$LCPP/build" -j $NT --target llama-junk-ref llama-server llama-imatrix llama-quantize llama-perplexity
  for b in llama-junk-ref llama-server llama-imatrix llama-quantize llama-perplexity; do [ -x "$LB/$b" ] || { echo "missing $b"; return 1; }; done
}

st_build_exl3() {
  wait_for venv
  cd "$EXL3SRC"
  TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=$NT py -m pip install --no-build-isolation -v . 2>&1 | tail -n 30
  cd /
  py -c "import exllamav3, exllamav3.ext; print('exllamav3', exllamav3.__file__)"
}

st_build_ik() {
  cmake -S "$IK" -B "$IK/build" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
  cmake --build "$IK/build" -j $NT_Q --target llama-quantize llama-perplexity
  [ -x "$IKB/llama-quantize" ] && [ -x "$IKB/llama-perplexity" ]
}

st_convert() {
  wait_for venv; wait_for download_model
  mkdir -p "$GG"
  py "$LCPP/convert_hf_to_gguf.py" "$HF" --outtype bf16 --outfile "$BF16.tmp"
  mv "$BF16.tmp" "$BF16"
  ls -la "$BF16"
}

st_corpus() {
  wait_for venv; wait_for download_small
  mkdir -p "$CORP"
  py "$TOOLS/prep_corpus.py" --tokenizer "$HF/tokenizer.json" --wiki-dir "$RAW/wikitext-2-raw" \
     --code-root "$LCPP" --prose "$RAW/pg2600.txt" --out "$CORP"
  py "$TOOLS/gen_chat.py" --hf-dir "$HF" --prompts "$PROMPTS" --out "$CORP" --check-only
  ls -la "$CORP"
}

# junk-ref smoke on an existing GGUF while the model downloads: dump 2 rows with routing, compare to itself
st_smoke() {
  wait_for build_lcpp; wait_for corpus
  local m=$ANCHORS/Qwen3.6-35B-A3B-UD-IQ4_NL.gguf s=$D/smoke
  mkdir -p "$s"
  "$LB/llama-junk-ref" -m "$m" -ngl 99 -fa on --fit off -c 4096 -b 2048 -ub 2048 -t $NT \
     --jr-mode dump --jr-tokens "$CORP/S1_wiki.jtok" --jr-rows 0:2 --jr-out "$s/s1.jref" --jr-summary "$s/dump.json"
  "$LB/llama-junk-ref" -m "$m" -ngl 99 -fa on --fit off -c 4096 -b 2048 -ub 2048 -t $NT \
     --jr-mode compare --jr-tokens "$CORP/S1_wiki.jtok" --jr-rows 0:2 --jr-ref "$s/s1.jref" --jr-summary "$s/cmp.json" --jr-label smoke-self
  # routing pass (graph split at the MoE routing nodes -> unfused top-k path): trace + its KL floor
  "$LB/llama-junk-ref" -m "$m" -ngl 99 -fa on --fit off -c 4096 -b 2048 -ub 2048 -t $NT \
     --jr-mode compare --jr-tokens "$CORP/S1_wiki.jtok" --jr-rows 0:2 --jr-ref "$s/s1.jref" --jr-route "$s/s1.jrtr" --jr-summary "$s/route.json" --jr-label smoke-route
  cat "$s/dump.json" "$s/cmp.json" "$s/route.json"
  py - "$s" <<'EOF'
import json, os, struct, sys, numpy as np
s = sys.argv[1]
c = json.load(open(f"{s}/cmp.json"))
assert c["n"] == 2 * 1023, c
assert c["kl_mean"] < 1e-5 and c["top1_agree"] > 0.999, c       # same model, same path: deterministic
rt = json.load(open(f"{s}/route.json"))
assert rt["kl_mean"] < 0.03 and rt["top1_agree"] > 0.9, rt      # unfused routing path: small floor only
assert 1.5 < c["ppl_model"] < 40, c
f = open(f"{s}/s1.jrtr", "rb"); assert f.read(4) == b"JRTR"
ver, nl, k = struct.unpack("<III", f.read(12)); assert k == 8, k
r, L = struct.unpack("<II", f.read(8)); assert L == 2048
ids = np.frombuffer(f.read(L * nl * k), dtype=np.uint8).reshape(L, nl, k)
w = np.frombuffer(f.read(L * nl * k * 2), dtype=np.float16).reshape(L, nl, k).astype(np.float32)
seen = [l for l in range(nl) if (ids[:, l] != 255).all()]
assert len(seen) == 40, (nl, len(seen))
assert (ids[:, seen] < 256).all() and np.allclose(w[:, seen].sum(-1), 1.0, atol=2e-2), w[:, seen].sum(-1)[:3]
print("smoke OK: n_layer", nl, "moe layers", len(seen), "kl", c["kl_mean"])
EOF
}

JR_BF16="-m $BF16 -ngl 99 --cpu-moe -fa on --fit off -t $NT"

st_ref_short() {
  wait_for convert; wait_for smoke
  mkdir -p "$REFS"
  for s in S1_wiki S2_code; do
    "$LB/llama-junk-ref" $JR_BF16 -c 4096 -b 2048 -ub 2048 --jr-mode dump --jr-tokens "$CORP/$s.jtok" \
       --jr-out "$REFS/$s.jref" --jr-summary "$REFS/$s.dump.json"
  done
  cat "$REFS"/*.dump.json
}

st_validate_ref() {
  wait_for ref_short
  # BF16 vs its own reference: must be ~0 (catches reference/tool bugs before anything depends on them)
  "$LB/llama-junk-ref" $JR_BF16 -c 4096 -b 2048 -ub 2048 --jr-mode compare --jr-tokens "$CORP/S1_wiki.jtok" \
     --jr-rows 0:4 --jr-ref "$REFS/S1_wiki.jref" --jr-summary "$REFS/selfcheck.json" --jr-label bf16-self
  py -c "
import json; c=json.load(open('$REFS/selfcheck.json')); print(c)
assert c['kl_mean'] < 5e-4 and c['top1_agree'] > 0.99, 'BF16 self-check failed'"
}

st_chat() {
  wait_for convert; wait_for corpus
  # the TEXT of S3 is generated by a fast, fully GPU-resident quant (BF16 with CPU experts ran at 1.1 tok/s
  # per slot); the reference LOGITS over those tokens still come from BF16 (dump below)
  "$LB/llama-server" -m "$CHAT_MODEL" -ngl 99 -fa on --fit off -c 32768 -np 4 -t $NT --port $SERVER_PORT \
     > "$LOGS/chat-server.log" 2>&1 &
  local spid=$!
  # EXIT (not RETURN): under errexit the stage subshell exits without returning from this function
  trap "kill $spid 2>/dev/null; wait $spid 2>/dev/null || true" EXIT
  local t=0
  until curl -sf "http://127.0.0.1:$SERVER_PORT/health" | grep -q ok; do
    sleep 5; t=$((t + 5)); kill -0 $spid || { echo "server died"; tail -50 "$LOGS/chat-server.log"; return 1; }
    [ $t -lt 900 ] || { echo "server not ready"; return 1; }
  done
  py "$TOOLS/gen_chat.py" --hf-dir "$HF" --prompts "$PROMPTS" --out "$CORP" --server "http://127.0.0.1:$SERVER_PORT" --parallel 4
  kill $spid; wait $spid 2>/dev/null || true
  trap - EXIT
  "$LB/llama-junk-ref" $JR_BF16 -c 8192 -b 2048 -ub 2048 --jr-mode dump --jr-tokens "$CORP/S3_chat.jtok" \
     --jr-out "$REFS/S3_chat.jref" --jr-summary "$REFS/S3_chat.dump.json"
  cat "$REFS/S3_chat.dump.json"
}

st_ref_long() {
  wait_for validate_ref
  for s in L_code L_prose; do
    "$LB/llama-junk-ref" $JR_BF16 -c 262144 -b 4096 -ub 4096 --jr-mode dump --jr-tokens "$CORP/$s.jtok" \
       --jr-out "$REFS/$s.jref" --jr-summary "$REFS/$s.dump.json"
  done
  cat "$REFS"/L_*.dump.json
}

# routing traces from a separate pass: asking for the routing tensors splits the graph and runs the
# unfused top-k path, which moves the logits slightly (measured 2026-10-15 on IQ4_NL: KL 0.005); the
# reference itself comes from the standard path. The KL of this pass vs the reference = that floor for BF16.
st_routes() {
  wait_for ref_short; wait_for chat
  mkdir -p "$EVAL"
  local s
  for s in S1_wiki S2_code S3_chat L_code L_prose; do
    [ -s "$REFS/$s.jref" ] || { echo "no reference for $s, skipped"; continue; }
    local c=4096 b=2048
    case $s in S3_chat) c=8192 ;; L_*) c=262144; b=4096 ;; esac
    "$LB/llama-junk-ref" $JR_BF16 -c $c -b $b -ub $b --jr-mode compare --jr-tokens "$CORP/$s.jtok" --jr-ref "$REFS/$s.jref" \
       --jr-route "$REFS/$s.jrtr" --jr-summary "$EVAL/bf16-routepass.$s.json" --jr-label bf16-routepass
    cat "$EVAL/bf16-routepass.$s.json" >> "$KLD"
  done
}

st_pplbase() {
  wait_for convert
  "$LB/llama-perplexity" $JR_BF16 -f "$RAW/wikitext-2-raw/wiki.test.raw" -c 2048 -b 2048 -ub 2048 --chunks 24 \
     --kl-divergence-base "$REFS/ppl_wiki_c2048.kldbase"
  ls -la "$REFS/ppl_wiki_c2048.kldbase"
}

st_imatrix() {
  wait_for chat; wait_for corpus
  cat "$CORP/calib_base.txt" "$CORP/calib_chat.txt" > "$CORP/calib_all.txt"
  "$LB/llama-imatrix" $JR_BF16 -f "$CORP/calib_all.txt" -c 2048 -b 2048 -ub 2048 --chunks 100 \
     --output-format dat -o "$GG/imatrix.dat"
  ls -la "$GG/imatrix.dat"
}

# ---- lane B (CPU): quantize; runs in background next to the GPU lane
quant_mainline() {
  local q
  for q in Q8_0 IQ4_XS Q4_K_M IQ3_S IQ3_XXS Q6_K; do
    local out="$GG/Qwen3.6-35B-A3B-$q.gguf"
    [ -s "$out" ] && continue
    if [ $q = Q8_0 ]; then
      "$LB/llama-quantize" "$BF16" "$out.tmp" $q $NT_Q
    else
      # the MTP block (blk.40) never runs during imatrix collection: keep it at q8_0
      "$LB/llama-quantize" --imatrix "$GG/imatrix.dat" --tensor-type 'blk\.40\.=q8_0' "$BF16" "$out.tmp" $q $NT_Q
    fi
    mv "$out.tmp" "$out"
    touch "$STATE/q_$q.ready"
  done
}
quant_ik() {
  local q
  for q in IQ4_KT IQ3_KT; do
    local out="$GG/Qwen3.6-35B-A3B-$q.gguf"
    [ -s "$out" ] && continue
    "$IKB/llama-quantize" --imatrix "$GG/imatrix.dat" "$BF16" "$out.tmp" $q $NT_Q
    mv "$out.tmp" "$out"
    touch "$STATE/q_$q.ready"
  done
}
st_quant_mainline() { wait_for imatrix; quant_mainline; }
st_quant_ik() { wait_for imatrix; wait_for build_ik; quant_ik; }

# ---- EXL3
exl3_convert() {  # bits
  local b=$1 out=$EXL/$1bpw work=$EXL/work-$1
  mkdir -p "$EXL"
  if [ -f "$out/config.json" ] && ls "$out"/*.safetensors >/dev/null 2>&1 && [ ! -d "$work" ]; then return 0; fi
  if [ -f "$work/args.json" ]; then
    py "$EXL3SRC/convert.py" -w "$work" -r
  else
    py "$EXL3SRC/convert.py" -i "$HF" -o "$out" -w "$work" -b "$b"
  fi
  ls "$out"/*.safetensors >/dev/null
  rm -rf "$work"
}
exl3_eval() {  # bits
  py "$TOOLS/exl3_kld.py" -m "$EXL/$1bpw" --label "exl3-$1bpw" --summary "$KLD" \
     --sets "$CORP/S1_wiki.jtok:$REFS/S1_wiki.jref" "$CORP/S2_code.jtok:$REFS/S2_code.jref" "$CORP/S3_chat.jtok:$REFS/S3_chat.jref"
}
st_exl3_35() { wait_for build_exl3; wait_for chat; wait_for download_model; exl3_convert 3.5; exl3_eval 3.5; }
st_exl3_40() { wait_for build_exl3; exl3_convert 4.0; exl3_eval 4.0; }
st_exl3_30() { wait_for build_exl3; exl3_convert 3.0; exl3_eval 3.0; }

# ---- GGUF / KV evaluations against the reference
eval_gguf() {  # label path [extra junk-ref flags...]
  local label=$1 m=$2; shift 2
  local placement="-ngl 99"
  local gib=$(( $(stat -c %s "$m") / 1073741824 ))
  [ $gib -ge 20 ] && placement="-ngl 99 --cpu-moe"
  for s in S1_wiki S2_code S3_chat; do
    local sj="$EVAL/$label.$s.json"
    [ -s "$sj" ] && continue
    "$LB/llama-junk-ref" -m "$m" $placement -fa on --fit off -t $NT -c 8192 -b 2048 -ub 2048 "$@" \
       --jr-mode compare --jr-tokens "$CORP/$s.jtok" --jr-ref "$REFS/$s.jref" --jr-summary "$sj" --jr-label "$label" \
       || { echo "eval failed: $label $s"; rm -f "$sj"; continue; }
    cat "$sj" >> "$KLD"
  done
}
eval_ready_ggufs() {
  mkdir -p "$EVAL"
  local q
  for q in Q8_0 IQ4_XS Q4_K_M IQ3_S IQ3_XXS Q6_K; do
    [ -f "$STATE/q_$q.ready" ] && eval_gguf "gguf-$q" "$GG/Qwen3.6-35B-A3B-$q.gguf"
  done
  return 0
}
st_eval_anchors() {
  wait_for chat
  mkdir -p "$EVAL"
  eval_gguf anchor-UD-IQ4_NL      "$ANCHORS/Qwen3.6-35B-A3B-UD-IQ4_NL.gguf"
  eval_gguf anchor-bartowski-IQ4_NL "$ANCHORS/Qwen_Qwen3.6-35B-A3B-IQ4_NL.gguf"
  eval_gguf anchor-APEX-I-Balanced "$ANCHORS/Qwen3.6-35B-A3B-APEX-I-Balanced.gguf"
  eval_gguf anchor-UD-Q6_K_XL      "$ANCHORS/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf"
  eval_ready_ggufs
}
st_eval_kv() {  # the KV-cache quality bar on this model: BF16 weights, quantized KV, long + chat sets
  wait_for ref_long; wait_for chat
  mkdir -p "$EVAL"
  local kv s
  for kv in q8_0 q4_0; do
    for s in L_code L_prose; do
      local sj="$EVAL/kv-$kv.$s.json"; [ -s "$sj" ] && continue
      "$LB/llama-junk-ref" $JR_BF16 -c 262144 -b 4096 -ub 4096 -ctk $kv -ctv $kv --jr-mode compare \
         --jr-tokens "$CORP/$s.jtok" --jr-ref "$REFS/$s.jref" --jr-summary "$sj" --jr-label "kv-$kv"
      cat "$sj" >> "$KLD"
    done
    local sj="$EVAL/kv-$kv.S3_chat.json"
    if [ ! -s "$sj" ]; then
      "$LB/llama-junk-ref" $JR_BF16 -c 8192 -b 2048 -ub 2048 -ctk $kv -ctv $kv --jr-mode compare \
         --jr-tokens "$CORP/S3_chat.jtok" --jr-ref "$REFS/S3_chat.jref" --jr-summary "$sj" --jr-label "kv-$kv"
      cat "$sj" >> "$KLD"
    fi
  done
}
st_eval_quants() {
  wait_for quant_mainline 30000
  eval_ready_ggufs
  # cross-check junk-ref against llama-perplexity's own full-logit KLD on one quant
  "$LB/llama-perplexity" -m "$GG/Qwen3.6-35B-A3B-IQ4_XS.gguf" -ngl 99 -fa on --fit off -t $NT -c 2048 -b 2048 -ub 2048 --chunks 24 \
     --kl-divergence-base "$REFS/ppl_wiki_c2048.kldbase" --kl-divergence > "$EVAL/ppl-crosscheck-IQ4_XS.txt" 2>&1 || true
  grep -E "Mean +KLD|Same top" "$EVAL/ppl-crosscheck-IQ4_XS.txt" || true
}
st_eval_ik() {
  wait_for quant_ik 30000
  local q
  for q in IQ4_KT IQ3_KT; do
    "$IKB/llama-perplexity" -m "$GG/Qwen3.6-35B-A3B-$q.gguf" -ngl 99 -fa -t $NT -c 2048 -b 2048 -ub 2048 --chunks 24 \
       --kl-divergence-base "$REFS/ppl_wiki_c2048.kldbase" --kl-divergence > "$EVAL/ik-$q.ppl.txt" 2>&1 || true
    grep -E "Mean +KLD|Same top" "$EVAL/ik-$q.ppl.txt" || true
  done
}

st_report() {
  py "$TOOLS/report.py" "$KLD" "$EVAL/REPORT.md" "$D"
  ( cd "$REFS" && sha256sum *.jref *.jrtr > SHA256SUMS ) || true
  ( cd "$CORP" && sha256sum *.jtok > SHA256SUMS ) || true
  du -sh "$D"/* | sort -h
  cat "$EVAL/REPORT.md"
}

# ------------------------------------------------------------------ main

STAGES="venv download_small download_model build_lcpp build_exl3 build_ik convert corpus smoke ref_short validate_ref chat \
ref_long routes pplbase imatrix quant_mainline quant_ik exl3_35 eval_anchors eval_kv eval_quants exl3_40 eval_ik exl3_30 report"

if [ $LIST = 1 ]; then
  for s in $STAGES; do
    st="pending"; done_mark $s && st="done $(cat $STATE/$s.done)"; [ -f $STATE/$s.failed ] && st="FAILED $(cat $STATE/$s.failed)"
    printf '  %-16s %s\n' $s "$st"
  done
  exit 0
fi

if [ $DRY = 1 ]; then
  preflight && { say "DRY RUN OK - stages: $STAGES"; exit 0; } || { say "DRY RUN FAILED"; exit 1; }
fi

[ -n "$ONLY" ] || { preflight || die "preflight failed"; }
mkdir -p "$LOGS" "$STATE"

if [ -n "$ONLY" ]; then
  run_stage "$ONLY" 1 "st_$ONLY"
  exit 0
fi

say "=== reference run start (pid $$) ==="
# lane 0: parallel setup (network + CPU builds)
run_stage download_small 1 st_download_small
( run_stage download_model 1 st_download_model ) &
( run_stage venv 1 st_venv; run_stage build_exl3 0 st_build_exl3 ) &
run_stage build_lcpp 1 st_build_lcpp
run_stage corpus 1 st_corpus
run_stage smoke 1 st_smoke
run_stage convert 1 st_convert
# lane B (CPU) starts itself once the imatrix exists
( run_stage quant_mainline 0 st_quant_mainline; ) &
( run_stage build_ik 0 st_build_ik; run_stage quant_ik 0 st_quant_ik ) &
# lane A (GPU), strictly sequential
run_stage ref_short 1 st_ref_short
run_stage validate_ref 1 st_validate_ref
run_stage chat 1 st_chat
run_stage imatrix 0 st_imatrix
run_stage ref_long 0 st_ref_long
run_stage routes 0 st_routes
run_stage pplbase 0 st_pplbase
run_stage exl3_35 0 st_exl3_35
run_stage eval_anchors 0 st_eval_anchors
run_stage eval_kv 0 st_eval_kv
run_stage eval_quants 0 st_eval_quants
run_stage exl3_40 0 st_exl3_40
run_stage eval_ik 0 st_eval_ik
run_stage exl3_30 0 st_exl3_30
wait
run_stage report 0 st_report
say "=== reference run end ==="
