#!/usr/bin/env bash
# =====================================================================================
#  EdgeStream-VLM — tải dữ liệu + chạy benchmark referring MOT (HOTA@L)
#
#  Cách dùng:
#    bash scripts/run_benchmarks.sh                      # tất cả: aerialmind + refer_kitti_v2
#    bash scripts/run_benchmarks.sh aerialmind           # chỉ AerialMind (in-domain + cross-domain)
#    bash scripts/run_benchmarks.sh --download-only
#    bash scripts/run_benchmarks.sh --bench-only aerialmind
#    bash scripts/run_benchmarks.sh --smoke aerialmind   # chạy thử nhanh (3 video, 3 expression/video)
#
#  Các bộ: aerialmind (chính, UAV, giao thức chính thức)   refer_kitti_v2 (tổng quát, camera xe)
#
#  Biến môi trường:
#    DATA_DIR      nơi chứa dữ liệu                      (mặc định: data)
#    RUN_DIR       nơi ghi kết quả                       (mặc định: runs/bench_<ngày giờ>)
#    RMOT_METHODS  các phương pháp  (mặc định: search,filter_kf,filter_ef,prior)
#                  thêm: search_single search_nolink filter_set ceiling
#    LAGS          độ trễ commit L (giây)                 (mặc định: "0,2,5,inf")
#    RMOT_VLM      openai | transformers | oracle          (mặc định: openai = server vLLM)
#    RMOT_VLM_MODEL                                        (mặc định: Qwen/Qwen3-VL-4B-Instruct)
#    RMOT_VLM_URL  endpoint vLLM                           (mặc định: http://localhost:8000/v1)
#    N_WORKERS     số request VLM song song                (mặc định: 4)
#    JOBS          số process chạy song song các stream    (mặc định: 1; >1 chỉ với openai/oracle)
#    MAX_VIDEOS    giới hạn số video mỗi split (debug)
#    MAX_EXPR      số expression mỗi video (debug)
#    AM_SPLITS     "in_domain cross_domain"
#    PYTHON        trình thông dịch Python               (mặc định: python3)
#    PARALLEL      1 = tải các bộ song song (chỉ với --download-only)        (mặc định: 1)
#    RCLONE_REMOTE remote Google Drive của rclone cho Refer-KITTI-V2         (mặc định: gdrive)
#
#  Kết quả: $RUN_DIR/<bộ>_<split>/{summary.md,results.json,<method>_L<lag>/...} và $RUN_DIR/summary.md
# =====================================================================================
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
DATA_DIR="${DATA_DIR:-data}"
RUN_DIR="${RUN_DIR:-runs/bench_$(date +%Y%m%d_%H%M)}"
MAX_VIDEOS="${MAX_VIDEOS:-}"
PARALLEL="${PARALLEL:-1}"
RMOT_METHODS="${RMOT_METHODS:-search,filter_kf,filter_ef,prior}"
LAGS="${LAGS:-0,2,5,inf}"
RMOT_VLM="${RMOT_VLM:-openai}"
RMOT_VLM_MODEL="${RMOT_VLM_MODEL:-Qwen/Qwen3-VL-4B-Instruct}"
RMOT_VLM_URL="${RMOT_VLM_URL:-http://localhost:8000/v1}"
N_WORKERS="${N_WORKERS:-4}"
JOBS="${JOBS:-1}"
MAX_EXPR="${MAX_EXPR:-}"
AM_SPLITS="${AM_SPLITS:-in_domain cross_domain}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"   # tải Hugging Face nhiều luồng

DO_DOWNLOAD=1
DO_BENCH=1
DATASETS=()
for arg in "$@"; do
  case "$arg" in
    --download-only) DO_BENCH=0 ;;
    --bench-only)    DO_DOWNLOAD=0 ;;
    --smoke)         MAX_VIDEOS="${MAX_VIDEOS:-3}"; MAX_EXPR="${MAX_EXPR:-3}" ;;
    -h|--help)       sed -n '2,32p' "$0"; exit 0 ;;
    all|rmot)        DATASETS+=(aerialmind refer_kitti_v2) ;;
    aerialmind|refer_kitti_v2) DATASETS+=("$arg") ;;
    *) echo "Không hiểu tham số: $arg (xem --help)"; exit 2 ;;
  esac
done
[ ${#DATASETS[@]} -eq 0 ] && DATASETS=(aerialmind refer_kitti_v2)

mkdir -p "$DATA_DIR" "$RUN_DIR"
LOG="$RUN_DIR/run.log"
exec > >(tee -a "$LOG") 2>&1

c_info() { printf '\033[1;34m[%s] %s\033[0m\n' "$(date +%H:%M:%S)" "$*"; }
c_ok()   { printf '\033[1;32m[%s] ✔ %s\033[0m\n' "$(date +%H:%M:%S)" "$*"; }
c_warn() { printf '\033[1;33m[%s] ! %s\033[0m\n' "$(date +%H:%M:%S)" "$*"; }
c_err()  { printf '\033[1;31m[%s] ✘ %s\033[0m\n' "$(date +%H:%M:%S)" "$*"; }
on_err() { c_err "Lỗi ở dòng $1: $2 (xem $LOG)"; }
trap 'on_err $LINENO "$BASH_COMMAND"' ERR
SKIP=42   # mã thoát khi thiếu dữ liệu (không phải lỗi)
skip() { c_warn "$*"; exit "$SKIP"; }

STATUS_LINES=""

# ------------------------------------------------------------------------------ setup
setup() {
  c_info "Kiểm tra môi trường Python ($("$PYTHON" -c 'import sys; print(sys.executable)'))"
  if ! "$PYTHON" -c "import edgestream, perception, cv2, numpy, yaml" 2>/dev/null; then
    c_info "Chưa cài dự án vào môi trường này -> pip install -e .  (lần đầu mất vài phút: torch, ultralytics)"
    "$PYTHON" -m pip install -e . || { c_err "pip install -e . thất bại (xem lỗi phía trên)"; exit 1; }
  fi
  "$PYTHON" -m pip install -q -U "huggingface_hub[hf_xet]>=0.32" gdown pillow >/dev/null
  "$PYTHON" -c "import ortools" 2>/dev/null || "$PYTHON" -m pip install -q "ortools>=9.10" >/dev/null || \
    c_warn "ortools chưa cài được: solver dùng greedy + Viterbi thay cho CP-SAT"
  if [ "$DO_BENCH" = 1 ] && [ ! -f weights/falcon_yolo_lqcv2_baseline_jde_mot.pth ]; then
    if [ -f "$HOME/Downloads/falcon_yolo_lqcv2_baseline_jde_mot.pth" ]; then
      c_info "Chép checkpoint từ ~/Downloads vào weights/"
      mkdir -p weights && cp "$HOME/Downloads/falcon_yolo_lqcv2_baseline_jde_mot.pth" weights/
    else
      c_err "Thiếu weights/falcon_yolo_lqcv2_baseline_jde_mot.pth (checkpoint MOT 960x544 của AERO-MOT)"; exit 1
    fi
  fi
  if [ "$DO_BENCH" = 1 ] && [ "$RMOT_VLM" = transformers ]; then
    "$PYTHON" -m pip install -q -U "transformers>=4.57" accelerate >/dev/null
    c_info "Tải model $RMOT_VLM_MODEL từ Hugging Face (lần đầu; 4B ≈ 9 GB, 8B ≈ 17 GB)"
    "$PYTHON" - "$RMOT_VLM_MODEL" <<'PY'
import sys, torch
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"], max_workers=16)
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"[vlm] {sys.argv[1]} ready, will run on {dev}")
if dev == "cpu":
    print("[vlm] WARNING: no GPU/MPS — each VLM call will take tens of seconds")
PY
  fi
}

# ------------------------------------------------------------------------------ helpers
fetch() { "$PYTHON" scripts/datasets/fetch.py "$@"; }
have() { [ -e "$1" ] && [ -n "$(ls -A "$1" 2>/dev/null)" ]; }

# ------------------------------------------------------------------------------ referring MOT
rmot_bench() {   # rmot_bench <name> <dataset> <root> <split> <seqmap> <config>
  local name="$1" ds="$2" root="$3" split="$4" seqmap="$5" cfg="$6"
  local extra=(--methods "$RMOT_METHODS" --lags "$LAGS" --vlm "$RMOT_VLM" --n-workers "$N_WORKERS" --jobs "$JOBS" --match-compute)
  [ -f "$seqmap" ] && extra+=(--seqmap "$seqmap")
  [ -n "$MAX_VIDEOS" ] && extra+=(--max-videos "$MAX_VIDEOS")
  [ -n "$MAX_EXPR" ] && extra+=(--max-expr "$MAX_EXPR")
  if [ "$RMOT_VLM" = openai ]; then
    extra+=(--vlm-url "$RMOT_VLM_URL" --vlm-model "$RMOT_VLM_MODEL")
    if ! curl -sf "$RMOT_VLM_URL/models" >/dev/null; then
      c_err "Không gọi được $RMOT_VLM_URL. Mở server trước, ví dụ:"
      echo "  vllm serve $RMOT_VLM_MODEL --enable-prefix-caching --max-model-len 8192 \\"
      echo "       --gpu-memory-utilization 0.90 --limit-mm-per-prompt '{\"image\": 4}' --port 8000"
      echo "  (hoặc RMOT_VLM=transformers để chạy model trực tiếp, chậm hơn vì tuần tự)"
      return 1
    fi
  elif [ "$RMOT_VLM" = transformers ]; then
    extra+=(--vlm-model "$RMOT_VLM_MODEL")
  fi
  c_info "RMOT $name: $ds/$split, methods=$RMOT_METHODS, L=$LAGS, VLM=$RMOT_VLM"
  "$PYTHON" scripts/rmot_benchmark.py --dataset "$ds" --root "$root" --split "$split" --config "$cfg" \
      --out "$RUN_DIR/$name" "${extra[@]}"
}

run_aerialmind() {
  local dst="$DATA_DIR/AerialMind"
  if [ "$DO_DOWNLOAD" = 1 ] && ! have "$dst/expression" && [ -z "$(find "$dst" -maxdepth 3 -type d -name expression 2>/dev/null | head -1)" ]; then
    fetch aerialmind --out "$dst"
  fi
  [ "$DO_BENCH" = 1 ] || return 0
  [ -n "$(find "$dst" -maxdepth 3 -type d -name expression 2>/dev/null | head -1)" ] || skip "AerialMind chưa có ở $dst"
  for sp in $AM_SPLITS; do
    local sm="$dst/seqmap.txt"; [ "$sp" = cross_domain ] && sm="$dst/seqmap_cd.txt"
    rmot_bench "aerialmind_$sp" aerialmind "$dst" "$sp" "$sm" configs/rmot_aerialmind.yaml
  done
}

run_refer_kitti_v2() {
  local dst="$DATA_DIR/refer-kitti-v2"
  if [ "$DO_DOWNLOAD" = 1 ] && ! have "$dst/expression"; then
    fetch refer-kitti-v2 --out "$dst"
  fi
  [ "$DO_BENCH" = 1 ] || return 0
  have "$dst/expression" || skip "Refer-KITTI-V2 chưa có ở $dst"
  rmot_bench refer_kitti_v2 refer_kitti_v2 "$dst" test "$dst/seqmap.txt" configs/rmot_kitti.yaml
}

# ------------------------------------------------------------------------------ summary
summarize() {
  "$PYTHON" - "$RUN_DIR" <<'EOF'
import sys
from pathlib import Path
run = Path(sys.argv[1])
lines = ["# Kết quả benchmark referring MOT (HOTA@L)"]
for sm in sorted(run.glob("*/summary.md")):
    lines += ["", f"## {sm.parent.name}", "", sm.read_text()]
(run / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
EOF
}

# ------------------------------------------------------------------------------ main
c_info "Datasets: ${DATASETS[*]} | DATA_DIR=$DATA_DIR | RUN_DIR=$RUN_DIR | download=$DO_DOWNLOAD bench=$DO_BENCH"
setup
if [ "$DO_BENCH" = 0 ] && [ "$PARALLEL" = 1 ] && [ ${#DATASETS[@]} -gt 1 ]; then
  # chỉ tải: chạy các bộ cùng lúc, mỗi bộ một log riêng
  pids=""
  for ds in "${DATASETS[@]}"; do
    c_info "Tải song song: $ds  (log: $RUN_DIR/download_$ds.log)"
    ( set -Eeuo pipefail; "run_$ds" ) > "$RUN_DIR/download_$ds.log" 2>&1 &
    pids="$pids $!:$ds"
  done
  for p in $pids; do
    pid="${p%%:*}"; ds="${p#*:}"
    set +e; wait "$pid"; rc=$?; set -e
    case $rc in 0) st="ok" ;; "$SKIP") st="bỏ qua (thiếu dữ liệu)" ;; *) st="LỖI (mã $rc) — xem $RUN_DIR/download_$ds.log" ;; esac
    STATUS_LINES="${STATUS_LINES}$(printf '   %-15s %s' "$ds" "$st")"$'\n'
  done
  DATASETS=()
fi
for ds in ${DATASETS[@]+"${DATASETS[@]}"}; do
  c_info "================ $ds ================"
  # chạy mỗi bộ trong subshell riêng (để lỗi ở một bộ không dừng các bộ khác); không đặt trong 'if'
  # vì bash tắt 'set -e' trong điều kiện của if
  trap - ERR; set +e
  ( set -Eeuo pipefail; trap 'on_err $LINENO "$BASH_COMMAND"' ERR; "run_$ds" ); rc=$?
  set -e; trap 'on_err $LINENO "$BASH_COMMAND"' ERR
  case $rc in 0) st="ok" ;; "$SKIP") st="bỏ qua (thiếu dữ liệu)" ;; *) st="LỖI (mã $rc) — xem log" ;; esac
  STATUS_LINES="${STATUS_LINES}$(printf '   %-15s %s' "$ds" "$st")"$'\n'
done
[ "$DO_BENCH" = 1 ] && summarize
echo
c_info "Tóm tắt:"
printf '%s' "$STATUS_LINES"
c_info "Log: $LOG"
