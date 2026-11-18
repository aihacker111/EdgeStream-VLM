#!/usr/bin/env bash
# =====================================================================================
#  EdgeStream-VLM — tải dữ liệu + chạy benchmark VAD cho từng bộ dữ liệu
#
#  Cách dùng:
#    bash scripts/run_benchmarks.sh                      # tất cả các bộ
#    bash scripts/run_benchmarks.sh a2seek uit_adrone    # chỉ vài bộ
#    bash scripts/run_benchmarks.sh --download-only xd_violence
#    bash scripts/run_benchmarks.sh --bench-only ucf_crime
#    bash scripts/run_benchmarks.sh --smoke all          # chạy thử nhanh (3 video, 300 frame)
#
#  Các bộ: a2seek  uit_adrone  drone_anomaly  muvad  ucf_crime  xd_violence
#
#  Tùy chỉnh bằng biến môi trường, ví dụ:
#    VLM=openai VLM_MODEL=qwen3-vl:8b BUDGETS="0 2 6 12" STRIDE=2 bash scripts/run_benchmarks.sh a2seek
#
#    DATA_DIR      nơi chứa dữ liệu                      (mặc định: data)
#    RUN_DIR       nơi ghi kết quả                       (mặc định: runs/bench_<ngày giờ>)
#    VLM           openai | oracle | simulated           (mặc định: openai = VLM thật qua Ollama/vLLM)
#    VLM_URL       OpenAI-compatible endpoint            (mặc định: http://localhost:11434/v1 = Ollama)
#    VLM_MODEL     tên model trên server                 (mặc định: qwen3-vl:4b)
#    BUDGETS       số lượt gọi VLM / phút cần quét       (mặc định: "0 2 6 12"; 0 = chỉ tầng rẻ)
#    STRIDE        xử lý 1 trên k frame                  (mặc định: 1 cho UAV, 2 cho CCTV 30 fps)
#    MAX_VIDEOS    giới hạn số video mỗi bộ (debug)      (mặc định: không giới hạn)
#    MAX_FRAMES    giới hạn số frame mỗi video (debug)
#    FIT_BANK      1 = khởi động ấm kho chuyển động từ video bình thường của tập train (mặc định: 1)
#    UCF_TRAIN_NORMAL 1 = tải thêm 38 GB video bình thường của UCF-Crime để fit bank (mặc định: 0)
#    KEEP_ZIPS     1 = giữ file zip sau khi giải nén     (mặc định: 0)
#    PYTHON        trình thông dịch Python               (mặc định: python3)
#
#  Dung lượng ước tính: A2Seek ~vài GB (+ video test), UIT-ADrone ~(Drive), Drone-Anomaly ~vài GB,
#  UCF-Crime ~30 GB (test) / +38 GB (train normal), XD-Violence test ~800 video.
#  Kết quả: $RUN_DIR/<bộ dữ liệu>/metrics.json, budget_curve.csv và $RUN_DIR/summary.md
# =====================================================================================
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
DATA_DIR="${DATA_DIR:-data}"
RUN_DIR="${RUN_DIR:-runs/bench_$(date +%Y%m%d_%H%M)}"
VLM="${VLM:-openai}"
VLM_URL="${VLM_URL:-http://localhost:11434/v1}"
VLM_MODEL="${VLM_MODEL:-qwen3-vl:4b}"
BUDGETS="${BUDGETS:-0 2 6 12}"
FIT_BANK="${FIT_BANK:-1}"
UCF_TRAIN_NORMAL="${UCF_TRAIN_NORMAL:-0}"
KEEP_ZIPS="${KEEP_ZIPS:-0}"
MAX_VIDEOS="${MAX_VIDEOS:-}"
MAX_FRAMES="${MAX_FRAMES:-}"

DO_DOWNLOAD=1
DO_BENCH=1
DATASETS=()
for arg in "$@"; do
  case "$arg" in
    --download-only) DO_BENCH=0 ;;
    --bench-only)    DO_DOWNLOAD=0 ;;
    --smoke)         MAX_VIDEOS="${MAX_VIDEOS:-3}"; MAX_FRAMES="${MAX_FRAMES:-300}"; BUDGETS="${BUDGETS_SMOKE:-0 6}" ;;
    -h|--help)       sed -n '2,36p' "$0"; exit 0 ;;
    all)             DATASETS+=(a2seek uit_adrone drone_anomaly muvad ucf_crime xd_violence) ;;
    a2seek|uit_adrone|drone_anomaly|muvad|ucf_crime|xd_violence) DATASETS+=("$arg") ;;
    *) echo "Không hiểu tham số: $arg (xem --help)"; exit 2 ;;
  esac
done
[ ${#DATASETS[@]} -eq 0 ] && DATASETS=(a2seek uit_adrone drone_anomaly muvad ucf_crime xd_violence)

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
  c_info "Kiểm tra môi trường Python"
  "$PYTHON" - <<'EOF' || { c_err "Thiếu package của dự án: chạy 'pip install -e .' trước"; exit 1; }
import edgestream, perception, cv2, numpy, yaml  # noqa
EOF
  "$PYTHON" -m pip install -q "huggingface_hub>=0.30" gdown pyarrow pillow >/dev/null
  if [ ! -f weights/falcon_yolo_lqcv2_baseline_jde_mot.pth ]; then
    c_warn "Không thấy weights/falcon_yolo_lqcv2_baseline_jde_mot.pth — perception sẽ không chạy được"
  fi
  if [ "$DO_BENCH" = 1 ] && [ "$VLM" = openai ]; then
    if curl -sf "${VLM_URL%/v1}/api/tags" >/dev/null 2>&1; then
      # Ollama: kéo model nếu chưa có
      if command -v ollama >/dev/null && ! ollama list 2>/dev/null | grep -q "^${VLM_MODEL}"; then
        c_info "ollama pull $VLM_MODEL"; ollama pull "$VLM_MODEL"
      fi
      c_ok "VLM server sẵn sàng: $VLM_URL ($VLM_MODEL)"
    elif curl -sf "$VLM_URL/models" >/dev/null 2>&1; then
      c_ok "VLM server (vLLM/OpenAI-compatible) sẵn sàng: $VLM_URL"
    else
      c_err "Không kết nối được VLM ở $VLM_URL."
      echo "   Mở Ollama:  ollama serve &  &&  ollama pull $VLM_MODEL"
      echo "   hoặc vLLM:  vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8000  (VLM_URL=http://localhost:8000/v1)"
      echo "   hoặc chạy không cần VLM:  VLM=oracle (cận trên) / VLM=simulated (chỉ kiểm tra pipeline)"
      exit 1
    fi
  fi
}

# ------------------------------------------------------------------------------ helpers
fetch() { "$PYTHON" scripts/datasets/fetch.py "$@"; }

bench() {   # bench <name> <config> <stride> <extra args...>
  local name="$1" cfg="$2" stride="$3"; shift 3
  local out="$RUN_DIR/$name"
  local extra=()
  [ -n "$MAX_VIDEOS" ] && extra+=(--max-videos "$MAX_VIDEOS")
  [ -n "$MAX_FRAMES" ] && extra+=(--max-frames "$MAX_FRAMES")
  if [ -f "$out/bank.npz" ]; then extra+=(--bank "$out/bank.npz"); fi
  c_info "Benchmark $name (VLM=$VLM, budgets: $BUDGETS, stride $stride)"
  # shellcheck disable=SC2086
  "$PYTHON" scripts/vad_benchmark.py "$@" --config "$cfg" --vlm "$VLM" \
      --vlm-url "$VLM_URL" --vlm-model "$VLM_MODEL" --budgets $BUDGETS --stride "$stride" \
      --out "$out" ${extra[@]+"${extra[@]}"}
  c_ok "$name xong -> $out/metrics.json"
}

fit_bank() {   # fit_bank <name> <config> <normal video dir> <stride> [fps]
  local name="$1" cfg="$2" dir="$3" stride="$4" fps="${5:-}"
  [ "$FIT_BANK" = 1 ] || return 0
  [ -d "$dir" ] && [ -n "$(ls -A "$dir" 2>/dev/null)" ] || { c_warn "$name: không có video bình thường để fit bank, bỏ qua"; return 0; }
  mkdir -p "$RUN_DIR/$name"
  [ -f "$RUN_DIR/$name/bank.npz" ] && return 0
  c_info "Fit kho chuyển động bình thường cho $name từ $dir"
  local extra=(--max-videos "${BANK_MAX_VIDEOS:-60}")
  [ -n "$fps" ] && extra+=(--fps "$fps")
  [ -n "$MAX_FRAMES" ] && extra+=(--max-frames "$MAX_FRAMES")
  "$PYTHON" scripts/fit_normal_bank.py --videos "$dir" --config "$cfg" --stride "$stride" \
      --out "$RUN_DIR/$name/bank.npz" ${extra[@]+"${extra[@]}"}
}

have() { [ -e "$1" ] && [ -n "$(ls -A "$1" 2>/dev/null)" ]; }

# ------------------------------------------------------------------------------ A2Seek (UAV, chính)
run_a2seek() {
  local root="$DATA_DIR/A2Seek" vids="$DATA_DIR/A2Seek-Videos"
  local map="$root/edgestream_videos_test.json"
  if [ "$DO_DOWNLOAD" = 1 ]; then
    fetch a2seek --root "$root"
    fetch a2seek-videos --root "$root" --videos "$vids" --split test
  fi
  [ "$DO_BENCH" = 1 ] || return 0
  [ -f "$map" ] || skip "A2Seek: chưa có $map (chạy lại không có --bench-only)"
  local scale
  scale=$("$PYTHON" -c "import json;print(json.load(open('$map')).get('_frame_scale',1.0))")
  bench a2seek configs/vad_uav.yaml "${STRIDE:-1}" --dataset a2seek \
      --a2seek-samples "$root/dataset/cot_single/test.jsonl" --videos "$map" --a2seek-frame-scale "$scale"
  # chấm bằng evaluate.py chính thức của A2Seek
  local ref="third_party/A2Seek"
  [ -d "$ref" ] || git clone -q --depth 1 https://github.com/2-mo/A2Seek "$ref"
  for b in $BUDGETS; do
    local bdir="$RUN_DIR/a2seek/budget_${b}"
    local pred="$bdir/a2seek_predictions.jsonl"
    [ -f "$pred" ] || continue
    # evaluate.py đòi đủ dự đoán cho mọi mẫu: nếu vài clip không có video, chấm trên tập con có dự đoán
    "$PYTHON" - "$root/dataset/cot_single/test.jsonl" "$pred" "$bdir/a2seek_references_covered.jsonl" <<'PY'
import json, sys
ref, pred, out = sys.argv[1:]
key = lambda r: f"{r['g_id']}/{r['v_id']}/{int(r['f_id']):05d}"
have = {key(json.loads(l)) for l in open(pred) if l.strip()}
rows = [l for l in open(ref) if l.strip() and key(json.loads(l)) in have]
open(out, "w").writelines(rows)
total = sum(1 for l in open(ref) if l.strip())
print(f"[a2seek] coverage {len(rows)}/{total} samples ({100 * len(rows) / max(total, 1):.1f}%)")
PY
    A2SEEK_DATA_ROOT="$root" "$PYTHON" "$ref/scripts/evaluate.py" --predictions "$pred" \
        --references "$bdir/a2seek_references_covered.jsonl" \
        --label-aliases "$ref/src/a2seek/configs/evaluation/label_aliases.json" \
        --output "$RUN_DIR/a2seek/budget_${b}/a2seek_official_metrics.json" \
      || c_warn "A2Seek evaluate.py lỗi ở budget $b (có thể do thiếu dự đoán cho clip không có video)"
  done
}

# ------------------------------------------------------------------------------ UIT-ADrone (UAV, VN)
run_uit_adrone() {
  local src="$DATA_DIR/UIT-ADrone" conv="$DATA_DIR/UIT-ADrone/edgestream"
  if [ "$DO_DOWNLOAD" = 1 ]; then
    have "$src" || fetch uit-adrone --out "$src" || skip "UIT-ADrone: tải thủ công từ Google Drive vào $src rồi chạy lại"
    fetch frame-labels --src "$src" --out "$conv"
  fi
  [ "$DO_BENCH" = 1 ] || return 0
  fit_bank uit_adrone configs/vad_uav.yaml "$conv/train" "${STRIDE:-1}" 30
  bench uit_adrone configs/vad_uav.yaml "${STRIDE:-1}" --dataset generic \
      --videos "$conv/videos" --labels "$conv/labels" --fps 30
}

# ------------------------------------------------------------------------------ Drone-Anomaly (UAV)
run_drone_anomaly() {
  local src="$DATA_DIR/Drone-Anomaly" conv="$DATA_DIR/Drone-Anomaly/edgestream"
  if [ "$DO_DOWNLOAD" = 1 ]; then
    if ! have "$src"; then
      c_warn "Drone-Anomaly chỉ có trên OneDrive, không tải tự động được."
      echo "   1) Mở link OneDrive trong https://github.com/Jin-Pu/Drone-Anomaly"
      echo "   2) Tải về và giải nén vào $src  (cấu trúc <scene>/<sequence>/{train,test})"
      echo "   3) Chạy lại: bash scripts/run_benchmarks.sh drone_anomaly"
      exit "$SKIP"
    fi
    fetch frame-labels --src "$src" --out "$conv"
  fi
  [ "$DO_BENCH" = 1 ] || return 0
  fit_bank drone_anomaly configs/vad_uav.yaml "$conv/train" "${STRIDE:-1}" 30
  bench drone_anomaly configs/vad_uav.yaml "${STRIDE:-1}" --dataset drone_anomaly \
      --videos "$conv/videos" --labels "$conv/labels" --fps 30
}

# ------------------------------------------------------------------------------ MUVAD (UAV, camera di chuyển)
run_muvad() {
  local src="$DATA_DIR/MUVAD" conv="$DATA_DIR/MUVAD/edgestream"
  have "$src" || skip "MUVAD chưa được công bố (https://github.com/uavano/FTDMamba). Khi có, giải nén vào $src rồi chạy lại."
  [ "$DO_DOWNLOAD" = 1 ] && fetch frame-labels --src "$src" --out "$conv"
  [ "$DO_BENCH" = 1 ] || return 0
  fit_bank muvad configs/vad_uav.yaml "$conv/train" "${STRIDE:-1}" 30
  bench muvad configs/vad_uav.yaml "${STRIDE:-1}" --dataset generic \
      --videos "$conv/videos" --labels "$conv/labels" --fps "${MUVAD_FPS:-30}"
}

# ------------------------------------------------------------------------------ UCF-Crime (CCTV)
run_ucf_crime() {
  local dst="$DATA_DIR/UCF-Crime"
  if [ "$DO_DOWNLOAD" = 1 ]; then
    local extra=()
    [ "$UCF_TRAIN_NORMAL" = 1 ] && extra+=(--with-train-normal)
    [ "$KEEP_ZIPS" = 1 ] && extra+=(--keep-zips)
    [ -f "$dst/Temporal_Anomaly_Annotation_for_Testing_Videos.txt" ] || fetch ucf-crime --out "$dst" ${extra[@]+"${extra[@]}"}
  fi
  [ "$DO_BENCH" = 1 ] || return 0
  fit_bank ucf_crime configs/vad_cctv.yaml "$dst/train_normal" "${STRIDE:-2}"
  bench ucf_crime configs/vad_cctv.yaml "${STRIDE:-2}" --dataset ucf_crime \
      --videos "$dst/videos" --annotations "$dst/Temporal_Anomaly_Annotation_for_Testing_Videos.txt"
}

# ------------------------------------------------------------------------------ XD-Violence (CCTV + phim)
run_xd_violence() {
  local dst="$DATA_DIR/XD-Violence"
  if [ "$DO_DOWNLOAD" = 1 ]; then
    local extra=()
    [ -n "$MAX_VIDEOS" ] && extra+=(--max-videos "$MAX_VIDEOS")
    fetch xd-violence --out "$dst" ${extra[@]+"${extra[@]}"}
  fi
  [ "$DO_BENCH" = 1 ] || return 0
  bench xd_violence configs/vad_cctv.yaml "${STRIDE:-2}" --dataset xd_violence \
      --videos "$dst/data/video" --annotations "$dst/data/test_annotations.txt" \
      --test-list "$dst/data/test_list.txt"
}

# ------------------------------------------------------------------------------ summary
summarize() {
  "$PYTHON" - "$RUN_DIR" <<'EOF'
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
lines = ["# Kết quả benchmark", "",
         "| Bộ dữ liệu | Budget (/phút) | Lượt VLM | Micro-AUC online | Micro-AUC offline | Macro-AUC offline | AP offline | A2Seek top-1 |",
         "|---|---|---|---|---|---|---|---|"]
for m in sorted(run.glob("*/metrics.json")):
    for r in json.loads(m.read_text()):
        f = lambda k: "" if r.get(k) is None else (f"{100 * r[k]:.1f}" if isinstance(r[k], float) else str(r[k]))
        lines.append(f"| {m.parent.name} | {r['budget_calls_per_min']:g} | {r['vlm_calls']} | {f('online_micro_auc')} | "
                     f"{f('offline_micro_auc')} | {f('offline_macro_auc')} | {f('offline_ap')} | "
                     f"{f('a2seek_first_category_accuracy')} |")
(run / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
EOF
}

# ------------------------------------------------------------------------------ main
c_info "Datasets: ${DATASETS[*]} | DATA_DIR=$DATA_DIR | RUN_DIR=$RUN_DIR | download=$DO_DOWNLOAD bench=$DO_BENCH"
setup
for ds in "${DATASETS[@]}"; do
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
