# EdgeStream-VLM

Budget-aware streaming VLM event understanding for dense multi-agent scenes
(ground CCTV and UAV) on edge devices.

```
frames ─► perception ─► scene state ─► candidates ─► priority ─► memory ─► density gate ─► VLM ─► events ─► routing
          (AERO-MOT)    (ego-motion,    (motion       (logistic)  (2 tiers,  (token bucket,   (event   (merge,   (command
                         flow field)     rules)                    budget)    adaptive tau)    cards)   evidence)  center ...)
```

Everything runs end to end on a video (`python -m edgestream.run`). Open work is in [TODO.md](TODO.md).

## Layout

```
configs/perception.yaml     detector + tracker settings
configs/pipeline.yaml       scene / candidates / gate / memory / VLM / events settings
event_cards/traffic/        one YAML per event type (definition, cues, hard negatives)
edgestream/                 reasoning pipeline
  harness/                  causal video replay, ring buffer, budget ledger, JSONL logs, perception cache
  scene/                    ego-motion (background LK + RANSAC), flow field, track histories, density
  candidates/               counter_flow, stopped_in_flow, sudden_stop, speeding, congestion_forming
  gate/                     priority model, token bucket with reserve, density-adaptive threshold
  memory/                   scene tier + entity tier under a visual-token budget
  vlm/                      cards, Set-of-Mark prompt builder, backends (openai / transformers / simulated)
  events/                   aggregation, cooldown, evidence clips, routing
  pipeline.py, run.py       orchestration + CLI
  rmot/                     online aerial referring MOT ("search, don't filter"), see below
perception/                 detector (YOLO11s-P2) + AERO-MOT tracker, ported
  perceptor.py              Perceptor: frame -> FrameOutput (the only API downstream uses)
  types.py                  TrackOutput / FrameOutput contracts
  config.py                 YAML config, AERO-MOT paper tracker defaults
  preprocess.py             training-identical preprocessing
  postprocess.py            NMS + low-score candidate band
  models/                   detector wrapper, ReID head, EAD/LQC modules, YAMLs
  tracker/                  SPEAR + QAM tracker (from AERO-MOT)
  cli.py                    run on a video / frame folder
tests/                      pytest (port parity, preprocessing, end-to-end, RMOT)
scripts/                    rmot_benchmark.py, run_benchmarks.sh, datasets/fetch.py, synthetic_demo.py
weights/                    checkpoints (not tracked by git)
```

## Online aerial referring MOT — "search, don't filter" (main direction)

Task: while the drone flies, output the boxes and identities of every object that matches a
natural-language query ("white cars moving opposite to us"), with a bounded commit lag L.

Existing MLLM RMOT pipelines *filter*: each candidate track is accepted or rejected on its own
(YesTrack: per-candidate yes/no on keyframes; VMRMOT: motion turned into text). Identity errors of
the tracker are inherited, temporal validity is ignored, and every decision is final. Here the
referring decision and the association are **one joint inference** over a hypothesis graph,
re-solved on a receding horizon:

```
frames ─► AERO-MOT tracks + background affine (ego motion)
       ─► tracklet hypothesis graph: tracks cut early into ≤1 s pieces; cont / stitch link hypotheses
       ─► provisional prior: class family + measured kinematics (moving, turning, same/opposite to the drone)
       ─► deadline-aware scheduler ─► async VLM pool (vLLM, n_workers)
             set-level comparative prompt:  [overview with marks | close-up gallery | measured facts | EXPRESSION]
             pairwise link prompt for uncertain stitches
       ─► window solver every 0.5 s (CP-SAT): labels + links jointly, frozen committed boundary
       ─► commit frames older than t − L (+ stitched-gap interpolation, zoom re-detection)
```

* **One joint objective.** `max Σ n_i(z_i − τ)y_i + κΣ l_e x_e − λΣ|y_i − y_j|x_e` with flow constraints
  (`edgestream/rmot/solver.py`). A wrong VLM answer inside a confident chain is out-voted; a
  fragmented track is re-joined when the stitch evidence and the referring evidence agree.
* **Commit at t − L; offline = L = ∞.** Committed frames are never re-written, but committed pieces
  stay in the window as variables so their evidence keeps flowing to new pieces ("frozen outputs,
  live beliefs"). HOTA@L for L ∈ {0, 2, 5, ∞} is the main result curve.
* **Verifier prompt v2 (default).** One candidate per call, answered with ONE token (P(yes) read
  from the first token's logprobs, no JSON decoding). Layout: unmarked frame (byte-identical for
  every candidate at that time) -> description -> candidate close-ups [surroundings | zoom | 1 s
  earlier] + box in 0-1000 coordinates + measured motion. The frame and description are a shared
  prefix, so the server's prefix cache encodes them once. The keyframe-filter baselines use the
  same prompt (without the motion text), so search-vs-filter is the only difference.
  The older set-level prompt (8 marked candidates per call) is kept as the ablation `search_set`:
  with Qwen3-VL-2B it ranked candidates barely above chance (AUC 0.67 vs 0.94 for one candidate).
* **Deadline-aware scheduling.** value = Σ 4p(1−p) × frames still uncommitted at completion / (1+answers),
  divided by the time left before the first deadline; requests that cannot finish in time are worth 0.
* **Test-time scaling knobs**: K answers per piece, R partitions per anchor (`rounds`), group size.
* **Measured kinematics in ego-stabilised coordinates**: a single frame cannot tell "parked" from
  "waiting at a light" or "same direction as us" from a moving drone; these are measured and given
  to the VLM as facts (and used for the provisional prior).

### Benchmarks (official protocols)

| dataset | role | split | metric |
|---|---|---|---|
| AerialMind (AAAI 2026) | primary, UAV | in-domain: 17 VisDrone test sequences (4,986 rows); cross-domain: 13 UAVDT sequences (3,415 rows) | HOTA / DetA / AssA / LocA |
| Refer-KITTI-V2 | generality, ground vehicle | 0005, 0011, 0013, 0019 (861 rows) | same |

`edgestream/rmot/hota.py` is a port of TrackEval's HOTA and of the RMOT fork's folder layout
(`<method>/<video>/<expression>/{gt.txt,predict.txt}`, frames 1..N, one sequence per
`video+expression` row). `tests/test_rmot.py` checks it against TrackEval's own code;
`--verify-trackeval PATH` repeats the check on every evaluated sequence. Leakage: AERO-MOT was
trained on VisDrone2019-MOT train (val for validation); AerialMind's in-domain test set is
VisDrone test-dev and its cross-domain set is UAVDT, neither seen by the detector.

### Baselines (same perception, prompts, VLM pool and lag rule)

| name | what |
|---|---|
| `filter_kf` | YesTrack-style keyframe filter: per-candidate yes/no, track keeps its last answer (nearest answer when L = ∞) |
| `filter_ef` | every-frame filter (keyframe_every 1) |
| `filter_set` | keyframe filter with set prompts and facts (verifier without search) |
| `prior` | no VLM: class family + kinematics |
| `search_single`, `search_nolink` | ablations: single-candidate prompts / no stitching |
| `ceiling` | the method with a ground-truth verifier: perception ceiling, never reported as a result |

Online, every method gets the same VLM throughput (n_workers × latency), so they are matched in
VLM time; offline, `--match-compute` caps each baseline at the calls the method used on that stream.

### Run

```bash
pip install -e ".[rmot]"                                   # ortools (CP-SAT)
python scripts/datasets/fetch.py aerialmind --out data/AerialMind        # HF + official seqmaps
vllm serve Qwen/Qwen3-VL-4B-Instruct-FP8 --enable-prefix-caching --max-model-len 4096 --max-num-seqs 16 \
     --limit-mm-per-prompt '{"image": 2, "video": 0}' --mm-processor-kwargs '{"max_pixels": 1048576}' \
     --gpu-memory-utilization 0.60 --port 8000                                            # one 24 GB GPU
python scripts/rmot_benchmark.py --dataset aerialmind --root data/AerialMind --split in_domain \
     --seqmap data/AerialMind/seqmap.txt --methods search,filter_kf,filter_ef,prior --lags 0,2,5,inf \
     --vlm openai --vlm-url http://localhost:8000/v1 --n-workers 4 --match-compute --out runs/rmot_am_id
# or everything (both AerialMind splits + Refer-KITTI-V2), download included:
RMOT_VLM=openai bash scripts/run_benchmarks.sh rmot
# quick plumbing check without a VLM (ground-truth oracle, NOT a result):
RMOT_VLM=oracle MAX_VIDEOS=1 MAX_EXPR=3 bash scripts/run_benchmarks.sh --bench-only aerialmind
```

Perception runs once per video and is cached (`<out>/../perception_cache/<dataset>/<video>/`,
tracks + ReID embeddings + ego affine). VLM answers are cached by exact prompt content
(`<out>/vlm_cache_<backend>.jsonl`), so the lag sweep and the ablations re-use them; the stored
latency is replayed, so the logical clock is unchanged. `--tracks-dir` runs everything on an
external tracker's MOT files instead (perception-matched comparison with other papers).

### Verifier fine-tuning (LoRA, AerialMind train split only)

Zero-shot verifiers top out around AUC 0.75 on AerialMind (`rmot_diagnose.py --mode probe`).
The verifier can be adapted on the **train** videos (never the test lists) with the exact
inference prompt (v2, built by `VLMVerifier.build_v2`) and a one-word yes/no target:

```bash
python scripts/rmot_make_sft.py --root data/AerialMind --out data/rmot_sft --max-expr 100 --pos 4 --neg 4
python scripts/rmot_train_lora.py --data data/rmot_sft --model Qwen/Qwen3-VL-2B-Instruct --out runs/lora_vl2b \
       --epochs 1 --batch 2 --grad-accum 8 --lr 1e-4               # stop vLLM first; --qlora for 4B on 24 GB
python scripts/rmot_train_lora.py --data data/rmot_sft --model Qwen/Qwen3-VL-2B-Instruct --out runs/lora_vl2b \
       --adapter runs/lora_vl2b/best --merge-to runs/lora_vl2b/merged
vllm serve runs/lora_vl2b/merged --served-model-name rmot --enable-prefix-caching ...   # same flags as above
python scripts/rmot_diagnose.py --mode probe ... --vlm-model rmot   # compare with the zero-shot probe
```

**Without a server.** `--vlm transformers` loads the model inside the benchmark process (base +
`--vlm-adapter`, merged at load, or a merged folder as `--vlm-model`). Prompt v2 needs one token,
so each request is a single forward pass; concurrent requests of the verifier pool are grouped
into one padded batch (`--vlm-batch`, default 8), and a batch gives the same P(yes) as scoring
each request alone. `--jobs` is forced to 1 (one model in memory).

```bash
python scripts/rmot_benchmark.py --dataset aerialmind --root data/AerialMind --split in_domain \
     --seqmap data/AerialMind/seqmap.txt --methods search,filter_kf --lags 0,2,5,inf \
     --vlm transformers --vlm-model Qwen/Qwen3-VL-2B-Instruct --vlm-adapter runs/lora_vl2b/best \
     --n-workers 4 --match-compute --out runs/rmot_am_id_lora
```

The answer cache is named after backend, model and adapter (`vlm_cache_<backend>_<model>+<adapter>.jsonl`),
so zero-shot and fine-tuned answers never mix in one output folder.

Candidates are the tracker's own pieces (test-time distribution), labelled from GT (referred
fraction ≥ 0.6 → yes, ≤ 0.1 → no); half the negatives are hard (right class, referred by another
expression). Validation is on held-out train videos; the best adapter by AUC is kept.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision          # pick the build for your CUDA / Apple Silicon
pip install -e ".[dev,fast]"           # "fast" = cython-bbox (optional)
pytest -q
```

## Quick start

```bash
# perception only
python -m perception.cli --source path/to/video.mp4 --out runs/demo --save-video

# full pipeline (simulated VLM: tests the plumbing, NOT real judgements)
python -m edgestream.run --video data/videos/drone_highway_1080p15.mp4 --out runs/highway

# re-run reasoning with different budgets without re-running the detector
python -m edgestream.run --video data/videos/drone_highway_1080p15.mp4 \
    --perception-cache runs/highway_perception/tracks.jsonl --out runs/highway_b2
```

### Real VLM

Any OpenAI-compatible server works. On a Mac, for example with Ollama (>= 0.12.7):

```bash
ollama pull qwen3-vl:4b
# configs/pipeline.yaml -> vlm.openai: base_url http://localhost:11434/v1, model qwen3-vl:4b
python -m edgestream.run --video data/videos/drone_highway_1080p15.mp4 --out runs/highway_qwen --vlm openai
```

With vLLM/SGLang the JSON schema is enforced (constrained decoding) and the verdict
confidence comes from token logprobs. If a server rejects `logprobs` or
`response_format`, the backend drops them and falls back to parsing.
If you use a "thinking" model variant, raise `vlm.max_new_tokens`.

### Outputs of a run

`summary.json` (budget, candidates, events), `pipeline.mp4` (annotated video),
`candidates.jsonl`, `gate.jsonl`, `vlm_calls.jsonl` (full prompt text + raw answer),
`events.jsonl`, `route_*.jsonl` (per receiver), `prompts/` (exact images sent to the VLM),
`evidence/` (clip + keyframe per event).

```python
from perception import Perceptor
p = Perceptor.from_yaml("configs/perception.yaml")
out = p.process(frame_bgr, timestamp=t)          # FrameOutput
for trk in out.tracks:
    trk.global_id, trk.cls_name, trk.tlwh, trk.velocity, trk.quality, trk.is_recovered
```

## Checkpoints

| File | What it is | Tracking mode |
|---|---|---|
| `weights/falcon_yolo_lqcv2_baseline_jde_mot.pth` (default) | JDE MOT model, epoch 21: YOLO11s-P2 baseline PAN-FPN + 128-d dense ReID head, 10 VisDrone classes, trained at 960×544 | full AERO-MOT: SPEAR + QAM + ReID gallery; `embedding` and `kappa` are filled |
| `weights/falcon_detector_1.pth` | detection-only (stage 1), epoch 120, same architecture without ReID head | motion-only; `embedding`/`kappa` are `None` |

Both match `perception/models/cfg/yolo_falcon_lqcv2_baseline.yaml` (no EAD/LQC modules),
so no Ultralytics patching is needed. The ReID head is detected from the checkpoint
and appearance is switched on automatically (`tracker.use_appearance: auto`).

## License

`perception/` is derived from AERO-MOT and inherits AGPL-3.0 (see
`perception/LICENSE` and `perception/NOTICE`).
