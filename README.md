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
event_cards/uav_surveillance/  24 VAD categories (A2Seek + UAV traffic) + _hypotheses.yaml
configs/vad_uav.yaml, vad_cctv.yaml   VAD pipeline settings
edgestream/                 reasoning pipeline
  harness/                  causal video replay, ring buffer, budget ledger, JSONL logs, perception cache
  scene/                    ego-motion (background LK + RANSAC), flow field, track histories, density
  candidates/               counter_flow, stopped_in_flow, sudden_stop, speeding, congestion_forming
  gate/                     priority model, token bucket with reserve, density-adaptive threshold
  memory/                   scene tier + entity tier under a visual-token budget
  vlm/                      cards, Set-of-Mark prompt builder, backends (openai / transformers / simulated)
  events/                   aggregation, cooldown, evidence clips, routing
  pipeline.py, run.py       orchestration + CLI
  vad/                      VAD: normality models, pixel cues, frame scorer, metrics, loaders, pipeline, CLI
perception/                 detector (YOLO11s-P2) + AERO-MOT tracker, ported
  perceptor.py              Perceptor: frame -> FrameOutput (the only API downstream uses)
  types.py                  TrackOutput / FrameOutput contracts
  config.py                 YAML config, AERO-MOT paper tracker defaults
  preprocess.py             training-identical preprocessing
  postprocess.py            NMS + low-score candidate band
  models/                   detector wrapper, ReID head, EAD/LQC modules, YAMLs
  tracker/                  SPEAR + QAM tracker (from AERO-MOT)
  cli.py                    run on a video / frame folder
tests/                      pytest (port parity, preprocessing, end-to-end, VAD)
scripts/                    vad_benchmark.py, fit_normal_bank.py, synthetic_demo.py
weights/                    checkpoints (not tracked by git)
```

## Video anomaly detection (UAV focus)

Streaming, object-centric, budgeted VLM anomaly detection. Same perception and budget machinery as the
traffic pipeline; the cheap tier proposes *object-level hypotheses*, the VLM verifies them open-set, and
object beliefs are turned into the frame scores that VAD benchmarks evaluate.

```
frames ─(stride)─► perception ─► scene (ego-motion) ─► cheap tier ───────────────► priority ─► VOI gate ─► VLM ─► beliefs ─► frame scores
                                                       human rules: running, falling,              (token      (open-set:  (product   online (causal)
                                                         lying_still, loitering,                    bucket,      taxonomy    of experts  offline (protocol)
                                                         close_interaction, crowd_change            density)     prefix +    with the    A2Seek answers
                                                       normality: location rarity (Poisson,                       hypothesis  cheap tier)
                                                         ego-stabilised), kinematic k-NN bank                     cards)
                                                       pixels: residual motion, static change
```

| Component | File | Idea |
|---|---|---|
| Object rules | `edgestream/candidates/objects.py` | scale-free thresholds (object sizes/s); walking speed and body shape of people learned online from the scene |
| Location rarity | `edgestream/vad/normality.py` `ClassLocationMap` | Poisson rate per cell and class in drift-compensated coordinates, Gamma prior at the global rate, own counts removed: "a pedestrian where in 2 min nobody walked" |
| Kinematic bank | `KinematicBank` | k-NN in robust-standardised motion space, unit = 95th-percentile leave-one-out distance of normal samples; self-gated insertion; persistence filter; warm start from normal videos |
| Pixel cues | `edgestream/vad/residual.py` | residual motion after camera compensation outside every box (animals, objects the detector does not know); dual-background static change (lost item, litter) |
| VOI gate | `edgestream/gate/gate.py` | order by p·severity·uncertainty/cost; re-ask only uncertain answers or a "no" whose evidence grew |
| Object-level sharing | `edgestream/vad/pipeline.py` | one VLM answer settles every hypothesis on the same object; new hypotheses on a verified object inherit it |
| Open-set prompt | `edgestream/vlm/prompt_vad.py` | fixed taxonomy prefix (KV-cacheable) + 1-3 cards chosen by the hypothesis (class-aware); answer can be any category, other_anomaly or normal |
| Frame scorer | `edgestream/vad/scorer.py` | b = σ(logit p_vlm + η logit q); unverified capped at w_u·q; attack/release filter; online scores respect VLM latency, offline scores are the usual protocol |
| Metrics / loaders | `edgestream/vad/metrics.py`, `datasets.py` | micro/macro AUC, AP, AUC on abnormal videos, EER; UCF-Crime, XD-Violence, A2Seek (+ answers in A2Seek's own format), generic frame labels |

```bash
# one video
python -m edgestream.vad.run --video clip.mp4 --config configs/vad_uav.yaml --vlm openai --save-video --out runs/vad_demo

# warm-start the motion bank from normal training videos (optional)
python scripts/fit_normal_bank.py --videos data/UIT-ADrone/train/videos --out runs/banks/uit.npz

# benchmark + budget sweep (perception is cached after the first pass)
python scripts/vad_benchmark.py --dataset generic --videos data/UIT-ADrone/test/videos \
    --labels data/UIT-ADrone/test/labels --fps 30 --budgets 0 2 6 12 --vlm openai --out runs/bench_uit
python scripts/vad_benchmark.py --dataset ucf_crime --videos data/UCF_Crimes/Videos \
    --annotations data/UCF_Crimes/Temporal_Anomaly_Annotation_for_Testing_Videos.txt \
    --config configs/vad_cctv.yaml --stride 2 --budgets 0 6 --out runs/bench_ucf
python scripts/vad_benchmark.py --dataset a2seek --a2seek-samples data/A2Seek/dataset/cot_single/test.jsonl \
    --videos "data/A2Seek-Videos/{g_id}/{v_id}.mp4" --out runs/bench_a2seek
#   -> runs/bench_a2seek/budget_6/a2seek_predictions.jsonl works with A2Seek's scripts/evaluate.py
```

**All benchmarks in one go** — download, convert labels, warm-start the motion bank, budget sweep, A2Seek's
official evaluator, summary table:

```bash
ollama serve & ollama pull qwen3-vl:4b                       # or a vLLM server (VLM_URL=...)
bash scripts/run_benchmarks.sh --smoke a2seek                # quick check: 3 videos, 300 frames
bash scripts/run_benchmarks.sh all                           # a2seek uit_adrone drone_anomaly muvad ucf_crime xd_violence
VLM=oracle BUDGETS="0 2 6 12" bash scripts/run_benchmarks.sh --bench-only uit_adrone   # upper bound
```

Results: `runs/bench_<date>/<dataset>/metrics.json`, `budget_curve.csv`, `summary.md`. Drone-Anomaly (OneDrive) must
be downloaded by hand into `data/Drone-Anomaly`; MUVAD is not released yet; UIT-ADrone comes from Google Drive
(gdown; download by hand if Drive refuses). Per-step tools: `scripts/datasets/fetch.py --help`.

VLM backends: `openai` (Ollama/vLLM, real), `transformers` (real), `oracle` (answers from the ground-truth
labels with a set accuracy: upper bound that isolates gate/memory/budget from VLM quality), `simulated`
(echoes the cheap tier: plumbing only, never report its numbers).

Notes: A2Seek's `f_id` -> video frame mapping must be checked on your copy of A2Seek-Videos
(`--a2seek-frame-scale`). The kinematic bank transfers across scenes; location statistics never do (they
are always learned online). Budget 0 = cheap tier only (the ablation baseline).

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
