"""Streaming, object-centric, budgeted VLM video anomaly detection.

frames --(stride)--> perception --> scene state (ego-motion, flow) --> cheap tier
    object rules + normality models + pixel residuals  -> candidates (hypotheses)
    traffic rules (optional)                            -> candidates
--> priority --> VOI/density gate (token bucket) --> memory --> VAD prompt --> VLM
--> beliefs --> frame scores (online + offline), events, A2Seek probe answers
"""
from __future__ import annotations

import json
import statistics
import time
from collections import Counter
from dataclasses import fields
from itertools import count
from pathlib import Path

import cv2
import numpy as np
import yaml

from ..candidates import MotionCandidateGenerator, MotionRules
from ..candidates.objects import ObjectAnomalyGenerator, ObjectRules
from ..events import EventAggregator, EventConfig
from ..gate import DensityGate, GateConfig
from ..harness import BudgetLedger, FrameRingBuffer, JsonlLogger
from ..harness.stream import open_stream
from ..memory import MemoryConfig, StreamingMemory
from ..scene import SceneState
from ..vlm import build_backend, load_cards
from ..vlm.backends import RawResult, VLMBackend
from ..vlm.prompt_vad import VADPromptBuilder, canonical_category, load_hypotheses
from .normality import KinematicBank
from .scorer import FrameScorer, ScorerConfig, vlm_probability


def _dc(cls, raw):
    raw = raw or {}
    names = {f.name for f in fields(cls)}
    return cls(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in raw.items() if k in names})


def load_vad_config(path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(Path(path).resolve().parent.parent)
    return cfg


class OracleBackend(VLMBackend):
    """Ground-truth verifier with controllable accuracy: isolates gate / memory / budget effects
    from VLM quality (upper-bound ablation). Answers from the frame label at the call time."""
    name = "oracle"

    def __init__(self, accuracy: float = 1.0, latency_s: float = 0.0, seed: int = 0):
        self.labels, self.fps = None, 1.0
        self.accuracy, self.latency_s = accuracy, latency_s
        self.rng = np.random.default_rng(seed)

    def set_labels(self, labels, fps: float) -> None:
        self.labels, self.fps = labels, fps

    def generate(self, req):
        y = 0
        if self.labels is not None and len(self.labels):
            i = int(min(max(round(req.meta["t"] * self.fps), 0), len(self.labels) - 1))
            y = int(self.labels[i])
        if self.rng.random() > self.accuracy:
            y = 1 - y
        verdict = "yes" if y else "no"
        text = json.dumps({"verdict": verdict, "event_type": req.meta.get("default_event", "other_anomaly")
                           if y else "normal", "severity": req.card.severity, "description": "[oracle]"})
        return RawResult(text, 0, 0, self.latency_s, 0.95 if y else 0.05)


class _Recorder:
    """Wraps a perceptor and writes tracks.jsonl so later runs (budget sweeps) can replay it."""

    def __init__(self, inner, path: Path):
        self.inner, self.f = inner, open(path, "w")
        self.appearance = getattr(inner, "appearance", False)
        self.cfg = getattr(inner, "cfg", None)

    def reset(self):
        self.inner.reset()

    def process(self, frame, timestamp=None):
        out = self.inner.process(frame, timestamp=timestamp)
        self.f.write(json.dumps(out.to_dict()) + "\n")
        return out

    def close(self):
        self.f.close()


class VADPipeline:
    def __init__(self, cfg: dict, perceptor=None, vlm: VLMBackend | None = None):
        self.cfg = cfg
        root = Path(cfg["_root"])
        self.root = root
        self.window_s = float(cfg.get("window_s", 0.5))
        self._perceptor = perceptor
        self.obj_rules = _dc(ObjectRules, cfg.get("objects"))
        tr = dict(cfg.get("traffic") or {})
        self.traffic_rules = None
        if tr.pop("enabled", False):
            if "types" in tr:
                tr["enabled"] = tr.pop("types")
            self.traffic_rules = _dc(MotionRules, tr)
        self.gate_cfg = _dc(GateConfig, cfg.get("gate"))
        self.mem_cfg = _dc(MemoryConfig, cfg.get("memory"))
        self.ev_cfg = _dc(EventConfig, cfg.get("events"))
        self.sc_cfg = _dc(ScorerConfig, cfg.get("scorer"))
        cards_dir = root / cfg["vlm"]["cards_dir"]
        self.cards = load_cards(cards_dir)
        self.builder = VADPromptBuilder(self.cards, load_hypotheses(cards_dir),
                                        overview_width=self.mem_cfg.keyframe_width,
                                        max_new_tokens=int(cfg["vlm"].get("max_new_tokens", 64)))
        if vlm is not None:
            self.vlm = vlm
        elif cfg["vlm"].get("backend") == "oracle":
            self.vlm = OracleBackend(**cfg["vlm"].get("oracle", {}))
        else:
            self.vlm = build_backend(cfg["vlm"])
        self.bank_path = cfg.get("normality", {}).get("bank")

    @property
    def perceptor(self):
        if self._perceptor is None:
            from perception import Perceptor
            self._perceptor = Perceptor.from_yaml(str(self.root / self.cfg["perception_config"]))
        return self._perceptor

    # ------------------------------------------------------------------ run one video
    def run(self, source: str, out_dir: str | None = None, labels=None, fps: float | None = None,
            stride: int = 1, probes: list | None = None, max_frames: int | None = None,
            save_video: bool = False, perceptor=None, record_tracks: bool = False,
            bank: KinematicBank | None = None, freeze_bank: bool = False) -> dict:
        out = Path(out_dir) if out_dir else None
        if out:
            out.mkdir(parents=True, exist_ok=True)
        stream = open_stream(source, fps=fps, max_frames=max_frames)
        native_fps = stream.fps
        stride = max(1, int(stride))
        eff_fps = native_fps / stride
        perc = perceptor or self.perceptor
        if hasattr(perc, "cfg") and perc.cfg is not None and hasattr(perc.cfg, "tracker"):
            perc.cfg.tracker.frame_rate = int(round(eff_fps))
        perc.reset()
        if record_tracks and out:
            perc = _Recorder(perc, out / "tracks.jsonl")
        if isinstance(self.vlm, OracleBackend):
            self.vlm.set_labels(labels, native_fps)

        scene = SceneState(fps=eff_fps, **self.cfg.get("scene", {}))
        ids = count(1)
        if bank is None:
            bank = KinematicBank(**self.cfg.get("normality", {}).get("knn", {}))
            if self.bank_path and Path(self.root / self.bank_path).exists():
                bank.load(str(self.root / self.bank_path))
        objgen = ObjectAnomalyGenerator(self.obj_rules, ids=ids, bank=bank,
                                        use_pixels=self.cfg.get("objects", {}).get("use_pixels", True))
        objgen.frozen = freeze_bank
        trafgen = None
        if self.traffic_rules is not None:
            trafgen = MotionCandidateGenerator(self.traffic_rules)
            trafgen._ids = ids
        gate = DensityGate(self.gate_cfg)
        memory = StreamingMemory(self.mem_cfg)
        scorer = FrameScorer(self.sc_cfg)
        agg = EventAggregator(self.ev_cfg)
        ledger = BudgetLedger()
        log = JsonlLogger(out) if out else None
        ring = FrameRingBuffer(seconds=10.0, fps=eff_fps)

        probes = sorted(probes or [], key=lambda p: p["frame"])
        pi, answers = 0, []
        frame_scores_online, frame_scores_idx = [], []
        next_window = 0.0
        n_proc, perc_ms = 0, []
        cand_types, sent = Counter(), set()
        seen_cands = set()
        last_fo, writer = None, None
        t0 = time.perf_counter()
        hyp_of = lambda c: self.builder.hypotheses_for(c)[0]   # noqa: E731

        for pkt in stream:
            if pkt.frame_idx % stride:
                continue
            t, frame = pkt.t, pkt.image
            fo = perc.process(frame, timestamp=t)
            last_fo = fo
            perc_ms.append(fo.latency_ms.get("total", 0.0))
            n_proc += 1
            scene.update(fo, frame)
            objgen.observe_frame(scene, frame)
            ring.push(t, frame)

            if t + 1e-9 >= next_window:
                next_window += self.window_s
                cands = objgen.step(scene)
                if trafgen is not None:
                    cands += trafgen.step(scene)
                for c in cands:
                    c.priority = gate.priority(c)
                    if c.cand_id not in seen_cands:
                        seen_cands.add(c.cand_id)
                        cand_types[c.type_hint] += 1
                scorer.update_candidates(t, cands, hyp_of)
                scorer.set_dense(objgen.dense_score)
                memory.observe(t, frame, scene, cands, gate.last_tau)
                groups = self._group_by_object(cands)
                self._inherit(groups, t, scorer)
                reps = [max(g, key=lambda c: c.priority) for g in groups.values()]
                selected, ginfo = gate.step(t, reps, scene.density, scene.density_ref, self._cost_fn(memory))
                if log:
                    log.write("gate", {**ginfo, "dense": round(objgen.dense_score, 3)})
                for cand in selected:
                    sent.add(cand.cand_id)
                    ctx = memory.context(cand, scene)
                    req = self.builder.build(cand, ctx, frame, self._boxes(cand, fo, scene))
                    v = self.vlm(req)
                    ledger.record(cand.type_hint, v.visual_tokens, v.prompt_tokens, v.completion_tokens,
                                  v.latency_s)
                    p = vlm_probability(v.verdict, v.confidence)
                    cat = "normal" if v.verdict == "no" else canonical_category(v.event_type, self.cards)
                    if v.verdict == "yes" and cat == "normal":
                        cat = req.meta["default_event"]
                    lat = v.latency_s if self.cfg["vlm"].get("count_latency", True) else 0.0
                    # the open-set answer is about the OBJECT, so it settles every hypothesis on it
                    for m in groups[self._obj_key(cand)]:
                        m.vlm_p, m.vlm_type = p, cat
                        if m is not cand:
                            m.times_sent, m.last_sent_t = max(m.times_sent, cand.times_sent), t
                            m.last_sent_priority = max(m.priority, m.last_sent_priority)
                        scorer.add_verdict(t + lat, m.cand_id, p, cat)
                    if log:
                        log.write("vlm_calls", {"t": round(t, 2), "cand_id": cand.cand_id, "type": cand.type_hint,
                                                "hypotheses": req.meta["hypotheses"], "priority": round(cand.priority, 3),
                                                "verdict": v.verdict, "category": cat, "p": round(p, 3),
                                                "parse_ok": v.parse_ok, "raw": v.raw_text,
                                                "visual_tokens": req.visual_tokens, "latency_s": round(v.latency_s, 3)})
                    if v.verdict == "yes":
                        ev, is_new = agg.add(t + lat, cand, v, event_type=cat)
                        if ev is not None and is_new and log:
                            log.write("events", ev.to_dict())

            s = scorer.frame(t)
            frame_scores_online.append(s)
            frame_scores_idx.append(pkt.frame_idx)
            while pi < len(probes) and probes[pi]["frame"] < pkt.frame_idx + stride:
                answers.append(self._answer(probes[pi], t, scorer, scene, fo))
                pi += 1
            if save_video and out:
                writer = self._draw(writer, out, frame, fo, objgen, scorer, s, t, eff_fps, ledger, scene)

        while pi < len(probes):                        # probes after the end of the stream
            answers.append(self._answer(probes[pi], scene.t, scorer, scene, last_fo))
            pi += 1
        if writer is not None:
            writer.release()
        if isinstance(perc, _Recorder):
            perc.close()

        n_native = (frame_scores_idx[-1] + stride) if frame_scores_idx else 0
        if labels is not None:
            n_native = len(labels) if max_frames is None else min(len(labels), max_frames)
        online = self._expand(frame_scores_idx, frame_scores_online, n_native)
        offline = self._expand(frame_scores_idx, list(scorer.offline_scores()), n_native)
        duration = n_proc / eff_fps if eff_fps else 0.0
        result = {
            "source": str(source), "native_fps": native_fps, "stride": stride, "processed_frames": n_proc,
            "native_frames": int(n_native), "duration_s": round(duration, 2),
            "wall_s": round(time.perf_counter() - t0, 2),
            "perception_ms_median": round(statistics.median(perc_ms), 1) if perc_ms else None,
            "candidates_total": len(seen_cands), "candidates_by_type": dict(cand_types),
            "candidates_sent": len(sent), "budget": ledger.summary(duration),
            "events": [e.to_dict() for e in agg.events], "probe_answers": answers,
            "bank_items": len(bank),
        }
        if out:
            np.save(out / "scores_online.npy", online)
            np.save(out / "scores_offline.npy", offline)
            with open(out / "summary.json", "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)
            log.close()
        result["scores_online"], result["scores_offline"] = online, offline
        result["_bank"], result["_scorer"] = bank, scorer
        return result

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _obj_key(c) -> tuple:
        return tuple(sorted(c.track_ids)) if c.track_ids else ("region", c.cand_id)

    def _group_by_object(self, cands) -> dict:
        groups: dict = {}
        for c in cands:
            groups.setdefault(self._obj_key(c), []).append(c)
        return groups

    @staticmethod
    def _inherit(groups: dict, t: float, scorer: FrameScorer) -> None:
        """A new hypothesis on an already-verified object starts from that verification."""
        for members in groups.values():
            done = [m for m in members if m.vlm_p is not None]
            if not done:
                continue
            ref = max(done, key=lambda m: m.last_sent_t or -1)
            for m in members:
                if m.vlm_p is None:
                    m.vlm_p, m.vlm_type = ref.vlm_p, ref.vlm_type
                    m.times_sent, m.last_sent_t = 1, ref.last_sent_t
                    m.last_sent_priority = ref.last_sent_priority
                    st = scorer.cands.get(m.cand_id)
                    if st is not None:
                        st.p_vlm, st.category = ref.vlm_p, ref.vlm_type

    @staticmethod
    def _expand(idx: list, vals: list, n: int) -> np.ndarray:
        outv = np.zeros(max(n, 0), np.float32)
        if not idx:
            return outv
        idx = np.asarray(idx)
        vals = np.asarray(vals, np.float32)
        pos = np.searchsorted(idx, np.arange(len(outv)), side="right") - 1
        outv[:] = vals[np.clip(pos, 0, len(vals) - 1)]
        return outv

    def _cost_fn(self, memory: StreamingMemory):
        unit = memory.keyframe_tokens + 3 * memory.crop_tokens

        def cost(c):
            n = min(len(c.track_ids), 3)
            return (memory.keyframe_tokens + n * self.mem_cfg.crops_per_slot * memory.crop_tokens) / unit
        return cost

    @staticmethod
    def _boxes(cand, fo, scene) -> list:
        if cand.box is not None and not cand.track_ids:
            return [tuple(float(v) for v in cand.box)]
        cur = {t.global_id: t.tlwh for t in fo.tracks}
        boxes = []
        for gid in cand.track_ids[:3]:
            if gid in cur:
                boxes.append(tuple(float(v) for v in cur[gid]))
            elif gid in scene.tracks:
                p = scene.tracks[gid].points[-1]
                boxes.append((p[1] - p[3] / 2, p[2] - p[4] / 2, p[3], p[4]))
        return boxes

    def _answer(self, probe: dict, t: float, scorer: FrameScorer, scene, fo) -> dict:
        """A2Seek-format answer for one probe frame, using only information up to now."""
        belief, st = scorer.top_hypothesis(t)
        out = {k: probe[k] for k in ("sample_id", "g_id", "v_id", "f_id") if k in probe}
        if st is None or belief < self.sc_cfg.probe_threshold:
            cat_name, box = "Normal", [0, 0, 0, 0]
        else:
            cid = st.category if st.category not in (None, "normal") else st.hypothesis
            cat_name = self.cards[cid].name if cid in self.cards else "Normal"
            box = self._probe_box(st, scene, fo, probe.get("image_wh"))
        out["prediction"] = f"<answer>{[cat_name]}; {[box]}</answer>"
        out["belief"] = round(float(belief), 4)
        out["category"] = cat_name
        return out

    @staticmethod
    def _probe_box(st, scene, fo, image_wh) -> list:
        boxes = []
        cur = {t.global_id: t.tlwh for t in fo.tracks} if fo is not None else {}
        for gid in st.track_ids[:3]:
            if gid in cur:
                boxes.append(cur[gid])
            elif gid in scene.tracks:
                p = scene.tracks[gid].points[-1]
                boxes.append((p[1] - p[3] / 2, p[2] - p[4] / 2, p[3], p[4]))
        if not boxes and st.box is not None:
            boxes = [st.box]
        if not boxes:
            return [0, 0, 0, 0]
        x0 = min(b[0] for b in boxes)
        y0 = min(b[1] for b in boxes)
        x1 = max(b[0] + b[2] for b in boxes)
        y1 = max(b[1] + b[3] for b in boxes)
        sx = sy = 1.0
        if image_wh and scene.image_hw[1] > 1:
            sx, sy = image_wh[0] / scene.image_hw[1], image_wh[1] / scene.image_hw[0]
        return [int(round(x0 * sx)), int(round(y0 * sy)), int(round(x1 * sx)), int(round(y1 * sy))]

    def _draw(self, writer, out, frame, fo, objgen, scorer, s, t, fps, ledger, scene):
        vis = frame.copy()
        lw = max(1, frame.shape[1] // 900)
        for trk in fo.tracks:
            x, y, w, h = (int(v) for v in trk.tlwh)
            cv2.rectangle(vis, (x, y), (x + w, y + h), (180, 180, 180), lw)
        for c in objgen.open.values():
            st = scorer.cands.get(c.cand_id)
            b = st.belief(t, scorer.cfg) if st else 0.0
            col = (0, 0, 255) if (st and st.p_vlm is not None and st.p_vlm >= 0.5) else \
                ((0, 165, 255) if b >= 0.3 else (0, 255, 255))
            for bx in self._boxes(c, fo, scene):
                x, y, w, h = (int(v) for v in bx)
                cv2.rectangle(vis, (x - 3, y - 3), (x + w + 3, y + h + 3), col, 2 * lw)
                label = (st.category if st and st.category else c.type_hint)[:16]
                cv2.putText(vis, f"{label} {b:.2f}", (x, max(y - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5 * lw, col, lw)
        H, W = vis.shape[:2]
        bar_h = 40 * lw
        cv2.rectangle(vis, (0, 0), (W, bar_h), (0, 0, 0), -1)
        cv2.rectangle(vis, (0, 0), (int(W * s), 8 * lw), (0, 0, 255) if s >= 0.5 else (0, 200, 255), -1)
        cv2.putText(vis, f"t={t:5.1f}s  anomaly={s:.2f}  objects={len(fo.tracks)}  VLM calls={ledger.vlm_calls}",
                    (10, 30 * lw), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * lw, (255, 255, 255), 2 * lw)
        if writer is None:
            sc = min(1.0, 1280 / W)
            self._vsize = (int(W * sc) // 2 * 2, int(H * sc) // 2 * 2)
            writer = cv2.VideoWriter(str(out / "vad.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, self._vsize)
        writer.write(cv2.resize(vis, self._vsize, interpolation=cv2.INTER_AREA))
        return writer

