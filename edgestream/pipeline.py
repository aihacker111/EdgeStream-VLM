"""End-to-end streaming pipeline:

frames -> perception -> scene state -> candidates -> priority -> memory -> density gate
       -> prompt -> VLM -> verdicts -> events -> routing
"""
from __future__ import annotations

import json
import statistics
import time
from collections import Counter, deque
from dataclasses import asdict, fields
from pathlib import Path

import cv2
import yaml

from .candidates import MotionCandidateGenerator, MotionRules
from .events import EventAggregator, EventConfig, EventRouter, save_evidence
from .gate import DensityGate, GateConfig
from .harness import BudgetLedger, FrameRingBuffer, JsonlLogger, VideoStream
from .memory import MemoryConfig, StreamingMemory
from .scene import SceneState
from .vlm import PromptBuilder, build_backend, load_cards

TYPE_COLORS = {"counter_flow": (0, 0, 255), "stopped_in_flow": (0, 165, 255), "sudden_stop": (0, 255, 255),
               "speeding": (255, 0, 255), "congestion_forming": (255, 128, 0)}


def _dc(cls, raw: dict | None):
    raw = raw or {}
    names = {f.name for f in fields(cls)}
    return cls(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in raw.items() if k in names})


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(Path(path).resolve().parent.parent)
    return cfg


class Pipeline:
    def __init__(self, cfg: dict, perceptor=None):
        self.cfg = cfg
        root = Path(cfg["_root"])
        self.window_s = float(cfg.get("window_s", 1.0))
        if perceptor is None:
            from perception import Perceptor
            perceptor = Perceptor.from_yaml(str(root / cfg["perception_config"]))
        self.perceptor = perceptor
        self.rules = _dc(MotionRules, cfg.get("candidates"))
        self.gate_cfg = _dc(GateConfig, cfg.get("gate"))
        self.mem_cfg = _dc(MemoryConfig, cfg.get("memory"))
        self.ev_cfg = _dc(EventConfig, cfg.get("events"))
        self.cards = load_cards(root / cfg["vlm"]["cards_dir"])
        self.builder = PromptBuilder(self.cards, overview_width=self.mem_cfg.keyframe_width,
                                     max_new_tokens=int(cfg["vlm"].get("max_new_tokens", 64)))
        self.vlm = build_backend(cfg["vlm"])

    # ------------------------------------------------------------------ run
    def run(self, video: str, out_dir: str, max_frames: int | None = None, save_video: bool = True,
            start_s: float = 0.0) -> dict:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        stream = VideoStream(video, max_frames=max_frames, start_s=start_s)
        fps = stream.fps
        if hasattr(self.perceptor, "cfg"):
            self.perceptor.cfg.tracker.frame_rate = int(round(fps))
            self.perceptor.reset()
        scene = SceneState(fps=fps, **self.cfg.get("scene", {}))
        gen = MotionCandidateGenerator(self.rules)
        gate = DensityGate(self.gate_cfg)
        memory = StreamingMemory(self.mem_cfg)
        agg = EventAggregator(self.ev_cfg)
        log = JsonlLogger(out)
        router = EventRouter(log)
        ledger = BudgetLedger()
        ring = FrameRingBuffer(seconds=30.0, fps=fps)
        boxes_by_t: dict = {}
        box_keep = deque()

        writer = None
        vis_w = int(self.cfg.get("output", {}).get("video_width", 1280))
        next_window = 0.0
        summary_every = float(self.cfg.get("output", {}).get("summary_every_s", 30.0))
        next_summary = summary_every
        perc_ms, n_frames = [], 0
        cand_types, sent_ids = Counter(), set()
        all_cands = {}
        last_selected, banner = [], []
        t_wall0 = time.perf_counter()
        last_frame_out = None

        for pkt in stream:
            t, frame = pkt.t, pkt.image
            fo = self.perceptor.process(frame, timestamp=t)
            last_frame_out = fo
            perc_ms.append(fo.latency_ms.get("total", 0.0))
            n_frames += 1
            scene.update(fo, frame)
            ring.push(t, frame)
            key = round(t, 3)
            boxes_by_t[key] = {trk.global_id: tuple(float(v) for v in trk.tlwh) for trk in fo.tracks}
            box_keep.append(key)
            while len(box_keep) > ring.buf.maxlen:
                boxes_by_t.pop(box_keep.popleft(), None)

            open_cands = []
            if t + 1e-9 >= next_window:
                next_window += self.window_s
                snap = scene.snapshot()
                log.write("scene", snap)
                open_cands = gen.step(scene)
                for c in open_cands:
                    c.priority = gate.priority(c)
                    cand_types[c.type_hint] += 0
                    if c.cand_id not in all_cands:
                        cand_types[c.type_hint] += 1
                    all_cands[c.cand_id] = c
                memory.observe(t, frame, scene, open_cands, gate.last_tau)
                cost = self._cost_fn(memory)
                selected, ginfo = gate.step(t, open_cands, scene.density, scene.density_ref, cost)
                ginfo.update(memory_tokens=memory.used_tokens(), memory_budget=self.mem_cfg.budget_tokens,
                             lambda_t=round(memory.lam, 3), n_slots=len(memory.slots))
                log.write("gate", ginfo)
                for c in open_cands:
                    log.write("candidates", {"t": round(t, 2), **c.to_dict()})
                last_selected = [c.cand_id for c in selected]

                for cand in selected:
                    sent_ids.add(cand.cand_id)
                    ctx = memory.context(cand, scene)
                    boxes = self._boxes(cand, fo, scene)
                    req = self.builder.build(cand, ctx, frame, 1.0, boxes)
                    req.meta["card_ids"] = list(self.cards)
                    v = self.vlm(req)
                    ledger.record(cand.type_hint, v.visual_tokens, v.prompt_tokens, v.completion_tokens,
                                  v.latency_s)
                    t_alert = t + (v.latency_s if self.cfg["vlm"].get("count_latency", True) else 0.0)
                    log.write("vlm_calls", {"t": round(t, 2), "cand_id": cand.cand_id, "type": cand.type_hint,
                                            "priority": round(cand.priority, 3), "tau": ginfo["tau"],
                                            "n_images": req.meta["n_images"], "visual_tokens": req.visual_tokens,
                                            "prompt_text": req.text, "verdict": v.verdict,
                                            "confidence": v.confidence, "parse_ok": v.parse_ok,
                                            "raw": v.raw_text, "latency_s": round(v.latency_s, 3),
                                            "backend": v.backend})
                    if self.cfg.get("output", {}).get("save_prompts", True):
                        self._save_prompt_images(out / "prompts", cand, req)
                    ev, is_new = agg.add(t_alert, cand, v)
                    if ev is not None and is_new:
                        if self.ev_cfg.save_clips:
                            ev.evidence = save_evidence(ev, ring, out / "evidence", fps, self.ev_cfg.clip_pre_s,
                                                        boxes_by_t)
                        ev.evidence["channels"] = router.route(ev)
                        log.write("events", ev.to_dict())
                        banner.append((t, f"EVENT #{ev.event_id} {ev.event_type} ({ev.severity})"))

                if t >= next_summary:
                    next_summary += summary_every
                    log.write("route_analytics", {"kind": "scene_summary", "t": round(t, 2),
                                                  "text": scene.stats_text(summary_every)})

            if save_video:
                vis = self._draw(frame, fo, open_cands or list(gen.open.values()), last_selected, banner, t,
                                 scene, gate, memory, ledger)
                if writer is None:
                    s = vis_w / vis.shape[1]
                    size = (vis_w, int(round(vis.shape[0] * s)) // 2 * 2)
                    writer = cv2.VideoWriter(str(out / "pipeline.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
                writer.write(cv2.resize(vis, size, interpolation=cv2.INTER_AREA))

        if writer is not None:
            writer.release()
        duration = n_frames / fps
        summary = {
            "video": str(video), "frames": n_frames, "fps": fps, "duration_s": round(duration, 2),
            "wall_s": round(time.perf_counter() - t_wall0, 1),
            "perception_ms_median": round(statistics.median(perc_ms), 1) if perc_ms else None,
            "appearance_enabled": bool(getattr(self.perceptor, "appearance", False)),
            "candidates_total": len(all_cands), "candidates_by_type": dict(cand_types),
            "candidates_sent_to_vlm": len(sent_ids),
            "budget": ledger.summary(duration),
            "gate": asdict(self.gate_cfg), "memory": {**asdict(self.mem_cfg),
                                                      "crop_tokens": memory.crop_tokens,
                                                      "keyframe_tokens": memory.keyframe_tokens},
            "vlm_backend": self.vlm.name,
            "events": [e.to_dict() for e in agg.events],
        }
        with open(out / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        log.close()
        return summary

    # ------------------------------------------------------------------ helpers
    def _cost_fn(self, memory: StreamingMemory):
        unit = memory.keyframe_tokens + 3 * memory.crop_tokens

        def cost(c):
            n = min(len(c.track_ids), 3)
            return (memory.keyframe_tokens + n * self.mem_cfg.crops_per_slot * memory.crop_tokens) / unit
        return cost

    @staticmethod
    def _boxes(cand, fo, scene) -> list:
        cur = {t.global_id: t.tlwh for t in fo.tracks}
        boxes = []
        for gid in cand.track_ids[:3]:
            if gid in cur:
                boxes.append(tuple(float(v) for v in cur[gid]))
            elif gid in scene.tracks:
                p = scene.tracks[gid].points[-1]
                boxes.append((p[1] - p[3] / 2, p[2] - p[4] / 2, p[3], p[4]))
        return boxes

    @staticmethod
    def _save_prompt_images(d: Path, cand, req) -> None:
        d.mkdir(parents=True, exist_ok=True)
        imgs = req.images()
        if not imgs:
            return
        h = 224
        tiles = [cv2.resize(im, (max(1, int(im.shape[1] * h / im.shape[0])), h)) for im in imgs]
        sheet = cv2.hconcat(tiles)
        cv2.imwrite(str(d / f"cand_{cand.cand_id:04d}_t{cand.t_last:06.1f}.jpg"), sheet)
        with open(d / f"cand_{cand.cand_id:04d}_t{cand.t_last:06.1f}.txt", "w") as f:
            f.write(req.system + "\n\n" + req.text)

    @staticmethod
    def _draw(frame, fo, cands, selected, banner, t, scene, gate, memory, ledger):
        vis = frame.copy()
        lw = max(1, frame.shape[1] // 900)
        for trk in fo.tracks:
            x, y, w, h = (int(v) for v in trk.tlwh)
            cv2.rectangle(vis, (x, y), (x + w, y + h), (200, 200, 200), lw)
        cur = {trk.global_id: trk.tlwh for trk in fo.tracks}
        for c in cands:
            col = TYPE_COLORS.get(c.type_hint, (0, 255, 0))
            for gid in c.track_ids[:3]:
                if gid not in cur:
                    continue
                x, y, w, h = (int(v) for v in cur[gid])
                th = 3 * lw if c.cand_id in selected else 2 * lw
                cv2.rectangle(vis, (x - 3, y - 3), (x + w + 3, y + h + 3), col, th)
                cv2.putText(vis, f"{c.type_hint[:10]} p={c.priority:.2f}", (x, max(y - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5 * lw, col, lw)
        hud = (f"t={t:5.1f}s  objects={scene.density}  tau={gate.last_tau:.2f}  bucket={gate.bucket.tokens:.2f}"
               f"  mem={memory.used_tokens()}/{memory.cfg.budget_tokens} tok  slots={len(memory.slots)}"
               f"  VLM calls={ledger.vlm_calls}")
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 40 * lw), (0, 0, 0), -1)
        cv2.putText(vis, hud, (10, 28 * lw), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * lw, (255, 255, 255), 2 * lw)
        recent = [b for b in banner if t - b[0] < 3.0]
        for k, (_, txt) in enumerate(recent[-3:]):
            cv2.putText(vis, txt, (10, (80 + 40 * k) * lw), cv2.FONT_HERSHEY_SIMPLEX, 1.0 * lw, (0, 0, 255), 3 * lw)
        return vis
