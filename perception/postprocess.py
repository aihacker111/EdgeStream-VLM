# AERO-MOT 🚁 AGPL-3.0 License
# Copyright (C) 2026 AERO-MOT Authors

"""
YoloFalconPostProcessor  (standalone — Ultralytics YOLO11 NMS)
==============================================================
Hậu xử lý ĐỘC LẬP cho `YoloFalconModel`. Bản này KHÔNG còn kế thừa
`FalconJDEPostProcessor` và KHÔNG dùng `embedding_aware_nms` nữa.

Thay vào đó, phần khử trùng lặp box dùng ĐÚNG CHUẨN NMS của Ultralytics
(`ultralytics.utils.nms.non_max_suppression` — pipeline chuẩn của YOLO11:
xywh->xyxy, lọc theo conf, batched class-aware NMS bằng torchvision với offset
`cls * max_wh`, cắt `max_det`). Nhờ vậy detection của nhánh YOLO hành xử y hệt
Ultralytics YOLO11 gốc, tách bạch hoàn toàn khỏi logic của FalconJDE.

Khác biệt cốt lõi so với FalconJDE vẫn giữ nguyên:
  * FalconJDE có N query cố định -> `pred_reid` per-query có sẵn ở model.
  * YOLO không có query -> per-detection ReID được LẤY MẪU (grid_sample) từ dense
    `emb_map` tại TÂM mỗi box SAU NMS. Đây chính là "emb" analog của YOLO, cùng
    không gian với `.template` mà tracker QAM sample -> nhất quán tuyệt đối.

Output mỗi ảnh (GIỮ NGUYÊN contract cũ để track.py/eval_mot.py không phải sửa):
    {'boxes'(xyxy, orig-scale), 'scores', 'labels',
     'reid'(L2-normed, per-det),
     ['cand_boxes','cand_scores','cand_labels','cand_reid']}

`yolo_out` (eval mode của Ultralytics Detect) = (pred, feats) với
    pred: (B, 4+nc, A)  — box xywh ở PIXEL net-resolution, cls đã sigmoid.

ByteTrack low-stage (dải candidate [cand_floor, conf_thres)) được tạo theo CÁCH
CHUẨN của ByteTrack: chạy 1 lần NMS ở ngưỡng thấp nhất (`cand_floor`) rồi tách
survivors thành dải cao (>= conf_thres) và dải thấp ([cand_floor, conf_thres)).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _load_ultralytics_nms():
    """Import `non_max_suppression` của Ultralytics (hỗ trợ cả layout mới lẫn cũ)."""
    try:
        # Ultralytics >= 8.4: hàm được tách sang module riêng
        from ultralytics.utils.nms import non_max_suppression
    except Exception:  # pragma: no cover - fallback cho bản cũ
        from ultralytics.utils.ops import non_max_suppression
    return non_max_suppression


class YoloFalconPostProcessor(nn.Module):
    def __init__(
        self,
        num_classes:        int,
        num_top_queries:    int   = 300,     # == max_det của Ultralytics
        conf_thres:         float = 0.0,
        nms_iou:            float = 0.6,     # == iou_thres
        class_agnostic:     bool  = False,   # NMS phân biệt class (chuẩn YOLO detect)
        multi_label:        bool  = False,   # best-class-only (chuẩn YOLO val)
        pre_nms_topk:       int   = 3000,    # == max_nms (bound chi phí NMS)
        reid_align_corners: bool  = False,   # khớp grid_sample của reid loss (train)
        letterbox:          bool  = False,   # eval_mot.py dùng PLAIN RESIZE -> False
        **legacy,   # nuốt các kwargs cũ đã bỏ (nms_emb_hi, nms_emb_relax, use_focal_loss)
    ):
        super().__init__()
        self.num_classes        = int(num_classes)
        self.num_top_queries    = int(num_top_queries)
        self.conf_thres         = float(conf_thres)
        self.nms_iou            = float(nms_iou)
        self.class_agnostic     = bool(class_agnostic)
        self.multi_label        = bool(multi_label)
        self.pre_nms_topk       = int(pre_nms_topk)
        self.reid_align_corners = bool(reid_align_corners)
        self.letterbox          = bool(letterbox)

        # đặt từ ngoài (eval/track): net-resolution để decode & scale box về ảnh gốc.
        self._net_hw    = None
        # đặt từ ngoài: >0 = xuất dải candidate cho ByteTrack low-stage.
        self.cand_floor = 0.0
        # cờ export ONNX/TensorRT (bỏ NMS động, xuất top-k cố định).
        self.deploy_mode = False

        self._nms_fn = None  # lazy-load hàm NMS của Ultralytics

    # ------------------------------------------------------------------ config
    def set_net_hw(self, net_h: int, net_w: int):
        self._net_hw = (net_h, net_w)

    def deploy(self):
        """Bật chế độ export (top-k cố định, không NMS động)."""
        self.eval()
        self.deploy_mode = True
        return self

    def _nms(self):
        if self._nms_fn is None:
            self._nms_fn = _load_ultralytics_nms()
        return self._nms_fn

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _raw_pred(yolo_out):
        """Lấy tensor (B, 4+nc, A) từ output của Detect head (eval)."""
        if isinstance(yolo_out, (list, tuple)):
            return yolo_out[0]
        if isinstance(yolo_out, dict):
            b = yolo_out['boxes']
            s = yolo_out['scores']
            return torch.cat([b, s], dim=1)
        return yolo_out

    def _sample_reid(self, emb_map_b, centers_px, net_w, net_h):
        """emb_map_b:(C,H,W); centers_px:(M,2) net-pixels -> (M,C) L2-normed."""
        if emb_map_b is None or centers_px.numel() == 0:
            return None
        gx = 2.0 * centers_px[:, 0] / max(net_w, 1) - 1.0
        gy = 2.0 * centers_px[:, 1] / max(net_h, 1) - 1.0
        grid = torch.stack([gx, gy], dim=-1).view(1, 1, -1, 2)
        samp = F.grid_sample(
            emb_map_b.unsqueeze(0), grid,
            mode='bilinear', align_corners=self.reid_align_corners,
        )                                   # (1,C,1,M)
        feat = samp.squeeze(0).squeeze(1).t()   # (M,C)
        return F.normalize(feat, dim=1, eps=1e-6)

    def _sample_reid_batched(self, emb_map, centers_px, net_w, net_h):
        """emb_map:(B,C,H,W); centers_px:(B,K,2) net-pixels -> (B,K,C) L2-normed.
        Dùng cho deploy (ONNX) — grid_sample chạy batched, không loop."""
        gx = 2.0 * centers_px[..., 0] / max(net_w, 1) - 1.0
        gy = 2.0 * centers_px[..., 1] / max(net_h, 1) - 1.0
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(1)   # (B,1,K,2)
        samp = F.grid_sample(
            emb_map, grid, mode='bilinear', align_corners=self.reid_align_corners,
        )                                                    # (B,C,1,K)
        feat = samp.squeeze(2).permute(0, 2, 1)              # (B,K,C)
        return F.normalize(feat, dim=-1, eps=1e-6)

    def _scale_xyxy(self, xyxy_px, oh, ow, net_h, net_w, lb_pad_b):
        """box xyxy (net-pixels, đầu ra NMS) -> xyxy trên ảnh gốc.

        Mặc định = PLAIN RESIZE (per-axis, không pad) — khớp eval_mot.py:
            x_orig = x_net * ow/net_w ;  y_orig = y_net * oh/net_h
        letterbox=True -> aspect-preserving 1 ratio + padding (gỡ letterbox).
        """
        x1, y1, x2, y2 = xyxy_px.unbind(-1)

        if self.letterbox:
            ratio = min(net_h / oh, net_w / ow)
            if lb_pad_b is not None:
                pad_w = float(lb_pad_b[0]); pad_h = float(lb_pad_b[1])
            else:
                pad_w = (net_w - ow * ratio) * 0.5
                pad_h = (net_h - oh * ratio) * 0.5
            x1 = (x1 - pad_w) / ratio; x2 = (x2 - pad_w) / ratio
            y1 = (y1 - pad_h) / ratio; y2 = (y2 - pad_h) / ratio
        else:
            sx = ow / net_w; sy = oh / net_h
            x1 = x1 * sx; x2 = x2 * sx
            y1 = y1 * sy; y2 = y2 * sy

        return torch.stack([
            x1.clamp(0, ow), y1.clamp(0, oh),
            x2.clamp(0, ow), y2.clamp(0, oh)], dim=-1)

    # ------------------------------------------------------------------ net hw
    def _resolve_net_hw(self, emb_map, stride):
        if self._net_hw is not None:
            return self._net_hw
        if emb_map is not None:
            return emb_map.shape[-2] * stride, emb_map.shape[-1] * stride
        raise RuntimeError(
            "YoloFalconPostProcessor cần net_hw: gọi set_net_hw(net_h, net_w) "
            "hoặc bật use_reid để suy từ emb_map.")

    # ------------------------------------------------------------------ main
    @torch.no_grad()
    def forward(self, outputs: dict, orig_target_sizes: torch.Tensor,
                lb_pad: torch.Tensor = None) -> list:
        pred = self._raw_pred(outputs['yolo_out'])          # (B, 4+nc, A) xywh px + cls
        B = pred.shape[0]

        emb_map = outputs.get('emb_map', None)              # (B,C,H,W) hoặc None
        stride  = int(outputs.get('reid_dense_stride', 4))
        net_h, net_w = self._resolve_net_hw(emb_map, stride)

        if self.deploy_mode:
            return self._forward_deploy(pred, emb_map, orig_target_sizes, net_h, net_w)

        # ByteTrack 2-stage: 1 lần NMS ở ngưỡng thấp nhất rồi tách dải theo score.
        use_cand = 0.0 < self.cand_floor < self.conf_thres
        nms_conf = self.cand_floor if use_cand else max(self.conf_thres, 0.0)

        # Ultralytics NMS ghi đè box xywh->xyxy IN-PLACE -> clone để không hỏng
        # outputs['yolo_out'] (model output có thể được dùng lại, vd reid_dense).
        dets = self._nms()(
            pred.clone(),
            conf_thres = nms_conf,
            iou_thres  = self.nms_iou,
            nc         = self.num_classes,
            agnostic   = self.class_agnostic,
            multi_label= self.multi_label,
            max_det    = self.num_top_queries,
            max_nms    = self.pre_nms_topk,
        )   # list[B] các tensor (n,6): [x1,y1,x2,y2, conf, cls] ở net-pixels

        results = []
        for b in range(B):
            oh = float(orig_target_sizes[b, 0]); ow = float(orig_target_sizes[b, 1])
            lb_b = lb_pad[b] if lb_pad is not None else None
            emap_b = emb_map[b] if emb_map is not None else None
            det = dets[b]

            if det.shape[0] == 0:
                results.append({
                    'labels': det.new_zeros((0,), dtype=torch.long),
                    'boxes':  det.new_zeros((0, 4)),
                    'scores': det.new_zeros((0,))})
                continue

            box_px = det[:, :4]                              # xyxy net px
            sc     = det[:, 4]
            lbl    = det[:, 5].long()

            # tâm box (net px) để lấy mẫu ReID dense — analog của emb per-detection.
            centers = torch.stack([
                (box_px[:, 0] + box_px[:, 2]) * 0.5,
                (box_px[:, 1] + box_px[:, 3]) * 0.5], dim=-1)

            boxes_xyxy = self._scale_xyxy(box_px, oh, ow, net_h, net_w, lb_b)
            reid_b     = self._sample_reid(emap_b, centers, net_w, net_h)

            if use_cand:
                hi = sc >= self.conf_thres
                lo = ~hi
            else:
                hi = torch.ones_like(sc, dtype=torch.bool)
                lo = None

            res = {'labels': lbl[hi], 'boxes': boxes_xyxy[hi], 'scores': sc[hi]}
            if reid_b is not None:
                res['reid'] = reid_b[hi]
            if use_cand and lo is not None and lo.any():
                res['cand_boxes']  = boxes_xyxy[lo]
                res['cand_scores'] = sc[lo]
                res['cand_labels'] = lbl[lo]
                if reid_b is not None:
                    res['cand_reid'] = reid_b[lo]
            results.append(res)

        return results

    # ------------------------------------------------------------------ deploy
    @torch.no_grad()
    def _forward_deploy(self, pred, emb_map, orig_target_sizes, net_h, net_w):
        """Đường xuất ONNX/TensorRT: bỏ NMS động, xuất top-k CỐ ĐỊNH (giống DEIM
        deploy) để đồ thị có shape tĩnh. Trả về tuple:
            (labels, boxes(xyxy, orig-scale), scores, [reid]).
        Lưu ý: YOLO không có reid_conf; chỉ PLAIN RESIZE được hỗ trợ ở deploy.
        """
        p = pred.permute(0, 2, 1)                            # (B, A, 4+nc)
        box_px     = p[..., :4]                              # cx,cy,w,h net px
        scores_all = p[..., 4:4 + self.num_classes]
        sc, lbl = scores_all.max(dim=-1)                     # (B, A)

        K = min(self.num_top_queries, sc.shape[1])
        sc, idx = torch.topk(sc, K, dim=1)                   # (B, K)
        lbl = lbl.gather(1, idx)
        box_sel = box_px.gather(1, idx.unsqueeze(-1).expand(-1, -1, 4))

        cx, cy, w, h = box_sel.unbind(-1)
        x1 = cx - 0.5 * w; y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w; y2 = cy + 0.5 * h

        img_h = orig_target_sizes[:, 0].view(-1, 1)
        img_w = orig_target_sizes[:, 1].view(-1, 1)
        sx = img_w / net_w; sy = img_h / net_h
        boxes = torch.stack([x1 * sx, y1 * sy, x2 * sx, y2 * sy], dim=-1)

        out = (lbl, boxes, sc)
        if emb_map is not None:
            centers = torch.stack([cx, cy], dim=-1)          # (B,K,2) net px
            reid = self._sample_reid_batched(emb_map, centers, net_w, net_h)
            out = out + (reid,)
        return out