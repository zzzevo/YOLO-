# -*- coding: utf-8 -*-
"""
给小目标（远距离的细长定位柱）用的**分块放大推理**（SAHI 思路的轻量实现）。

原理：整图缩到 640 时，远处的柱子只剩二三十像素，模型看不见。做法是把图切成
有重叠的若干块，每块**放大**到 640 再检测，最后把框映射回原图并做跨块 NMS。
代价是推理次数变多（4 块 = 4 次），换来小目标召回大幅提升。

和 detect_multiscale 的区别：那个是**缩小**整图（治"大球失明"），这个是**放大局部**（治"小目标失明"）。

用法:
  from infer_onnx import Detector
  from infer_tiled import detect_tiled
  det = Detector("runs/ball_v10/weights/best.onnx", 640, 0.25)
  boxes = detect_tiled(det, img, tiles=2, overlap=0.25)
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from infer_onnx import nms


def _slice_boxes(w, h, tiles, overlap):
    """把图切成 tiles x tiles 块（带重叠）"""
    tw, th = int(w / (tiles - (tiles - 1) * overlap)), int(h / (tiles - (tiles - 1) * overlap))
    tw = min(tw, w)
    th = min(th, h)
    step_x = max(1, int((w - tw) / max(1, tiles - 1))) if tiles > 1 else 0
    step_y = max(1, int((h - th) / max(1, tiles - 1))) if tiles > 1 else 0
    out = []
    for i in range(tiles):
        for j in range(tiles):
            x0 = min(i * step_x, w - tw)
            y0 = min(j * step_y, h - th)
            out.append((x0, y0, x0 + tw, y0 + th))
    return out


def detect_tiled(det, img, tiles=2, overlap=0.25, iou_thr=0.5, min_side=24,
                 upscale_to=640, cls_filter=None):
    """分块放大检测。返回与 Detector 相同格式的列表 [{xyxy, conf, cls}]"""
    h, w = img.shape[:2]
    boxes, scores, classes = [], [], []

    tiles_list = [(0, 0, w, h)] if tiles <= 1 else _slice_boxes(w, h, tiles, overlap)
    for (x0, y0, x1, y1) in tiles_list:
        crop = img[y0:y1, x0:x1]
        ch, cw = crop.shape[:2]
        # 放大到接近 upscale_to（小目标变大）
        scale = min(3.0, upscale_to / max(ch, cw))
        if scale > 1.0:
            crop = cv2.resize(crop, (int(cw * scale), int(ch * scale)),
                              interpolation=cv2.INTER_CUBIC)
        for d in det(crop)[0]:
            if cls_filter is not None and d["cls"] not in cls_filter:
                continue
            bx1, by1, bx2, by2 = d["xyxy"]
            # 映射回原图坐标
            if scale > 1.0:
                bx1, by1, bx2, by2 = bx1 / scale, by1 / scale, bx2 / scale, by2 / scale
            bx1, by1, bx2, by2 = bx1 + x0, by1 + y0, bx2 + x0, by2 + y0
            if (bx2 - bx1) < min_side and (by2 - by1) < min_side:
                continue                      # 太小的框大概是噪声
            boxes.append((bx1, by1, bx2, by2))
            scores.append(d["conf"])
            classes.append(d["cls"])
    if not boxes:
        return []
    out = []
    for c in sorted(set(classes)):
        idx = [i for i, cc in enumerate(classes) if cc == c]
        xywh = [[(boxes[i][0] + boxes[i][2]) / 2, (boxes[i][1] + boxes[i][3]) / 2,
                 boxes[i][2] - boxes[i][0], boxes[i][3] - boxes[i][1]] for i in idx]
        for j in nms(xywh, [scores[i] for i in idx], iou_thr):
            i = idx[j]
            out.append({"xyxy": boxes[i], "conf": scores[i], "cls": c})
    out.sort(key=lambda x: -x["conf"])
    return out
