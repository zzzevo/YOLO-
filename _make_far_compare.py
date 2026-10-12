# -*- coding: utf-8 -*-
"""出图：最远的定位柱帧上，v9 vs v10 并排对比（红=v9没检出，绿=v10检出）。"""
import sys
from pathlib import Path

sys.path.insert(0, ".")
import cv2
import numpy as np

from auto_label_pillar import detect_pillar
from infer_onnx import CLASS_NAMES, Detector, draw

root = Path("dataset_ball_v10")
val = sorted((root / "images" / "val").glob("pf_*.jpg"))
# 按柱子高度排序，取最远的 6 帧
sizes = []
for p in val:
    img = cv2.imread(str(p))
    bs = detect_pillar(img)
    if bs:
        sizes.append((bs[0][3] - bs[0][1], p))
sizes.sort()
picked = [p for _, p in sizes[:6]]

d9 = Detector("runs/ball_v9/weights/best.onnx", 640, 0.25)
d10 = Detector("runs/ball_v10/weights/best.onnx", 640, 0.25)

rows = []
for p in picked:
    img = cv2.imread(str(p))
    h, w = img.shape[:2]
    bs = detect_pillar(img)
    gt = bs[0] if bs else None
    pair = []
    for tag, det in (("v9", d9), ("v10", d10)):
        ds = [d for d in det(img)[0] if d["cls"] == 2]
        vis = draw(img.copy(), ds)
        if gt:
            cv2.rectangle(vis, (gt[0], gt[1]), (gt[2], gt[3]), (255, 255, 0), 2)   # 真值=黄
        # 裁出柱子周围区域（放大看）
        if gt:
            pad = int(max(gt[2] - gt[0], gt[3] - gt[1]) * 1.2)
            cx, cy = (gt[0] + gt[2]) // 2, (gt[1] + gt[3]) // 2
            x1, y1 = max(0, cx - pad), max(0, cy - pad)
            x2, y2 = min(w, cx + pad), min(h, cy + pad)
            vis = vis[y1:y2, x1:x2]
        s = 300 / max(vis.shape[:2])
        vis = cv2.resize(vis, (int(vis.shape[1] * s), int(vis.shape[0] * s)))
        c = np.full((320, 300, 3), 25, np.uint8)
        c[(320 - vis.shape[0]) // 2:(320 - vis.shape[0]) // 2 + vis.shape[0],
          (300 - vis.shape[1]) // 2:(300 - vis.shape[1]) // 2 + vis.shape[1]] = vis
        conf = f"{max((d['conf'] for d in ds), default=0):.2f}"
        cv2.putText(c, f"{tag}  {'检出' if ds else '漏检'} {conf if ds else ''}  柱高{gt[3]-gt[1] if gt else 0}px",
                    (4, 312), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (0, 255, 0) if ds else (0, 0, 255), 1)
        pair.append(c)
    rows.append(np.hstack(pair))

cv2.imwrite("pillar_far_v9_vs_v10.jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
print("已写出 pillar_far_v9_vs_v10.jpg")
