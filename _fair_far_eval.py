# -*- coding: utf-8 -*-
"""用 v10 从未训练过的远距离帧（val 划分）公平对比 v9 与 v10。"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, ".")
import cv2

from auto_label_pillar import detect_pillar
from infer_onnx import CLASS_NAMES, Detector
from infer_tiled import detect_tiled

root = Path("dataset_ball_v10")
val_pf = sorted((root / "images" / "val").glob("pf_*.jpg"))
print(f"v10 未训练过的远距离帧: {len(val_pf)} 张")

DET = {
    "v9  @640 整图": (Detector("runs/ball_v9/weights/best.onnx", 640, 0.25), False),
    "v9  @960 整图": (Detector("runs/ball_v9/weights/best_960.onnx", 960, 0.25), False),
    "v10 @640 整图": (Detector("runs/ball_v10/weights/best.onnx", 640, 0.25), False),
    "v10 @960 整图": (Detector("runs/ball_v10/weights/best_960.onnx", 960, 0.25), False),
    "v10 @640 分块": (Detector("runs/ball_v10/weights/best.onnx", 640, 0.25), True),
}


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0


gt = {}
for p in val_pf:
    img = cv2.imread(str(p))
    bs = detect_pillar(img)
    if bs:
        h, w = img.shape[:2]
        gt[p.name] = ((bs[0][0] / w, bs[0][1] / h, bs[0][2] / w, bs[0][3] / h), bs[0][3] - bs[0][1])

print(f"有真值 {len(gt)} 张；柱高范围 {min(v[1] for v in gt.values())}~{max(v[1] for v in gt.values())}px")
print(f"\n{'方式':16s} {'召回':>6s} {'平均置信度':>10s}")
print("-" * 36)
for tag, (det, tiled) in DET.items():
    hit = 0
    confs = []
    for p in val_pf:
        if p.name not in gt:
            continue
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        dets = (detect_tiled(det, img, tiles=2, overlap=0.25, cls_filter={2}) if tiled
                else [d for d in det(img)[0] if d["cls"] == 2])
        g, _gh = gt[p.name]
        best = max((iou(g, (d["xyxy"][0] / w, d["xyxy"][1] / h, d["xyxy"][2] / w, d["xyxy"][3] / h)),
                    d["conf"]) for d in dets) if dets else (0, 0)
        hit += int(best[0] >= 0.3)
        if best[0] >= 0.3:
            confs.append(best[1])
    import numpy as np
    print(f"{tag:16s} {hit/max(1,len(gt))*100:5.0f}% {(np.mean(confs) if confs else 0):10.2f}")
