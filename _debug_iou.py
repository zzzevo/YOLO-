# -*- coding: utf-8 -*-
"""揪出两个评估工具结果不一致的原因：打印同一帧的 GT、检测框和 IoU。"""
import sys
from pathlib import Path

sys.path.insert(0, ".")
import cv2

from auto_label_pillar import detect_pillar
from infer_onnx import Detector

root = Path("dataset_ball_v10")
val_pf = sorted((root / "images" / "val").glob("pf_*.jpg"))
det = Detector("runs/ball_v10/weights/best.onnx", 640, 0.25)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[1]) * (a[3] - a[2] if False else a[3] - a[1]) + \
         (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0


for p in val_pf[:6]:
    img = cv2.imread(str(p))
    h, w = img.shape[:2]
    bs = detect_pillar(img)
    ds = [d for d in det(img)[0] if d["cls"] == 2]
    print(f"\n{p.name}  图 {w}x{h}")
    if bs:
        x1, y1, x2, y2, bf, gf = bs[0]
        print(f"  颜色真值(像素): ({x1},{y1},{x2},{y2})  归一化: "
              f"({x1/w:.3f},{y1/h:.3f},{x2/w:.3f},{y2/h:.3f})")
    else:
        print("  颜色真值: 无")
    for d in ds[:3]:
        bx = d["xyxy"]
        print(f"  模型框(像素): ({bx[0]:.0f},{bx[1]:.0f},{bx[2]:.0f},{bx[3]:.0f}) conf={d['conf']:.2f}")
        if bs:
            gx = (x1 / w, y1 / h, x2 / w, y2 / h)
            dn = (bx[0] / w, bx[1] / h, bx[2] / w, bx[3] / h)
            print(f"     IoU = {iou(gx, dn):.3f}")
    if not ds:
        print("  模型: 无检测")
