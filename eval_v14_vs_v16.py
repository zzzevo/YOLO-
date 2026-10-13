# -*- coding: utf-8 -*-
"""v14 vs v16：在**真实**验证帧上按目标像素尺寸分档比较召回（合成数据有没有用的决定性指标）。"""
import sys
from pathlib import Path

sys.path.insert(0, ".")
from ultralytics import YOLO
from train import install_single_process_pool

install_single_process_pool()

import cv2
import numpy as np

ROOT = Path("dataset_ball_v16")
VAL = ROOT / "images" / "val"
LBL = ROOT / "labels" / "val"
BUCKETS = [(0, 32, "<32px"), (32, 64, "32~64"), (64, 128, "64~128"),
           (128, 256, "128~256"), (256, 512, "256~512"), (512, 10 ** 9, ">512")]


def bucket(h):
    for lo, hi, n in BUCKETS:
        if lo <= h < hi:
            return n
    return "?"


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0


frames = []
for p in sorted(VAL.glob("*.jpg")):
    lab = LBL / (p.stem + ".txt")
    if not lab.exists():
        continue
    gts = []
    for l in lab.read_text(encoding="utf-8").splitlines():
        if l.strip():
            c, cx, cy, bw, bh = l.split()
            gts.append((int(float(c)), float(cx), float(cy), float(bw), float(bh)))
    if gts:
        frames.append((p, gts))
print(f"真实验证帧: {len(frames)} 张")

for tag, mp in (("v14", "runs/ball_v14/weights/best.pt"), ("v16", "runs/ball_v16/weights/best.pt")):
    m = YOLO(mp)
    stat = {}
    tot = [0, 0]
    for p, gts in frames:
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        r = m.predict(str(p), conf=0.25, iou=0.45, imgsz=640, device=0, verbose=False)[0]
        preds = [(tuple(float(v) for v in b.xyxyn[0]), int(b.cls[0])) for b in r.boxes]
        for c, cx, cy, bw, bh in gts:
            gh = bh * h
            b = bucket(gh)
            hit = any(iou((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), pb) >= 0.3
                      for pb, pc in preds)
            s = stat.setdefault(b, [0, 0])
            s[1] += 1
            s[0] += int(hit)
            tot[0] += int(hit)
            tot[1] += 1
    print(f"\n【{tag}】")
    for _, _, n in BUCKETS:
        if n in stat and stat[n][1]:
            hit, cnt = stat[n]
            flag = "  <- 短板" if hit / cnt < 0.9 else ""
            print(f"  {n:>8s}: {hit/cnt*100:5.0f}%  ({hit}/{cnt}){flag}")
    if tot[1]:
        print(f"  {'合计':>8s}: {tot[0]/tot[1]*100:5.0f}%  ({tot[0]}/{tot[1]})")
