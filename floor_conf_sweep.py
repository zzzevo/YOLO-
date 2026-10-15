# -*- coding: utf-8 -*-
"""地板误检 vs 置信度阈值：给出"调到多少才不会把木地板认成球"的实用建议。
同时报告在该阈值下"真球召回"掉了多少（拿 ball11+web 验证集算）。
"""
import sys
from pathlib import Path

sys.path.insert(0, ".")
from train import install_single_process_pool

install_single_process_pool()

import cv2
import yaml
from ultralytics import YOLO

MP = sys.argv[1] if len(sys.argv) > 1 else "runs/ball_v17/weights/best.pt"
m = YOLO(MP)
root = Path("dataset_ball_v17")

# 地板帧（验证集里的空标签帧）
floor = [p for lab in sorted((root / "labels" / "val").glob("*.txt"))
         if not lab.read_text(encoding="utf-8").strip()
         for p in [root / "images" / "val" / (lab.stem + ".jpg")] if p.exists()]
# 有球的帧（用于看召回）
pos = []
for lab in sorted((root / "labels" / "val").glob("*.txt")):
    if lab.read_text(encoding="utf-8").strip():
        p = root / "images" / "val" / (lab.stem + ".jpg")
        if p.exists():
            pos.append((p, lab))
print(f"模型 {Path(MP).name} | 地板帧 {len(floor)} | 有球帧 {len(pos)}")
print(f"\n{'conf':>5s} {'地板误检帧':>10s} {'误检框':>7s} {'有球帧检出率':>12s}")
print("-" * 42)
for conf in (0.15, 0.25, 0.35, 0.45, 0.55, 0.65):
    fpf = fpb = 0
    for p in floor:
        r = m.predict(str(p), conf=conf, iou=0.45, imgsz=640, device=0, verbose=False)[0]
        n = sum(1 for b in r.boxes if int(b.cls[0]) in (0, 1))
        if n:
            fpf += 1
            fpb += n
    hit = 0
    for p, lab in pos:
        r = m.predict(str(p), conf=conf, iou=0.45, imgsz=640, device=0, verbose=False)[0]
        if any(int(b.cls[0]) in (0, 1) for b in r.boxes):
            hit += 1
    print(f"{conf:5.2f} {fpf:7d}/{len(floor):<3d} {fpb:7d} {hit/max(1,len(pos))*100:11.1f}%")
