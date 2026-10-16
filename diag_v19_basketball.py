# -*- coding: utf-8 -*-
"""诊断 v19 里"标成篮球"的那些框：是球被误判，还是框在地板上的误检？
输出每个框的 yolo 置信度、CLIP 的 P(球)/P(篮)/P(排)、以及框内主色统计。"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, ".")
import cv2
import numpy as np

SRC = Path("raw_v19")
LAB = Path("v19_labels/labels")
REPORT = Path("v19_labels/labels_report.csv")
rows = list(csv.DictReader(REPORT.open(encoding="utf-8-sig")))

print(f"{'帧':16s} {'模型':4s} {'CLIP':4s} {'clip_p':>7s} {'yolo':>5s} {'框内饱和色占比':>10s} {'主色H':>6s}")
print("-" * 78)
sel = [r for r in rows if r["final"] == "篮球"]
for r in sel[:24]:
    img = cv2.imread(str(SRC / r["file"]))
    lab = LAB / (Path(r["file"]).stem + ".txt")
    if img is None or not lab.exists():
        continue
    h, w = img.shape[:2]
    # 找到该帧里 class 0 的框
    for l in lab.read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        c, cx, cy, bw, bh = l.split()
        if c != "0":
            continue
        x1, y1 = int((float(cx) - float(bw) / 2) * w), int((float(cy) - float(bh) / 2) * h)
        x2, y2 = int((float(cx) + float(bw) / 2) * w), int((float(cy) + float(bh) / 2) * h)
        crop = img[max(0, y1):max(1, y2), max(0, x1):max(1, x2)]
        if crop.size < 100:
            continue
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        sat = (hsv[:, :, 1] > 90) & (hsv[:, :, 2] > 70)
        frac = sat.sum() / sat.size
        hue = int(np.median(hsv[:, :, 0][sat])) if sat.sum() > 30 else -1
        print(f"{r['file'][:16]:16s} {r['model']:4s} {r['clip']:4s} {r['clip_p']:>7s} "
              f"{r['yolo_conf']:>5s} {frac*100:9.1f}% {hue:6d}")
        break
print(f"\n共 {len(sel)} 个（上面最多列 24 个）")
