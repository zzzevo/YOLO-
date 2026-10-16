# -*- coding: utf-8 -*-
"""
推理后处理判据：只有"像球"的框才保留 —— 专治"地板被框成一大块排球"。
判据（在框内统计，对尺度不敏感）：
  1) 饱和色占比 >= 阈值（球有颜色；木地板/白线/阴影都低饱和）
  2) 最大饱和连通块的"圆形填充率" >= 阈值（球的色块接近圆/椭圆；地板色块是长条/大片）
同时报告：这道过滤对**真球召回**的影响（不能为了治误检把真球也滤掉）。
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, ".")
from train import install_single_process_pool

install_single_process_pool()

import cv2
import numpy as np
from ultralytics import YOLO


def ball_score(img, box):
    x1, y1, x2, y2 = [int(max(0, v)) for v in box]
    crop = img[y1:y2, x1:x2]
    if crop.size < 200:
        return 0.0, 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    sat = ((hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 60)).astype(np.uint8)
    frac = sat.sum() / sat.size
    sat = cv2.morphologyEx(sat * 255, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(sat, 8)
    if n <= 1:
        return frac, 0.0
    big = 1 + int(np.argmax(stats[1:, 4]))
    x, y, bw, bh, area = stats[big]
    fill = area / (bw * bh + 1e-6)
    return frac, fill


ap = argparse.ArgumentParser()
ap.add_argument("--model", default="runs/ball_2c_v3/weights/best.pt")
ap.add_argument("--sat-thr", type=float, default=0.15)
ap.add_argument("--fill-thr", type=float, default=0.45)
ap.add_argument("--conf", type=float, default=0.25)
args = ap.parse_args()
m = YOLO(args.model)

# ---- 1) 地板小块：误检率 ----
random_state = np.random.RandomState(21)
pool = sorted(Path("_seq_frames").glob("*.jpg"))
for d in ("raw_v19", "raw_v20", "raw_v21", "raw_v22"):
    pool += sorted(Path(d).glob("*.jpg"))[:150]
SIZES = [320, 448, 576, 640]
cases = []
for p in pool[:900]:
    img = cv2.imread(str(p))
    if img is None:
        continue
    h, w = img.shape[:2]
    for size in SIZES:
        if size > w or size > h:
            continue
        x = random_state.randint(0, w - size)
        y = random_state.randint(0, h - size)
        crop = img[y:y + size, x:x + size]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        satf = (((hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 60)).sum()) / (size * size)
        if satf > 0.08:
            continue
        cases.append((size, crop))
print(f"纯地板小块 {len(cases)} 个（含大尺度）")

fp_before = fp_after = 0
per_size = {s: [0, 0, 0] for s in SIZES}      # [n, fp_before, fp_after]
for size, crop in cases:
    r = m.predict(crop, conf=args.conf, iou=0.45, imgsz=640, device=0, verbose=False)[0]
    boxes = [b for b in r.boxes if int(b.cls[0]) in (0, 1)]
    per_size[size][0] += 1
    if boxes:
        fp_before += 1
        per_size[size][1] += 1
        keep = False
        for b in boxes:
            s_, f_ = ball_score(crop, [float(v) for v in b.xyxy[0]])
            if s_ >= args.sat_thr and f_ >= args.fill_thr:
                keep = True
        if keep:
            fp_after += 1
            per_size[size][2] += 1

# ---- 2) 真球帧：召回损失 ----
real = []
for lab in sorted(Path("dataset_ball_2c/labels/val").glob("*.txt")):
    if not lab.read_text(encoding="utf-8").strip() or "__" not in lab.stem:
        continue
    p = Path("dataset_ball_2c/images/val") / (lab.stem + ".jpg")
    if p.exists():
        real.append(p)
real = real[:400]
hit_before = hit_after = tot = 0
for p in real:
    img = cv2.imread(str(p))
    if img is None:
        continue
    r = m.predict(str(p), conf=args.conf, iou=0.45, imgsz=640, device=0, verbose=False)[0]
    boxes = [b for b in r.boxes if int(b.cls[0]) in (0, 1)]
    if not boxes:
        continue
    tot += 1
    hit_before += 1
    if any(ball_score(img, [float(v) for v in b.xyxy[0]])[0] >= args.sat_thr and
           ball_score(img, [float(v) for v in b.xyxy[0]])[1] >= args.fill_thr for b in boxes):
        hit_after += 1

print(f"\n{'尺度':>6s} {'样本':>5s} {'过滤前误检':>10s} {'过滤后误检':>10s}")
for s in SIZES:
    n, b, a = per_size[s]
    if n:
        print(f"{s:>6d} {n:>5d} {b:>6d}({b/n*100:4.1f}%) {a:>6d}({a/n*100:4.1f}%)")
print(f"{'合计':>6s} {len(cases):>5d} {fp_before:>6d}({fp_before/len(cases)*100:4.1f}%) "
      f"{fp_after:>6d}({fp_after/len(cases)*100:4.1f}%)")
print(f"\n真球帧: {tot} 帧有检出 -> 过滤后仍有框 {hit_after} 帧 "
      f"（召回损失 {100*(hit_before-hit_after)/max(1,hit_before):.1f}%）")
