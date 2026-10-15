# -*- coding: utf-8 -*-
"""真正的"地板误检"测试：从帧里裁**纯地板小块**（与任何标注框零重叠），看会不会检出球。

之前用"整帧无检出"当负样本的做法不成立——这个场地里几乎每帧都有球（远处/边缘），
所谓"误检"其实都是真球。裁纯地板小块才是干净的测法。
"""
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")
from train import install_single_process_pool

install_single_process_pool()

import cv2
from ultralytics import YOLO

MODELS = [("v14", "runs/ball_v14/weights/best.pt"),
          ("v16", "runs/ball_v16/weights/best.pt"),
          ("v18", "runs/ball_v18/weights/best.pt")]
ROOT = Path("dataset_ball_v18")
PATCH = 320
N_PER_FRAME = 3
random.seed(11)

cases = []          # (大图路径, 地板小块的绝对坐标)
imgs = sorted((ROOT / "images" / "val").glob("*.jpg"))
for p in imgs:
    lab = ROOT / "labels" / "val" / (p.stem + ".txt")
    if not lab.exists():
        continue
    img = cv2.imread(str(p))
    if img is None:
        continue
    h, w = img.shape[:2]
    boxes = []
    for l in lab.read_text(encoding="utf-8").splitlines():
        if l.strip():
            c, cx, cy, bw, bh = l.split()
            boxes.append((float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h))
    tries = 0
    picked = 0
    while picked < N_PER_FRAME and tries < 25:
        tries += 1
        x = random.randint(0, max(0, w - PATCH))
        y = random.randint(0, max(0, h - PATCH))
        ok = True
        for cx, cy, bw, bh in boxes:                     # 与任何标注框留 30px 余量
            bx1, by1, bx2, by2 = cx - bw / 2 - 30, cy - bh / 2 - 30, cx + bw / 2 + 30, cy + bh / 2 + 30
            if not (x + PATCH < bx1 or x > bx2 or y + PATCH < by1 or y > by2):
                ok = False
                break
        if ok:
            cases.append((p, x, y))
            picked += 1
print(f"纯地板小块样本: {len(cases)} 个（来自 {len(set(c[0] for c in cases))} 帧，每块 {PATCH}x{PATCH}，"
      f"与所有标注框至少隔 30px）")

cache = {}
for tag, mp in MODELS:
    if not Path(mp).exists():
        continue
    m = YOLO(mp)
    fp_boxes = 0
    fp_cases = 0
    confs = []
    for p, x, y in cases:
        if p not in cache:
            cache[p] = cv2.imread(str(p))
        patch = cache[p][y:y + PATCH, x:x + PATCH]
        r = m.predict(patch, conf=0.25, iou=0.45, imgsz=640, device=0, verbose=False)[0]
        n = [b for b in r.boxes if int(b.cls[0]) in (0, 1)]
        if n:
            fp_cases += 1
            fp_boxes += len(n)
            confs += [float(b.conf[0]) for b in n]
    rate = fp_boxes / max(1, len(cases))
    print(f"{tag}: 纯地板块 {len(cases)} 个 -> 误检 {fp_boxes} 个框（涉及 {fp_cases} 块），"
          f"平均每块 {rate*100:.2f}% 概率出误检"
          + (f"，误检置信度 {min(confs):.2f}~{max(confs):.2f}" if confs else ""))
