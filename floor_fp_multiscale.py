# -*- coding: utf-8 -*-
"""治"地板被框成一大块排球"的效果验证：
   在**多种尺度**的纯地板小块上测误检（320/448/576/640），对比修前修后。
   大尺度（576/640）正是"一大块"问题的对应尺度。
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, ".")
from train import install_single_process_pool

install_single_process_pool()

import cv2
import numpy as np
from ultralytics import YOLO

MODELS = [("ball_2c（修前）", "runs/ball_2c/weights/best.pt"),
          ("ball_2c_v2（修后）", "runs/ball_2c_v2/weights/best.pt")]
SIZES = [320, 448, 576, 640]
SRC = Path("_seq_frames")
EXTRA = [Path("raw_v19"), Path("raw_v20"), Path("raw_v21"), Path("raw_v17")]
random.seed(21)

# 候选帧（含连续帧，保证地板纹理多样）
pool = sorted(SRC.glob("*.jpg"))
for d in EXTRA:
    pool += sorted(d.glob("*.jpg"))[:200]
random.shuffle(pool)

# 先算每帧的球框（用现有标签，避免把球裁进来）；用低饱和保证是地板
LABDIRS = [Path("v19_labels/labels"), Path("v20_labels/labels"), Path("v21_labels/labels"),
           Path("v17_labels/labels"), Path("dataset_ball_2c/labels/val"),
           Path("dataset_ball_2c/labels/train")]
cache_boxes = {}


def boxes_of(p, w, h):
    if p.stem in cache_boxes:
        return cache_boxes[p.stem]
    out = []
    for d in LABDIRS:
        f = d / (p.stem + ".txt")
        if f.exists():
            for l in f.read_text(encoding="utf-8").splitlines():
                if l.strip():
                    c, cx, cy, bw, bh = l.split()
                    out.append(((float(cx) - float(bw) / 2) * w, (float(cy) - float(bh) / 2) * h,
                                (float(cx) + float(bw) / 2) * w, (float(cy) + float(bh) / 2) * h))
            break
    cache_boxes[p.stem] = out
    return out


cases = {s: [] for s in SIZES}
for p in pool:
    img = cv2.imread(str(p))
    if img is None:
        continue
    h, w = img.shape[:2]
    bs = boxes_of(p, w, h)
    for size in SIZES:
        if size > w or size > h:
            continue
        got = 0
        for _ in range(8):
            if got >= 2:
                break
            x = random.randint(0, w - size)
            y = random.randint(0, h - size)
            ok = all((x + size < bx1 - 20 or x > bx2 + 20 or y + size < by1 - 20 or y > by2 + 20)
                     for bx1, by1, bx2, by2 in bs)
            if not ok:
                continue
            crop = img[y:y + size, x:x + size]
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            sat = (((hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 60)).sum()) / (size * size)
            if sat > 0.10:
                continue
            cases[size].append(crop)
            got += 1

print("各地板尺度样本数:", {k: len(v) for k, v in cases.items()})
print(f"\n{'模型':18s} " + " ".join(f"{s:>12d}px" for s in SIZES) + "   合计")
print("-" * 72)
for tag, mp in MODELS:
    if not Path(mp).exists():
        print(f"{tag:18s} （还没训好）")
        continue
    m = YOLO(mp)
    row, tot_fp, tot_n = [], 0, 0
    for size in SIZES:
        fp = 0
        for crop in cases[size]:
            r = m.predict(crop, conf=0.25, iou=0.45, imgsz=640, device=0, verbose=False)[0]
            if any(int(b.cls[0]) in (0, 1) for b in r.boxes):
                fp += 1
        n = len(cases[size])
        tot_fp += fp
        tot_n += n
        row.append(f"{fp:4d}/{n:<3d}({fp/max(1,n)*100:4.1f}%)")
    print(f"{tag:18s} " + " ".join(f"{c:>12s}" for c in row) +
          f"   {tot_fp}/{tot_n} ({tot_fp/max(1,tot_n)*100:.1f}%)")
