# -*- coding: utf-8 -*-
"""2 类模型最终评测：① 篮球/排球类间判错率 ② 纯地板小块误检率 ③ 新排球视频召回。
对比 v18(3类) 与 ball_2c(纯2类)。
"""
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")
from train import install_single_process_pool

install_single_process_pool()

import cv2
import numpy as np
from ultralytics import YOLO

MODELS = [("v18(3类)", "runs/ball_v18/weights/best.pt"),
          ("ball_2c(纯2类)", "runs/ball_2c/weights/best.pt")]
ROOT = Path("dataset_ball_2c")
PATCH = 320
random.seed(11)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0


# 分类测试集：新排球视频 val 帧 + 网图 val 帧
frames = []
for sp, labdir in (("val", ROOT / "labels" / "val"),):
    for lab in sorted(labdir.glob("*.txt")):
        txt = lab.read_text(encoding="utf-8").strip()
        if not txt:
            continue
        p = ROOT / "images" / sp / (lab.stem + ".jpg")
        if not p.exists():
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        gts = []
        for l in txt.splitlines():
            c, cx, cy, bw, bh = l.split()
            gts.append((int(float(c)), (float(cx) - float(bw) / 2, float(cy) - float(bh) / 2,
                                        float(cx) + float(bw) / 2, float(cy) + float(bh) / 2)))
        if gts:
            frames.append((p, gts))
# 再并入网图验证集
web_root = Path("dataset_ball_v14")
if (web_root / "images" / "val").exists():
    for lab in sorted((web_root / "labels" / "val").glob("web*.txt")):
        p = web_root / "images" / "val" / (lab.stem + ".jpg")
        if not p.exists():
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        gts = [(int(float(l.split()[0])),
                (float(l.split()[1]) - float(l.split()[3]) / 2, float(l.split()[2]) - float(l.split()[4]) / 2,
                 float(l.split()[1]) + float(l.split()[3]) / 2, float(l.split()[2]) + float(l.split()[4]) / 2))
               for l in lab.read_text(encoding="utf-8").splitlines() if l.strip() and l.split()[0] in ("0", "1")]
        if gts:
            frames.append((p, gts))
print(f"分类测试集: {len(frames)} 帧, 框 {sum(len(g) for _, g in frames)} 个")

# 地板小块
cases = []
for p, gts in frames[:600]:
    img = cv2.imread(str(p))
    if img is None:
        continue
    h, w = img.shape[:2]
    if w < PATCH or h < PATCH:
        continue
    for _ in range(2):
        x = random.randint(0, w - PATCH)
        y = random.randint(0, h - PATCH)
        ok = True
        for _, (x1, y1, x2, y2) in gts:
            X1, Y1, X2, Y2 = x1 * w - 25, y1 * h - 25, x2 * w + 25, y2 * h + 25
            if not (x + PATCH < X1 or x > X2 or y + PATCH < Y1 or y > Y2):
                ok = False
                break
        if ok:
            cases.append((p, x, y))
print(f"纯地板小块: {len(cases)} 个\n")

print(f"{'模型':16s} {'篮球对':>6s} {'篮→排':>6s} {'排球对':>6s} {'排→篮':>6s} "
      f"{'判错率':>8s} {'地板误检':>9s}")
print("-" * 68)
for tag, mp in MODELS:
    if not Path(mp).exists():
        print(f"{tag:16s} （还没训好）")
        continue
    m = YOLO(mp)
    cm = Counter()
    for p, gts in frames:
        r = m.predict(str(p), conf=0.25, iou=0.45, imgsz=640, device=0, verbose=False)[0]
        preds = [(tuple(float(v) for v in b.xyxyn[0]), int(b.cls[0]))
                 for b in r.boxes if int(b.cls[0]) in (0, 1)]
        for gc, gbox in gts:
            best, bc = 0.0, None
            for pb, pc in preds:
                v = iou(gbox, pb)
                if v > best:
                    best, bc = v, pc
            cm[(gc, bc if (bc is not None and best >= 0.45) else "miss")] += 1
    b_ok, b_bad = cm[(0, 0)], cm[(0, 1)]
    v_ok, v_bad = cm[(1, 1)], cm[(1, 0)]
    matched = b_ok + b_bad + v_ok + v_bad
    wrong = b_bad + v_bad
    fp = 0
    cache = {}
    for p, x, y in cases:
        if p not in cache:
            cache[p] = cv2.imread(str(p))
        r = m.predict(cache[p][y:y + PATCH, x:x + PATCH], conf=0.25, iou=0.45,
                      imgsz=640, device=0, verbose=False)[0]
        if any(int(b.cls[0]) in (0, 1) for b in r.boxes):
            fp += 1
    print(f"{tag:16s} {b_ok:6d} {b_bad:6d} {v_ok:6d} {v_bad:6d} "
          f"{wrong/max(1,matched)*100:7.2f}% {fp:5d}/{len(cases):<3d} ({fp/max(1,len(cases))*100:.2f}%)")
