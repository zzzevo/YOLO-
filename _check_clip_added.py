# -*- coding: utf-8 -*-
"""抽查 CLIP 过滤后加进数据集的网图标签（画框 + 出图）。"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")
import cv2
import numpy as np

root = Path("dataset_ball_v8")
imgs = sorted((root / "images" / "train").glob("web_*.jpg"))
imgs = [p for p in imgs if not p.name.startswith("web_bb_")]     # 只看本轮新加的
c = Counter()
for p in imgs:
    for l in (root / "labels" / "train" / (p.stem + ".txt")).read_text(encoding="utf-8").strip().splitlines():
        c["篮球" if l.split()[0] == "0" else "排球"] += 1
print(f"本轮新加网图: {len(imgs)} 张 | 框: {dict(c)}")

idx = np.linspace(0, len(imgs) - 1, min(12, len(imgs))).astype(int)
tiles = []
for i in idx:
    p = imgs[i]
    img = cv2.imread(str(p))
    h, w = img.shape[:2]
    lab = (root / "labels" / "train" / (p.stem + ".txt")).read_text(encoding="utf-8").strip().splitlines()
    vis = img.copy()
    for l in lab:
        cc, cx, cy, bw, bh = l.split()
        cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
        col = (0, 200, 0) if cc == "0" else (0, 128, 255)
        cv2.rectangle(vis, (int(cx - bw / 2), int(cy - bh / 2)),
                      (int(cx + bw / 2), int(cy + bh / 2)), col, 3)
    s = 300 / max(h, w)
    vis = cv2.resize(vis, (int(w * s), int(h * s)))
    c2 = np.full((300, 300, 3), 25, np.uint8)
    c2[(300 - vis.shape[0]) // 2:(300 - vis.shape[0]) // 2 + vis.shape[0],
       (300 - vis.shape[1]) // 2:(300 - vis.shape[1]) // 2 + vis.shape[1]] = vis
    tags = ",".join("篮" if l.split()[0] == "0" else "排" for l in lab)
    cv2.putText(c2, tags[:12], (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    tiles.append(c2)
while len(tiles) % 4:
    tiles.append(np.full((300, 300, 3), 25, np.uint8))
cv2.imwrite("clip_added_check.jpg",
            np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]),
            [cv2.IMWRITE_JPEG_QUALITY, 88])
print("已写出 clip_added_check.jpg")
print("数据集 train:",
      len(list((root / "images" / "train").glob("*.jpg"))),
      "| val:", len(list((root / "images" / "val").glob("*.jpg"))))
