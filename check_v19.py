# -*- coding: utf-8 -*-
"""检查 v19 标注里可疑的框：最终标成"篮球"的（这批视频按用户说是排球）。出图供人工核对。"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, ".")
import cv2
import numpy as np

SRC = Path("raw_v19")
LAB = Path("v19_labels/labels")
REPORT = Path("v19_labels/labels_report.csv")
TILE = 200
CAP = 20


def make_tile(vis, cap):
    h, w = vis.shape[:2]
    sc = min(TILE / w, (TILE - CAP) / h)
    vis = cv2.resize(vis, (max(1, int(w * sc)), max(1, int(h * sc))))
    canvas = np.full((TILE, TILE, 3), 25, np.uint8)
    y0 = (TILE - CAP - vis.shape[0]) // 2
    x0 = (TILE - vis.shape[1]) // 2
    canvas[y0:y0 + vis.shape[0], x0:x0 + vis.shape[1]] = vis
    cv2.putText(canvas, cap[:26], (3, TILE - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 255, 255), 1)
    return canvas


def write_sheet(tiles, name, cols=6):
    if not tiles:
        print(f"  {name}: 无")
        return
    while len(tiles) % cols:
        tiles.append(np.full((TILE, TILE, 3), 25, np.uint8))
    cv2.imwrite(f"v19_{name}.jpg",
                np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]),
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"  v19_{name}.jpg ({len(tiles)} 个样本)")


rows = list(csv.DictReader(REPORT.open(encoding="utf-8-sig"))) if REPORT.exists() else []
bak = [r for r in rows if r["final"] == "篮球"]
unsure = [r for r in rows if float(r["clip_p"]) < 0.55]
print(f"标成篮球的框 {len(bak)} 个 | CLIP 没把握(<0.55)的框 {len(unsure)} 个")

for tag, group in (("labeled_basketball", bak), ("clip_unsure", unsure)):
    tiles, seen = [], set()
    for r in group:
        if r["file"] in seen:
            continue
        seen.add(r["file"])
        img = cv2.imread(str(SRC / r["file"]))
        if img is None:
            continue
        lab = LAB / (Path(r["file"]).stem + ".txt")
        h, w = img.shape[:2]
        vis = img.copy()
        for l in lab.read_text(encoding="utf-8").splitlines():
            if not l.strip():
                continue
            c, cx, cy, bw, bh = l.split()
            x1, y1 = int((float(cx) - float(bw) / 2) * w), int((float(cy) - float(bh) / 2) * h)
            x2, y2 = int((float(cx) + float(bw) / 2) * w), int((float(cy) + float(bh) / 2) * h)
            col = (0, 200, 0) if c == "0" else (0, 128, 255)
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, 3)
        tiles.append(make_tile(vis, f"{r['file'][:12]} 模{r['model']} C{r['clip']}"))
    write_sheet(tiles[:24], tag)

print("最终类别分布:", {k: sum(1 for r in rows if r["final"] == k) for k in ("篮球", "排球")})
