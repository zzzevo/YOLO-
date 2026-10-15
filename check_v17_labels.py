# -*- coding: utf-8 -*-
"""验证新视频的自动标注：分别出"排球框""篮球框"拼图 + 被丢弃的框（疑似地板误检）。"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, ".")
import cv2
import numpy as np

SRC, LAB = Path("raw_v17"), Path("v17_labels/labels")
OUT = "v17_check"


def sheet(items, name, cols=6, tile=170):
    tiles = []
    for imgp, (x1, y1, x2, y2), tag in items:
        img = cv2.imread(str(imgp))
        if img is None:
            continue
        pad = int(max(x2 - x1, y2 - y1) * 0.3)
        c = img[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
        if c.size == 0:
            continue
        s = tile / max(c.shape[:2])
        c = cv2.resize(c, (max(1, int(c.shape[1] * s)), max(1, int(c.shape[0] * s))))
        cv2.rectangle(c, (2, 2), (c.shape[1] - 3, c.shape[0] - 3), (0, 255, 0), 2)
        cv2.putText(c, tag[:20], (3, c.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 255, 255), 1)
        canvas = np.full((tile, tile, 3), 25, np.uint8)
        canvas[:c.shape[0], :c.shape[1]] = c
        tiles.append(canvas)
    if not tiles:
        print(f"  {name}: 无样本")
        return
    while len(tiles) % cols:
        tiles.append(np.full((tile, tile, 3), 25, np.uint8))
    cv2.imwrite(f"{OUT}_{name}.jpg",
                np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]),
                [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"  {OUT}_{name}.jpg  ({min(len(tiles), 24)} 个样本)")


by_cls = defaultdict(list)
for lab in sorted(LAB.glob("*.txt")):
    imgp = SRC / (lab.stem + ".jpg")
    img = cv2.imread(str(imgp))
    if img is None:
        continue
    h, w = img.shape[:2]
    for l in lab.read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        c, cx, cy, bw, bh = l.split()
        x1, y1 = int((float(cx) - float(bw) / 2) * w), int((float(cy) - float(bh) / 2) * h)
        x2, y2 = int((float(cx) + float(bw) / 2) * w), int((float(cy) + float(bh) / 2) * h)
        by_cls[int(float(c))].append((imgp, (x1, y1, x2, y2), f"{lab.stem[:12]}"))

print("按最终类别抽样：")
for c, name in ((0, "cls0_basketball"), (1, "cls1_volleyball")):
    # 视频1/2/3 各取一些，便于分辨
    items = [it for it in by_cls.get(c, []) if it[0].name.startswith("n1_")][:8]
    items += [it for it in by_cls.get(c, []) if it[0].name.startswith("n2_")][:8]
    items += [it for it in by_cls.get(c, []) if it[0].name.startswith("n3_")][:8]
    sheet(items, name)

# 被 CLIP 判为非球而丢弃的框（看是不是地板）
report = Path("v17_labels/labels_report.csv")
if report.exists():
    rows = list(csv.DictReader(report.open(encoding="utf-8-sig")))
    print(f"\n全部框 {len(rows)} 个；模型和 CLIP 不一致 {sum(1 for r in rows if r['model'] != r['clip'])} 个")
    from collections import Counter
    print("  不一致方向:", dict(Counter(f"{r['model']}→{r['clip']}" for r in rows if r['model'] != r['clip'])))
    print("  最终分布:", dict(Counter(r["final"] for r in rows)))
