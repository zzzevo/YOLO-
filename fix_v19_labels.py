# -*- coding: utf-8 -*-
"""修正 v19（这 4 段视频用户确认都是排球）：
   1) 所有框强制为排球(class 1) —— 因为模型/CLIP 对"红橙拼块排球"会争执，而视频内容已知
   2) 剔除"不像球"的框（地板/白线误检）：框内饱和色占比 + 圆形填充率 + 长宽比
   3) 顺带用同一判据统计剔除了多少（供核对）
"""
import sys
from pathlib import Path

sys.path.insert(0, ".")
import cv2
import numpy as np

SRC = Path("raw_v19")
LAB = Path("v19_labels/labels")


def ball_like(img, box):
    x1, y1, x2, y2 = [int(max(0, v)) for v in box]
    crop = img[y1:y2, x1:x2]
    if crop.size < 400:
        return False, "太小"
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    sat = ((hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 60)).astype(np.uint8)
    frac = sat.sum() / sat.size
    if frac < 0.15:
        return False, f"饱和色太少({frac:.2f})"          # 木地板/白线/背景
    k = np.ones((5, 5), np.uint8)
    sat = cv2.morphologyEx(sat * 255, cv2.MORPH_CLOSE, k)
    n, lab_, stats, _ = cv2.connectedComponentsWithStats(sat, 8)
    if n <= 1:
        return False, "无连通块"
    big = 1 + int(np.argmax(stats[1:, 4]))
    x, y, bw, bh, area = stats[big]
    if area < 0.25 * crop.shape[0] * crop.shape[1]:
        return False, "最大色块太小"
    fill = area / (bw * bh + 1e-6)
    ar = bw / (bh + 1e-6)
    if fill < 0.5:
        return False, f"不圆({fill:.2f})"
    if not (0.5 < ar < 2.0):
        return False, f"长宽比怪({ar:.2f})"
    return True, f"ok(fill={fill:.2f},sat={frac:.2f})"


kept = dropped = 0
reasons = {}
for lab in sorted(LAB.glob("*.txt")):
    img = cv2.imread(str(SRC / (lab.stem + ".jpg")))
    if img is None:
        continue
    h, w = img.shape[:2]
    out = []
    for l in lab.read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        c, cx, cy, bw, bh = l.split()
        box = ((float(cx) - float(bw) / 2) * w, (float(cy) - float(bh) / 2) * h,
               (float(cx) + float(bw) / 2) * w, (float(cy) + float(bh) / 2) * h)
        ok, why = ball_like(img, box)
        if not ok:
            dropped += 1
            key = why.split("(")[0]
            reasons[key] = reasons.get(key, 0) + 1
            continue
        out.append(f"1 {cx} {cy} {bw} {bh}")          # 强制排球
        kept += 1
    lab.write_text("\n".join(out), encoding="utf-8")

print(f"保留 {kept} 个框（全部标为排球）| 剔除 {dropped} 个不像球的框")
print("剔除原因:", dict(sorted(reasons.items(), key=lambda x: -x[1])))
frames_with = sum(1 for l in LAB.glob("*.txt") if l.read_text(encoding="utf-8").strip())
print(f"有标注的帧: {frames_with} / {len(list(LAB.glob('*.txt')))}")
