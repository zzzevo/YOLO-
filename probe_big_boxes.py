# -*- coding: utf-8 -*-
"""在指定视频帧上跑当前模型，把"大框"（画面占比 > 阈值）全部挑出来出图 —— 复现"地板被框成一大块排球"。
用法: python probe_big_boxes.py --src raw_v22 --model runs/ball_2c/weights/best.pt --out probe_v22
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, ".")
import cv2
import numpy as np
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="raw_v22")
ap.add_argument("--model", default="runs/ball_2c/weights/best.pt")
ap.add_argument("--conf", type=float, default=0.25)
ap.add_argument("--min-frac", type=float, default=0.06, help="框面积占画面比例超过它算'大框'")
ap.add_argument("--out", default="probe")
ap.add_argument("--max", type=int, default=24)
args = ap.parse_args()

m = YOLO(args.model)
src = Path(args.src)
rows, tiles = [], []
n_frame = n_det = n_big = 0
for p in sorted(src.glob("*.jpg")):
    img = cv2.imread(str(p))
    if img is None:
        continue
    n_frame += 1
    h, w = img.shape[:2]
    r = m.predict(str(p), conf=args.conf, iou=0.45, imgsz=640, device=0, verbose=False)[0]
    boxes = [(int(b.cls[0]), [float(v) for v in b.xyxy[0]], float(b.conf[0])) for b in r.boxes
             if int(b.cls[0]) in (0, 1)]
    if boxes:
        n_det += 1
    for cid, (x1, y1, x2, y2), cf in boxes:
        frac = ((x2 - x1) * (y2 - y1)) / (w * h)
        if frac >= args.min_frac:
            n_big += 1
            rows.append((p.name, cid, cf, frac))
            if len(tiles) < args.max:
                vis = img.copy()
                cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 128, 255), 4)
                cv2.putText(vis, f"{'篮球' if cid==0 else '排球'} {cf:.2f} {frac*100:.0f}%",
                            (int(x1), max(22, int(y1) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2)
                s = 300 / max(h, w)
                vis = cv2.resize(vis, (int(w * s), int(h * s)))
                canvas = np.full((320, 300, 3), 25, np.uint8)
                canvas[:vis.shape[0], :vis.shape[1]] = vis
                cv2.putText(canvas, p.stem[:18], (3, 314), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
                tiles.append(canvas)

print(f"帧数 {n_frame} | 有检出 {n_det} ({n_det/max(1,n_frame)*100:.1f}%) | "
      f"**大框(>={args.min_frac*100:.0f}%画面) {n_big} 个**")
if rows:
    print("大框清单（前 20）:")
    for name, cid, cf, frac in rows[:20]:
        print(f"   {name}  {'篮球' if cid==0 else '排球'} conf={cf:.2f} 占画面 {frac*100:.1f}%")
if tiles:
    while len(tiles) % 4:
        tiles.append(np.full((320, 300, 3), 25, np.uint8))
    cv2.imwrite(f"{args.out}_bigboxes.jpg",
                np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]),
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"大框样例 -> {args.out}_bigboxes.jpg")
