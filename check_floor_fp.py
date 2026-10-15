# -*- coding: utf-8 -*-
"""把 v17 在"纯地板帧"上检出的框画出来，人眼确认：是真误检，还是我标负样本时漏掉了球。"""
import sys
from pathlib import Path

sys.path.insert(0, ".")
from train import install_single_process_pool

install_single_process_pool()

import cv2
import numpy as np
from ultralytics import YOLO

m = YOLO("runs/ball_v17/weights/best.pt")
root = Path("dataset_ball_v17")
floor = [p for lab in sorted((root / "labels" / "val").glob("*.txt"))
         if not lab.read_text(encoding="utf-8").strip()
         for p in [root / "images" / "val" / (lab.stem + ".jpg")] if p.exists()]

tiles = []
for p in floor:
    r = m.predict(str(p), conf=0.25, iou=0.45, imgsz=640, device=0, verbose=False)[0]
    boxes = [(int(b.cls[0]), [float(v) for v in b.xyxy[0]], float(b.conf)) for b in r.boxes
             if int(b.cls[0]) in (0, 1)]
    if not boxes:
        continue
    img = cv2.imread(str(p))
    for c, (x1, y1, x2, y2), conf in boxes:
        col = (0, 200, 0) if c == 0 else (0, 128, 255)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), col, 3)
        cv2.putText(img, f"{'篮' if c==0 else '排'} {conf:.2f}", (int(x1), max(18, int(y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    h, w = img.shape[:2]
    s = 300 / max(h, w)
    img = cv2.resize(img, (int(w * s), int(h * s)))
    canvas = np.full((320, 300, 3), 25, np.uint8)
    canvas[:img.shape[0], :img.shape[1]] = img
    cv2.putText(canvas, p.stem[:16], (3, 314), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    tiles.append(canvas)

print(f"有检出的地板帧: {len(tiles)}")
if tiles:
    while len(tiles) % 4:
        tiles.append(np.full((320, 300, 3), 25, np.uint8))
    cv2.imwrite("floor_fp_check.jpg",
                np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]),
                [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("已写出 floor_fp_check.jpg")
