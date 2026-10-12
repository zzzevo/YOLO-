# -*- coding: utf-8 -*-
"""对比"远距离定位柱"的几种推理方式：整图640 / 整图960 / 分块放大。"""
import glob
import sys
import time

sys.path.insert(0, ".")
import cv2

from auto_label_pillar import detect_pillar
from infer_onnx import Detector
from infer_tiled import detect_tiled

MODEL_640 = sys.argv[1] if len(sys.argv) > 1 else "runs/ball_v9/weights/best.onnx"
MODEL_960 = sys.argv[2] if len(sys.argv) > 2 else "runs/ball_v9/weights/best_960.onnx"

files = sorted(glob.glob("raw_pillar_far/*.jpg"))
hard, easy = [], []
for f in files:
    im = cv2.imread(f)
    cb = detect_pillar(im)
    if not cb:
        continue
    h = cb[0][3] - cb[0][1]
    (hard if h < 250 else easy).append(f)
print(f"远档(柱高<250px) {len(hard)} 帧 | 近档(>=250px) {len(easy)} 帧")
print(f"模型640: {MODEL_640}")
print(f"模型960: {MODEL_960}")

d640 = Detector(MODEL_640, 640, 0.25)
d960 = Detector(MODEL_960, 960, 0.25)


def whole(det, im):
    return [d for d in det(im)[0] if d["cls"] == 2]


def tiled(det, im):
    return detect_tiled(det, im, tiles=2, overlap=0.25, cls_filter={2})


print(f"\n{'方式':24s} {'远档召回':>9s} {'近档召回':>9s} {'ms/帧':>8s}")
print("-" * 56)
for tag, fn in (("整图 640", lambda im: whole(d640, im)),
                ("整图 960", lambda im: whole(d960, im)),
                ("分块 2x2 放大 @640", lambda im: tiled(d640, im))):
    t0 = time.time()
    rh = sum(bool(fn(cv2.imread(f))) for f in hard)
    re_ = sum(bool(fn(cv2.imread(f))) for f in easy)
    dt = (time.time() - t0) / max(1, len(hard) + len(easy))
    print(f"{tag:24s} {rh/max(1,len(hard))*100:8.0f}% {re_/max(1,len(easy))*100:8.0f}% {dt*1000:7.0f}")
