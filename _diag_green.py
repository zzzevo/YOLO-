# 在"绿色篮球"帧上比几种自动标注手段，看哪种能认出来
# (1) v3 单尺度  (2) v3 尺度空间  (3) YOLO-World 文字提示  (4) 传统 HSV 颜色+轮廓
import glob
from pathlib import Path

import cv2
import numpy as np

from infer_onnx import Detector, detect_multiscale

FRAMES = sorted(glob.glob("raw_bb2/*.jpg"))[:6]
det = Detector("runs/ball_v3/weights/best.onnx", 640, 0.25)


def hsv_ball(img):
    """绿色/灰绿色球: 在 HSV 里找低饱和度的青绿色大色块"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # 绿-青范围, 饱和度中等以上, 排除地面(地面是灰色低饱和)
    mask = cv2.inRange(hsv, (30, 40, 40), (100, 255, 230))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    a = cv2.contourArea(c)
    if a < 400:
        return None
    x, y, w, h = cv2.boundingRect(c)
    # 圆度检查: 外接矩形填充率 (球接近 0.78)
    fill = a / (w * h + 1e-6)
    if not (0.5 < fill < 0.95) or not (0.6 < w / (h + 1e-6) < 1.7):
        return None
    return (x, y, x + w, y + h)


print(f"{'帧':22s} {'v3单尺度':>9s} {'v3尺度空间':>11s} {'YOLO-World':>11s} {'HSV颜色':>9s}")
print("-" * 70)
world = None
try:
    from train import install_single_process_pool
    install_single_process_pool()
    from ultralytics import YOLO
    world = YOLO("yolov8s-worldv2.pt")
    world.set_classes(["basketball"])
    print("(YOLO-World 已加载, 提示词 = basketball)")
except Exception as e:
    print(f"(YOLO-World 不可用: {type(e).__name__}: {e})")

for f in FRAMES:
    img = cv2.imread(f)
    a, _ = det(img)
    b = detect_multiscale(det, img, (1.0, 0.6, 0.4))
    w_n = -1
    if world is not None:
        r = world.predict(f, conf=0.05, verbose=False)[0]
        w_n = len(r.boxes)
    hb = hsv_ball(img)
    print(f"{Path(f).name:22s} {len(a):9d} {len(b):11d} {w_n:11d} "
          f"{('有 ' + str(hb)) if hb else '无':>9s}")
