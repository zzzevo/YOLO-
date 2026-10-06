# 诊断: 为什么有些帧"整帧无检测" —— 是我的管线 bug 还是模型本身的问题
from pathlib import Path

import cv2
import numpy as np

from infer_onnx import Detector, letterbox
from train import install_single_process_pool

install_single_process_pool()
from ultralytics import YOLO  # noqa: E402

CASES = [
    "review/images/v1__000360.jpg",
    "review/images/v2__001845.jpg",
    "review/images/v2__002640.jpg",
    "review/images/v2__002400.jpg",   # 远处小球
]
OK_CASE = "review/images/v1__000015.jpg" if Path("review/images/v1__000015.jpg").exists() else None

det = Detector("best.onnx", 640, 0.25)
pt = YOLO("runs/recovered/best_recovered.pt")

# 找一个能正常检测的对照帧
import glob
ok_frames = []
for f in sorted(glob.glob("review/images/*.jpg")):
    d, _ = det(cv2.imread(f))
    if d:
        ok_frames.append(f)
if ok_frames:
    CASES.append(ok_frames[0])
    print("对照(能检出的帧):", ok_frames[0])

print("\n" + "=" * 78)
for f in CASES:
    if not Path(f).exists():
        continue
    img = cv2.imread(f)
    h, w = img.shape[:2]

    # ---- A) 我的 ONNX 管线: 直接把原始输出拿出来看最高分 ----
    blob_img, r, dx, dy = letterbox(img, 640)
    blob = cv2.cvtColor(blob_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[None])
    out = det.sess.run([det.out_name], {det.inp.name: blob})[0][0].T     # (8400, 6)
    cls_scores = out[:, 4:]
    top = np.sort(cls_scores.max(1))[::-1][:3]
    d_a, _ = det(img)

    # ---- B) ultralytics 直接跑 .pt ----
    r_b = pt.predict(f, conf=0.02, imgsz=640, verbose=False)[0]

    print(f"\n{f}   {w}x{h}")
    print(f"  [A] 我的 onnx 管线 : conf>=0.25 得 {len(d_a)} 框 | 原始最高类别分 top3 = {np.round(top, 3).tolist()}")
    print(f"  [B] ultralytics .pt: conf=0.02 得 {len(r_b.boxes)} 框 "
          f"{[(r_b.names[int(b.cls[0])], round(float(b.conf[0]), 3)) for b in r_b.boxes[:4]]}")
