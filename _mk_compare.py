# 生成对比图：同一张真实帧上，纯网图模型 vs 网图+实拍模型
from pathlib import Path

import cv2
import numpy as np
from train import install_single_process_pool

install_single_process_pool()

from ultralytics import YOLO  # noqa: E402

ROOT = Path(__file__).parent.resolve()
img_path = sorted((ROOT / "realtest" / "images" / "val").glob("*.jpg"))[4]  # 挑一张有目标的
img = cv2.imread(str(img_path))
print("测试帧:", img_path.name, img.shape)

panels = []
for tag, w in [("A. 纯网图训练(64张)", ROOT / "runs/exp_web/weights/best.pt"),
               ("B. 网图+实拍(103张)", ROOT / "runs/exp_mix/weights/best.pt"),
               ("C. 纯实拍(39张)", ROOT / "runs/exp_real_only/weights/best.pt")]:
    m = YOLO(str(w))
    r = m.predict(str(img_path), conf=0.25, imgsz=640, verbose=False)[0]
    vis = r.plot()  # 已画好框的 BGR 图
    vis = cv2.resize(vis, (512, 512))
    cv2.putText(vis, tag, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, f"det={len(r.boxes)}", (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    panels.append(vis)
    print(f"{tag}: {len(r.boxes)} 个框")

out = np.hstack(panels)
dst = ROOT / "domain_gap_compare.jpg"
cv2.imwrite(str(dst), out, [cv2.IMWRITE_JPEG_QUALITY, 92])
print("已生成:", dst)
