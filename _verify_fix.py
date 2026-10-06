# 验证: 重训后模型能否在"原始尺度"(单尺度, scale=1.0)下看见之前失明的大球
import glob
from pathlib import Path

import cv2
import numpy as np

from infer_onnx import CLASS_NAMES, Detector

OLD = "best.onnx"                              # 你上传的原始权重
NEW = "runs/ball_v2/weights/best.onnx"         # 重训后
if not Path(NEW).exists():
    NEW = "runs/ball_v2/weights/best.pt"

# 8 张"尺度盲区"帧 + 2 张正常帧做对照
BLIND = ["v1__000360", "v1__001845", "v1__001860", "v2__000300",
         "v2__001845", "v2__001860", "v2__001890", "v2__002640"]
NORMAL = ["v1__000345", "v2__001095"]

old = Detector(OLD, 640, 0.25)
new = Detector(NEW, 640, 0.25) if NEW.endswith(".onnx") else None
pt = None
if new is None:
    from train import install_single_process_pool
    install_single_process_pool()
    from ultralytics import YOLO
    pt = YOLO(NEW)


def run(det, img):
    """单尺度(scale=1.0)检测, 返回 (最高分, 框)"""
    d, _ = det(img)
    if not d:
        return 0.0, None
    best = max(d, key=lambda x: x["conf"])
    return best["conf"], best


def run_pt(model, path):
    r = model.predict(path, conf=0.05, imgsz=640, verbose=False)[0]
    if not len(r.boxes):
        return 0.0, None
    i = int(r.boxes.conf.argmax())
    return float(r.boxes.conf[i]), {"xyxy": [float(v) for v in r.boxes.xyxy[i]], "cls": int(r.boxes.cls[i])}


print(f"{'帧':14s} {'旧模型':>8s} {'新模型':>8s}   新模型框/类别")
print("-" * 74)
rows, worse = [], 0
for stem in BLIND + NORMAL:
    f = f"raw_ball/{stem}.jpg"
    if not Path(f).exists():
        continue
    img = cv2.imread(f)
    s_old, _ = run(old, img)
    if new is not None:
        s_new, b = run(new, img)
    else:
        s_new, b = run_pt(pt, f)
    tag = ""
    if stem in BLIND:
        tag = " <-- 原来失明" if s_old < 0.25 else ""
        if s_new >= 0.25:
            tag += " 已修复"
    info = ""
    if b is not None:
        x1, y1, x2, y2 = b["xyxy"]
        w = (x2 - x1) / img.shape[1] * 100
        info = f"{CLASS_NAMES[b['cls']]} {w:.0f}%宽"
    print(f"{stem:14s} {s_old:8.3f} {s_new:8.3f}   {info}{tag}")
    rows.append((stem, s_old, s_new))

ok = sum(1 for s, o, n in rows if o < 0.25 and n >= 0.25)
print(f"\n原先失明的 {len(BLIND)} 帧中, 新模型单尺度检出 {ok} 帧")

# 拼图: 新模型在这些帧上的预测
if new is not None:
    from infer_onnx import draw
    tiles = []
    for stem in BLIND:
        f = f"raw_ball/{stem}.jpg"
        if not Path(f).exists():
            continue
        img = cv2.imread(f)
        d, _ = new(img)
        vis = draw(img.copy(), d)
        h, w = vis.shape[:2]
        s = 300 / max(h, w)
        vis = cv2.resize(vis, (int(w * s), int(h * s)))
        canvas = np.full((300, 300, 3), 25, np.uint8)
        y = (300 - vis.shape[0]) // 2
        x = (300 - vis.shape[1]) // 2
        canvas[y:y + vis.shape[0], x:x + vis.shape[1]] = vis
        cv2.putText(canvas, stem, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{len(d)} det", (4, 292), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 0) if d else (0, 0, 255), 1, cv2.LINE_AA)
        tiles.append(canvas)
    while len(tiles) % 4:
        tiles.append(np.full((300, 300, 3), 25, np.uint8))
    rows_img = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
    sheet = np.vstack(rows_img)
    cv2.imwrite("fix_scale_blindness.jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("已写出 fix_scale_blindness.jpg")
