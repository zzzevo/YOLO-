# 证明: 模型对"大球"失明是尺度问题 —— 把失败帧整体缩小(等于让球变小)再喂给模型
import cv2
import numpy as np

from infer_onnx import Detector, letterbox

det = Detector("best.onnx", 640, 0.25)

CASES = {
    "v2__001845 (球很大, 原分0.006)": "raw_ball/v2__001845.jpg",
    "v1__000360 (球大, 原分0.171)": "raw_ball/v1__000360.jpg",
    "v1__000345 (球中等, 原分0.521 对照)": "raw_ball/v1__000345.jpg",
}


def score_of(img):
    blob_img, r, dx, dy = letterbox(img, 640)
    blob = cv2.cvtColor(blob_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[None])
    out = det.sess.run([det.out_name], {det.inp.name: blob})[0][0].T
    conf = out[:, 4:].max(1)
    i = int(conf.argmax())
    # 换算回原图尺寸的框大小
    cx, cy, w, h = out[i, :4]
    frac = float(w * h) / (640 * 640) * 100
    return float(conf[i]), frac


print(f"{'帧':38s} {'缩放':>6s} {'最高分':>8s} {'框占画面':>9s}")
print("-" * 68)
for name, path in CASES.items():
    img = cv2.imread(path)
    h, w = img.shape[:2]
    for k in (1.0, 0.7, 0.5, 0.35):
        if k == 1.0:
            test = img
        else:
            # 把整帧缩小后贴回同样大小的灰底画布 = 等价于"把球变小、画面变空"
            small = cv2.resize(img, (int(w * k), int(h * k)))
            canvas = np.full((h, w, 3), 114, np.uint8)
            y = (h - small.shape[0]) // 2
            x = (w - small.shape[1]) // 2
            canvas[y:y + small.shape[0], x:x + small.shape[1]] = small
            test = canvas
        s, frac = score_of(test)
        flag = "  <-- 检出了" if s >= 0.25 else ""
        print(f"{name:38s} {k:6.2f} {s:8.3f} {frac:8.1f}%{flag}")
    print()
