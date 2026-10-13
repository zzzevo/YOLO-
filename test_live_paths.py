# -*- coding: utf-8 -*-
"""本地验证手机页的所有路径（不启动服务器）：通道自检 / 抓帧 / change 回调 / 视频逐帧检测。"""
import sys
import time

sys.path.insert(0, ".")
import cv2
import numpy as np

import app

FRAME = cv2.cvtColor(cv2.imread("raw_v7/s1__000000.jpg"), cv2.COLOR_BGR2RGB)   # 模拟摄像头帧
MODEL = "runs/ball_v11/weights/best.onnx"

print("1) 通道自检:", app.ping_test().splitlines()[0])

vis, msg = app.grab_once(FRAME, 0.25, 0.45, MODEL, True)
print("2) 抓帧:", None if vis is None else vis.shape, "|", msg.replace("\n", " | "))

app._last_live[0] = 0.0
vis2, msg2 = app.on_cam_change(FRAME, 0.25, 0.45, MODEL, True)
print("3) change 回调:", None if vis2 is None else vis2.shape, "|", msg2.replace("\n", " | "))

vis3, msg3 = app.on_cam_change(FRAME, 0.25, 0.45, MODEL, True)      # 立刻再来一次应被防抖挡住
print("4) 防抖:", "被挡住(返回 update)" if not isinstance(msg3, str) or msg3 == msg2 else "未挡住")

# 5) 造一个 24 帧的小视频，跑视频逐帧检测
src = "raw_v7/s1__000000.jpg"
frames = []
for i in range(24):
    f = cv2.imread(src)
    f = cv2.warpAffine(f, np.float32([[1, 0, i * 4], [0, 1, 0]]), (f.shape[1], f.shape[0]))
    frames.append(f)
h, w = frames[0].shape[:2]
vw = cv2.VideoWriter("_test_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
for f in frames:
    vw.write(f)
vw.release()
t0 = time.time()
out, txt = app.detect_video("_test_video.mp4", 0.25, 0.45, MODEL, True, max_frames=24)
print(f"5) 视频检测: 输出={out} 用时{time.time()-t0:.1f}s")
print("   ", txt.replace("\n", " | "))
print("6) 镜像检查:", end=" ")
a = app.grab_once(FRAME, 0.25, 0.45, MODEL, False)[0]
b = app.grab_once(FRAME, 0.25, 0.45, MODEL, True)[0]
print("关/开镜像 两张图不同 =", bool(a is not None and b is not None and not np.array_equal(a, b)),
      "| 开镜像确实是翻转 =", bool(a is not None and np.array_equal(b, a[:, ::-1])))
