#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实时识别页（框直接画在手机实时画面上，并平滑跟踪）—— 不依赖 Gradio。

为什么单独写一个：
  Gradio 的摄像头预览是浏览器本地的，服务端画的框只能显示在"另一个图"里，
  做不到"框就画在实时画面上、跟着球走"。所以这里用一个自建 WebSocket 服务：
    手机浏览器: 摄像头 -> canvas 抓帧(JPEG) --ws--> 服务器
    服务器:     解码 -> YOLO 推理 -> 返回 JSON 框
    手机浏览器: 收到框 -> 与上一帧按 IoU 匹配 -> 指数平滑 -> 直接画在画面上(带短暂保留)

用法:
  python live_overlay.py                 # 默认 HTTPS + 局域网, 端口 7871
  python live_overlay.py --model runs/ball_v11/weights/best.pt   # 走 GPU 更快
  python live_overlay.py --port 7871 --conf 0.3
"""
import argparse
import asyncio
import socket
import ssl
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response

from infer_onnx import CLASS_NAMES

app = FastAPI(title="YOLO 实时识别")
STATE = {"det": None, "model": "", "conf": 0.3, "iou": 0.45}


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def build_detector(model_path, conf, iou):
    """支持 .onnx(CPU) 与 .pt(GPU)；返回 (detect_fn, 描述)"""
    mp = str(model_path)
    if mp.lower().endswith(".pt"):
        from ultralytics import YOLO
        m = YOLO(mp)
        m.predict(np.zeros((64, 64, 3), np.uint8), device=0, verbose=False)   # 预热

        def fn(img):
            r = m.predict(img, conf=conf, iou=iou, imgsz=640, device=0, verbose=False)[0]
            out = []
            for b in r.boxes:
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                out.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                            "cls": int(b.cls[0]), "conf": float(b.conf[0])})
            return out
        return fn, f"{Path(mp).name} (PyTorch/GPU)"
    from infer_onnx import Detector
    d = Detector(mp, 640, conf, iou)
    d(np.zeros((64, 64, 3), np.uint8))                                        # 预热

    def fn(img):
        dets, _ = d(img)
        return [{"x1": d0["xyxy"][0], "y1": d0["xyxy"][1], "x2": d0["xyxy"][2],
                 "y2": d0["xyxy"][3], "cls": d0["cls"], "conf": d0["conf"]} for d0 in dets]
    return fn, f"{Path(mp).name} (ONNX/CPU)"


@app.get("/")
async def index():
    return HTMLResponse((ROOT / "live_overlay.html").read_text(encoding="utf-8"))


@app.get("/qr.png")
async def qr():
    p = ROOT / "gui_logs" / "overlay_qr.png"
    if p.exists():
        return Response(p.read_bytes(), media_type="image/png")
    return Response(b"", media_type="image/png")


@app.post("/detect")
async def detect(request: Request):
    """收一帧 JPEG，回一帧检测结果（JSON）。

    用 POST 而不是 WebSocket：语义一样（一帧进一框出），但**不依赖 websockets 库**，
    也没有断线重连的问题。前端是"上一帧回来才发下一帧"，天然自我保护、不会积压。
    """
    data = await request.body()
    t0 = time.time()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return Response('{"ms":0,"w":0,"h":0,"boxes":[]}', media_type="application/json")
    h, w = img.shape[:2]
    det = STATE["det"]
    try:
        boxes = det(img)
        for b in boxes:
            b["cid"] = int(b["cls"])
            b["cls"] = CLASS_NAMES[b["cls"]] if b["cls"] < len(CLASS_NAMES) else str(b["cls"])
    except Exception as e:
        boxes = []
        print("[detect error]", type(e).__name__, e)
    ms = (time.time() - t0) * 1000
    return Response(
        '{"ms": %.1f, "w": %d, "h": %d, "model": "%s", "boxes": [%s]}' % (
            ms, w, h, STATE["model"],
            ",".join('{"x1":%.1f,"y1":%.1f,"x2":%.1f,"y2":%.1f,"cls":"%s","cid":%d,"conf":%.3f}'
                     % (b["x1"], b["y1"], b["x2"], b["y2"], b["cls"], b["cid"], b["conf"])
                     for b in boxes)),
        media_type="application/json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7871)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--model", default="runs/ball_v11/weights/best.onnx")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--no-ssl", action="store_true", help="不用 HTTPS（手机开摄像头会失败）")
    ap.add_argument("--open", action="store_true", help="启动后用电脑浏览器打开页面（看二维码）")
    a = ap.parse_args()

    det, desc = build_detector(a.model, a.conf, a.iou)
    STATE.update(det=det, model=desc, conf=a.conf, iou=a.iou)

    url = f"{'http' if a.no_ssl else 'https'}://{lan_ip()}:{a.port}/"
    (ROOT / "手机访问地址.txt").write_text(
        f"实时识别页：{url}\n（手机浏览器打开；不要在微信里打开）\n", encoding="utf-8")
    try:
        import qrcode
        qrcode.make(url).save(ROOT / "gui_logs" / "overlay_qr.png")
    except Exception:
        pass

    ssl_kw = {}
    if not a.no_ssl:
        key, crt = ROOT / "certs" / "key.pem", ROOT / "certs" / "cert.pem"
        if not (key.exists() and crt.exists()):
            key.parent.mkdir(exist_ok=True)
            import glob
            ossl = "openssl"
            cnf = next(iter(glob.glob(r"D:\Anconda\Library\ssl\openssl.cnf")), None)
            cmd = [ossl, "req", "-x509", "-newkey", "rsa:2048", "-nodes"]
            if cnf:
                cmd += ["-config", cnf]
            cmd += ["-keyout", str(key), "-out", str(crt), "-days", "3650",
                    "-subj", f"/CN={lan_ip()}",
                    "-addext", f"subjectAltName=DNS:localhost,IP:{lan_ip()},IP:127.0.0.1"]
            subprocess.run(cmd, capture_output=True)
        ssl_kw = {"ssl_keyfile": str(key), "ssl_certfile": str(crt)}

    print("\n" + "=" * 62)
    print(f"  实时识别（框直接画在画面上 + 跟踪）")
    print(f"  模型: {desc}")
    print(f"  手机地址: {url}")
    print(f"  电脑打开看二维码: {url}")
    print("=" * 62 + "\n")
    if a.open:
        import webbrowser
        webbrowser.open(url)

    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning", **ssl_kw)


if __name__ == "__main__":
    main()
