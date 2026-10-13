#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实时识别页（框直接画在手机实时画面上，并平滑跟踪）+ **可调参数并保存覆盖**。

为什么单独写一个（不用 Gradio）：
  Gradio 的摄像头预览是浏览器本地的，服务端画的框只能显示在"另一个图"里，
  做不到"框就画在实时画面上、跟着球走"。

参数分两类（页面上都能调）：
  A. 推理类（服务端立即生效，只能靠它解决"认错/漏检/多框"）
     conf        总置信度门槛      ↑ 更少误检，↓ 更少漏检
     iou         NMS IoU 门槛     ↓ 更狠地合并重叠框（球挨着时调高）
     cls_conf    每类独立门槛      某一类老认错就单独把它调高（页面三根滑条）
     min_px      框最短边像素门槛   过滤远处小噪点造成的误检
  B. 传输/显示类（前端生效，影响"跟不跟手、卡不卡"）
     upload_w    上传帧宽度        大=远处小球更准，小=更流畅
     upload_fps  每秒上传上限
     smooth      框跟随平滑系数    大=更跟手，小=更顺滑
     coast_ms    目标丢失后保留时长  避免框一闪一闪

用法:
  python live_overlay.py                       # 默认 ONNX/CPU，端口 7871，HTTPS
  python live_overlay.py --model runs/ball_v11/weights/best.pt   # 换 GPU
  python live_overlay.py --port 7871 --conf 0.3
保存：页面点「保存并应用」-> 写入 live_params.json（下次启动自动读取，即"覆盖原参数"）
"""
import argparse
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

from infer_onnx import CLASS_NAMES

PARAMS_FILE = ROOT / "live_params.json"
DEFAULTS = {
    "conf": 0.30,          # 总置信度门槛（模型内部按低阈值检出，再用下面的细门槛筛）
    "iou": 0.45,           # NMS IoU
    "cls_conf": {CLASS_NAMES[0]: 0.35, CLASS_NAMES[1]: 0.35, CLASS_NAMES[2]: 0.25},
    "min_px": 0,           # 框最短边（上传帧像素）低于此值丢弃
    "upload_w": 640,       # 前端上传帧宽度
    "upload_fps": 12,      # 前端每秒上传上限
    "smooth": 0.45,        # 框跟随平滑
    "coast_ms": 600,       # 丢失后保留
}
DET_CONF_FLOOR = 0.05      # 模型固定用低阈值检出，细门槛在下面用参数筛

app = FastAPI(title="YOLO 实时识别")
STATE = {"det": None, "model": "", "params": dict(DEFAULTS)}


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def load_params():
    p = dict(DEFAULTS)
    if PARAMS_FILE.exists():
        try:
            saved = json.loads(PARAMS_FILE.read_text(encoding="utf-8"))
            for k, v in saved.items():
                if k == "cls_conf" and isinstance(v, dict):
                    p["cls_conf"] = {**DEFAULTS["cls_conf"], **v}
                elif k in DEFAULTS:
                    p[k] = v
        except Exception as e:
            print("[params] 读取失败, 用默认值:", e)
    return p


def build_detector(model_path, iou):
    """支持 .onnx(CPU) 与 .pt(GPU)；返回 (detect_fn, 描述)"""
    mp = str(model_path)
    if mp.lower().endswith(".pt"):
        from ultralytics import YOLO
        m = YOLO(mp)
        m.predict(np.zeros((64, 64, 3), np.uint8), device=0, verbose=False)     # 预热

        def fn(img, iou_now):
            r = m.predict(img, conf=DET_CONF_FLOOR, iou=iou_now, imgsz=640,
                          device=0, verbose=False)[0]
            out = []
            for b in r.boxes:
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                out.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                            "cid": int(b.cls[0]), "conf": float(b.conf[0])})
            return out
        return fn, f"{Path(mp).name} (PyTorch/GPU)"
    from infer_onnx import Detector
    d = Detector(mp, 640, DET_CONF_FLOOR, iou)
    d(np.zeros((64, 64, 3), np.uint8))                                         # 预热

    def fn(img, iou_now):
        d.iou = iou_now
        dets, _ = d(img)
        return [{"x1": x["xyxy"][0], "y1": x["xyxy"][1], "x2": x["xyxy"][2],
                 "y2": x["xyxy"][3], "cid": int(x["cls"]), "conf": x["conf"]} for x in dets]
    return fn, f"{Path(mp).name} (ONNX/CPU)"


@app.get("/")
async def index():
    return HTMLResponse((ROOT / "live_overlay.html").read_text(encoding="utf-8"))


@app.get("/qr.png")
async def qr():
    p = ROOT / "gui_logs" / "overlay_qr.png"
    return Response(p.read_bytes(), media_type="image/png") if p.exists() else Response(b"")


@app.get("/params")
async def get_params():
    return JSONResponse({"params": STATE["params"], "defaults": DEFAULTS,
                         "classes": CLASS_NAMES, "model": STATE["model"],
                         "saved_file": str(PARAMS_FILE.name)})


@app.get("/params.json")
async def get_params_file():
    """下载参数文件（可带到部署端用）"""
    return Response(json.dumps(STATE["params"], ensure_ascii=False, indent=2),
                    media_type="application/json",
                    headers={"Content-Disposition": "attachment; filename=live_params.json"})


@app.post("/params")
async def set_params(request: Request):
    """保存并立即生效；写入 live_params.json（覆盖原参数，下次启动自动读取）"""
    try:
        new = json.loads((await request.body()).decode("utf-8"))
    except Exception as e:
        return JSONResponse({"ok": False, "msg": f"参数格式错误: {e}"}, status_code=400)
    p = STATE["params"]
    for k in ("conf", "iou", "smooth"):
        if k in new:
            p[k] = max(0.01, min(0.95, float(new[k])))
    for k in ("upload_w", "upload_fps", "coast_ms", "min_px"):
        if k in new:
            p[k] = max(0, int(float(new[k])))
    if "cls_conf" in new and isinstance(new["cls_conf"], dict):
        for cname in CLASS_NAMES:
            if cname in new["cls_conf"]:
                p["cls_conf"][cname] = max(0.01, min(0.95, float(new["cls_conf"][cname])))
    PARAMS_FILE.write_text(json.dumps(p, ensure_ascii=False, indent=2), encoding="utf-8")
    return JSONResponse({"ok": True, "params": p, "msg": f"已保存到 {PARAMS_FILE.name}（下次启动自动生效）"})


@app.post("/params/reset")
async def reset_params():
    STATE["params"] = json.loads(json.dumps(DEFAULTS))
    if PARAMS_FILE.exists():
        PARAMS_FILE.unlink()
    return JSONResponse({"ok": True, "params": STATE["params"], "msg": "已恢复默认并删除 live_params.json"})


@app.post("/detect")
async def detect(request: Request):
    """收一帧 JPEG，回一帧检测结果（JSON）。用 POST 而非 WebSocket：语义一样且不依赖 websockets 库。"""
    data = await request.body()
    t0 = time.time()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"ms": 0, "w": 0, "h": 0, "boxes": []})
    h, w = img.shape[:2]
    p = STATE["params"]
    try:
        boxes = STATE["det"](img, float(p["iou"]))
    except Exception as e:
        print("[detect error]", type(e).__name__, e)
        boxes = []
    # 细筛：总门槛 + 每类门槛 + 最小像素
    keep = []
    for b in boxes:
        cname = CLASS_NAMES[b["cid"]] if b["cid"] < len(CLASS_NAMES) else str(b["cid"])
        thr = max(float(p["conf"]), float(p["cls_conf"].get(cname, 0.0)))
        if b["conf"] < thr:
            continue
        if p["min_px"] and min(b["x2"] - b["x1"], b["y2"] - b["y1"]) < p["min_px"]:
            continue
        # 注意：onnxruntime 给的是 numpy.float32，必须转成 Python 原生类型才能 JSON 序列化
        keep.append({"x1": float(b["x1"]), "y1": float(b["y1"]),
                     "x2": float(b["x2"]), "y2": float(b["y2"]),
                     "cid": int(b["cid"]), "conf": float(b["conf"]), "cls": cname})
    ms = (time.time() - t0) * 1000
    return JSONResponse({"ms": ms, "w": w, "h": h, "model": STATE["model"],
                         "conf": p["conf"], "iou": p["iou"], "boxes": keep})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7871)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--model", default="runs/ball_v11/weights/best.onnx")
    ap.add_argument("--conf", type=float, default=None, help="覆盖保存的总置信度门槛")
    ap.add_argument("--iou", type=float, default=None)
    ap.add_argument("--no-ssl", action="store_true")
    ap.add_argument("--open", action="store_true")
    a = ap.parse_args()

    params = load_params()
    if a.conf is not None:
        params["conf"] = a.conf
    if a.iou is not None:
        params["iou"] = a.iou
    det, desc = build_detector(a.model, params["iou"])
    STATE.update(det=det, model=desc, params=params)
    print(f"[params] conf={params['conf']} iou={params['iou']} "
          f"每类={params['cls_conf']} min_px={params['min_px']} "
          f"(来源: {'live_params.json' if PARAMS_FILE.exists() else '默认值'})")

    url = f"{'http' if a.no_ssl else 'https'}://{lan_ip()}:{a.port}/"
    (ROOT / "手机访问地址.txt").write_text(f"实时识别页：{url}\n（手机浏览器打开；不要在微信里打开）\n",
                                          encoding="utf-8")
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
            cnf = next(iter(glob.glob(r"D:\Anconda\Library\ssl\openssl.cnf")), None)
            cmd = ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes"]
            if cnf:
                cmd += ["-config", cnf]
            cmd += ["-keyout", str(key), "-out", str(crt), "-days", "3650",
                    "-subj", f"/CN={lan_ip()}",
                    "-addext", f"subjectAltName=DNS:localhost,IP:{lan_ip()},IP:127.0.0.1"]
            subprocess.run(cmd, capture_output=True)
        ssl_kw = {"ssl_keyfile": str(key), "ssl_certfile": str(crt)}

    print("\n" + "=" * 64)
    print("  实时识别（框直接画在画面上 + 跟踪 + 下方可调参数）")
    print(f"  模型: {desc}")
    print(f"  手机地址: {url}")
    print("=" * 64 + "\n")
    if a.open:
        import webbrowser
        webbrowser.open(url)

    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning", **ssl_kw)


if __name__ == "__main__":
    main()
