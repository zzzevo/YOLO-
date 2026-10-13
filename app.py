#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLO26 图形控制台（本地网页 GUI）—— 采集 / 自动标注 / 训练 / 推理 一条龙。

启动:
    conda activate yolo26
    python app.py
然后浏览器打开 http://127.0.0.1:7860

设计要点:
  * 训练用「日志文件 + 轮询」而不是管道，既不依赖命名管道（沙箱友好），也不会卡 UI；
  * 自动标注有三种后端:
      1) 自己的 best.onnx（对已训练过的类别最准，推荐）
      2) 开放词汇 YOLO-World（给没见过的类别打标，需要先装 CLIP）
      3) 不标注，只抽帧/整理（交给人工用 X-AnyLabeling 标）
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

CUSTOM_CSS = ""          # 不再需要自定义 CSS（镜像已移除）

BUILD_STAMP = "build " + __import__("time").strftime("%Y-%m-%d %H:%M")

ROOT = Path(__file__).parent.resolve()
PY = sys.executable
LOGS = ROOT / "gui_logs"
LOGS.mkdir(exist_ok=True)

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
os.environ.setdefault("PYTHONPATH", str(ROOT))


# ---------------------------------------------------------------- 训练互斥（后端拦重复/冲突）
# 两道保险：
#   1) 进程内句柄：本次 GUI 启动里发起的训练
#   2) 锁文件 runs/.training.lock：GUI 重启/刷新后仍能认出"已经有训练在跑"
#      （训练是独立子进程，关掉网页甚至关掉 GUI 它还在跑）
_TRAIN = {"proc": None, "log": None, "name": None}
LOCK = ROOT / "runs" / ".training.lock"


def _pid_alive(pid):
    try:
        import psutil
        return psutil.pid_exists(int(pid))
    except Exception:
        try:
            os.kill(int(pid), 0)
            return True
        except Exception:
            return False


def training_in_progress():
    """返回 (是否在跑, 说明文字)"""
    p = _TRAIN.get("proc")
    if p is not None and p.poll() is None:
        return True, f"PID {p.pid}（{_TRAIN.get('name')}）"
    if LOCK.exists():
        try:
            info = json.loads(LOCK.read_text(encoding="utf-8"))
            if _pid_alive(info.get("pid", 0)):
                return True, f"PID {info.get('pid')}（{info.get('name')}，{info.get('start','')} 启动）"
        except Exception:
            pass
        LOCK.unlink(missing_ok=True)          # 陈旧锁，清掉
    return False, ""


def clear_train_lock():
    _TRAIN["proc"] = None
    LOCK.unlink(missing_ok=True)


# ---------------------------------------------------------------- 工具

def _run(cmd, log_name, wait=True):
    """跑子进程，输出写日志文件（不用管道，避免受限环境下的命名管道问题）"""
    log = LOGS / f"{log_name}_{time.strftime('%H%M%S')}.log"
    f = open(log, "w", encoding="utf-8", errors="replace")
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = str(ROOT)
    p = subprocess.Popen(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, env=env)
    if wait:
        p.wait()
        f.close()
        return p.returncode, log.read_text(encoding="utf-8", errors="replace")
    return p, f, log


def _tail(path, n=6000):
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")[-n:]
    except Exception:
        return ""


def _gallery(folder, limit=24):
    folder = Path(folder)
    if not folder.exists():
        return []
    files = sorted([p for p in folder.rglob("*.jpg")] +
                   [p for p in folder.rglob("*.png")])[:limit]
    return [str(p) for p in files]


# ---------------------------------------------------------------- Tab1 采集

def do_extract(video, dst, stride, blur):
    if not video or not Path(video).exists():
        return "❌ 视频路径不存在", []
    cmd = [PY, "extract_frames.py", "--src", video, "--dst", dst,
           "--stride", str(int(stride)), "--blur-thresh", str(float(blur))]
    rc, out = _run(cmd, "extract")
    return out[-4000:], _gallery(dst)


def do_fetch(queries, dst, per_query, min_side):
    if not queries.strip():
        return "❌ 请填关键词（逗号分隔）", []
    qs = [q.strip() for q in queries.replace("，", ",").split(",") if q.strip()]
    cmd = [PY, "fetch_images.py", "--dst", dst, "--per-query", str(int(per_query)),
           "--min-side", str(int(min_side)), "--queries", *qs]
    rc, out = _run(cmd, "fetch")
    out += "\n\n⚠️ 网图有版权、且与你自己摄像头的域差异很大，务必人工筛一遍，最终用实拍数据微调。"
    return out[-4000:], _gallery(dst)


# ---------------------------------------------------------------- Tab2 自动标注

def do_label(src, dst, backend, conf, prompt, val_ratio, cls_mode):
    if not src or not Path(src).exists():
        return "❌ 图片目录不存在", "—"

    if backend.startswith("自己的"):
        cmd = [PY, "build_dataset.py", "--src", src, "--dst", dst, "--conf", str(float(conf)),
               "--val-ratio", str(float(val_ratio))]
        if cls_mode.startswith("只保留"):
            cmd += ["--keep", "1"]          # 1 = 排球；默认保留全部 2 类（篮球 + 排球）
    else:
        cmd = [PY, "auto_label_world.py", "--src", src, "--dst", dst, "--conf", str(float(conf)),
               "--prompt", prompt or "toy car", "--val-ratio", str(float(val_ratio))]

    rc, out = _run(cmd, "label")
    yml = Path(dst) / "data.yaml"
    info = f"data.yaml: {yml}\n" + (yml.read_text(encoding='utf-8') if yml.exists() else "(未生成)")
    info += f"\n\n图片: {len(_gallery(dst, 999))} 张"
    info += f"\n\n下一步: 用 X-AnyLabeling 打开 {dst}\\images 人工校对，然后去『训练』标签页"
    return out[-4000:], info


# ---------------------------------------------------------------- Tab3 训练

def start_train(data, model, epochs, batch, imgsz, name, single, extra, lr0):
    # ---- 后端拦重复：已经有训练在跑，直接拒绝 ----
    busy, who = training_in_progress()
    if busy:
        return (f"❌ 已有训练在跑（{who}），已拦下这次请求。\n"
                f"   同时跑两个训练会互相覆盖 runs/ 里的结果。\n"
                f"   等它结束（按钮会自动恢复），或先停掉那个进程。",
                None, gr.update(interactive=False))
    if not data or not Path(data).exists():
        return "❌ data.yaml 不存在", None, gr.update(interactive=True)
    if not model or not (Path(model).exists() or model.endswith((".pt", ".yaml"))):
        return f"❌ 起始权重不存在: {model}", None, gr.update(interactive=True)
    # ---- 后端拦冲突：同名目录已存在，拒绝（否则 ultralytics 会偷偷建 xxx-2）----
    if (ROOT / "runs" / name).exists():
        return (f"❌ runs/{name} 已存在，名字冲突，已拦下。\n"
                f"   换个「本次训练名」，或先删掉/移走那个目录。",
                None, gr.update(interactive=True))

    cmd = [PY, "train.py", "--data", data, "--model", model, "--epochs", str(int(epochs)),
           "--batch", str(int(batch)), "--imgsz", str(int(imgsz)), "--name", name,
           "--lr0", str(float(lr0)),
           "--project", str(ROOT / "runs"), "--workers", "0"]
    if single:
        cmd.append("--single-process")
    if extra.strip():
        cmd += extra.split()

    p, fh, log = _run(cmd, "train", wait=False)
    _TRAIN.update(proc=p, log=str(log), name=name)
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(json.dumps({"pid": p.pid, "name": name,
                                "start": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False),
                    encoding="utf-8")
    return (f"✅ 已启动训练（PID {p.pid}）\n日志: {log}\n\n"
            f"TensorBoard: http://127.0.0.1:6006/\n（训练期间按钮已置灰，结束自动恢复）",
            str(log), gr.update(interactive=False))


def poll_train(logfile, state):
    """轮询日志文件，把新增内容吐给界面（不用管道）；结束/失败后解锁按钮"""
    if not logfile:
        yield "（还没开始训练）", state, gr.update(interactive=True)
        return
    yield _tail(logfile, 3000), state, gr.update(interactive=False)
    for _ in range(3600):          # 最多盯 2 小时
        time.sleep(2)
        txt = _tail(logfile, 3000)
        done = ("epochs completed" in txt) or ("Traceback" in txt) or ("error:" in txt.lower())
        if done:
            clear_train_lock()
            yield txt, "✅ 结束（按钮已恢复）", gr.update(interactive=True)
            return
        yield txt, state, gr.update(interactive=False)


# ---------------------------------------------------------------- Tab4 推理

_DET = None
_DET_PATH = None


def do_predict(img_path, conf, iou, model_path):
    global _DET, _DET_PATH
    if not img_path:
        return None, "请上传图片"
    model_path = str(model_path or (ROOT / "best.onnx"))
    if not Path(model_path).exists():
        return None, f"❌ 找不到模型文件: {model_path}"

    # .pt 走 ultralytics（训练完可以立刻验），.onnx 走 onnxruntime（部署用的那份）
    if model_path.lower().endswith(".pt"):
        from ultralytics import YOLO
        r = YOLO(model_path).predict(str(img_path), conf=float(conf), iou=float(iou),
                                     imgsz=640, verbose=False)[0]
        vis = r.plot()
        msg = f"模型: {Path(model_path).name} (PyTorch)\n{len(r.boxes)} 个目标\n"
        for b in r.boxes:
            nm = r.names[int(b.cls[0])]
            msg += (f"  {nm}  conf={float(b.conf[0]):.3f}  "
                    f"xyxy={[round(float(v), 1) for v in b.xyxy[0]]}\n")
        return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), msg

    from infer_onnx import Detector, draw, CLASS_NAMES
    if _DET is None or _DET_PATH != model_path:
        _DET = Detector(model_path, 640, float(conf), float(iou))
        _DET_PATH = model_path
    _DET.conf, _DET.iou = float(conf), float(iou)
    img = cv2.imread(str(img_path))
    if img is None:
        return None, "读不到图片"
    dets, dt = _DET(img)
    vis = draw(img.copy(), dets)
    msg = f"模型: {Path(model_path).name} (ONNX)\n{len(dets)} 个目标, {dt:.0f} ms (CPU)\n"
    for d in dets:
        nm = CLASS_NAMES[d["cls"]] if d["cls"] < len(CLASS_NAMES) else d["cls"]
        msg += f"  {nm}  conf={d['conf']:.3f}  xyxy={[round(v,1) for v in d['xyxy']]}\n"
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), msg


# ---------------------------------------------------------------- 手机/电脑摄像头实时识别

_PT = None
_PT_PATH = None
_n = [0]          # 实时帧计数（界面上显示，用来判断服务端有没有收到画面）


LIVELOG = LOGS / "live.log"


def _loglive(text):
    try:
        with open(LIVELOG, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {text}\n")
    except Exception:
        pass


def ping_test():
    """一键判断浏览器<->服务器通道：能返回时间就说明通道是好的。"""
    _loglive("ping from browser")
    return (f"✅ 通道正常：服务器时间 {time.strftime('%H:%M:%S')}\n"
            f"说明浏览器和服务器连得上。若这个正常但摄像头没反应，"
            f"那就是【摄像头流式】这一段没工作——改用下面「上传视频检测」最稳。")


_last_live = [0.0]


def detect_video(video_path, conf, iou, model_path,
                 max_frames=900, progress=gr.Progress()):
    """上传视频 → 逐帧检测 → 输出画框视频。完全不依赖浏览器摄像头，最稳。"""
    if not video_path:
        return None, "先选一个视频文件"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, f"打不开视频: {video_path}"
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = LOGS / "detect_out.mp4"
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    total = min(n, max_frames) if n > 0 else max_frames
    stat, i = {}, 0
    t0 = time.time()
    while i < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        vis, msg = _safe_live(frame, conf, iou, model_path, log=False)
        if vis is not None:
            vw.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        # 统计
        head = msg.split("本帧:")[-1].strip() if "本帧:" in msg else ""
        if head and head != "无目标":
            for part in head.split("、"):
                k = part.split("×")[0]
                stat[k] = stat.get(k, 0) + 1
        i += 1
        if total:
            progress(i / total, desc=f"检测中 {i}/{total}")
    cap.release()
    vw.release()
    dt = time.time() - t0
    txt = (f"✅ 完成 {i} 帧，用时 {dt:.1f}s（{dt/max(i,1)*1000:.0f} ms/帧）\n"
           f"检出统计（按帧计）: " + ("、".join(f"{k} {v} 帧" for k, v in
                                          sorted(stat.items(), key=lambda x: -x[1])) or "无")
           + f"\n结果视频: {out}")
    _loglive(f"detect_video {i} frames {dt:.0f}s stat={stat}")
    return str(out), txt


def _safe_live(frame, conf, iou, model_path, log=True):
    """统一入口：规范通道 + 异常兜底"""
    if frame is None:
        return None, "（没有画面）"
    try:
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        return _live_core(frame, conf, iou, model_path, log=log)
    except Exception as e:
        import traceback
        _loglive("ERROR " + traceback.format_exc()[-500:])
        return frame, f"❌ 服务端出错: {type(e).__name__}: {e}"


def on_cam_change(frame, conf, iou, model_path):
    """摄像头值变化时触发（拍照模式也能用）+ 防抖，避免和流式双触发"""
    now = time.time()
    if now - _last_live[0] < 0.12:
        return gr.update(), gr.update()
    _last_live[0] = now
    return _safe_live(frame, conf, iou, model_path)


def grab_once(frame, conf, iou, model_path):
    """手动抓当前摄像头画面检测一次——流式万一不工作时用这个。"""
    if frame is None:
        return None, "还没有摄像头画面，先点一下摄像头框、允许摄像头"
    try:
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        _loglive(f"grab_once shape={frame.shape}")
        return _live_core(frame, conf, iou, model_path)
    except Exception as e:
        import traceback
        _loglive("ERROR grab_once " + traceback.format_exc()[-500:])
        return frame, f"❌ 出错: {type(e).__name__}: {e}"


def lan_ip():
    """本机在局域网里的 IPv4（手机要用它访问；换了 Wi-Fi 这个就变）"""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def _early_args():
    """在构建界面前就拿到端口/SSL，这样二维码里的地址才是对的。"""
    import argparse
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--browser", action="store_true")
    ap.add_argument("--ssl", action="store_true")
    ap.add_argument("--lan", action="store_true")
    a, _ = ap.parse_known_args()
    if a.lan:
        a.host, a.ssl = "0.0.0.0", True
    return a


ARGS = _early_args()
PHONE_URL = f"https://{lan_ip()}:{ARGS.port}" if ARGS.ssl else f"http://{lan_ip()}:{ARGS.port}"

# 手机访问地址也写到文件里，方便随时查看（IP 会随 Wi-Fi 变，重启本程序即刷新）
try:
    (ROOT / "手机访问地址.txt").write_text(
        f"{PHONE_URL}\n（用手机浏览器打开；不要在微信里打开）\n页面顶部也会显示这个地址和二维码。\n", encoding="utf-8")
except Exception:
    pass

# 二维码：电脑上打开页面 -> 手机扫一下，不用手输 IP
QR_PATH = LOGS / "phone_qr.png"
try:
    import qrcode
    qrcode.make(PHONE_URL).save(QR_PATH)
except Exception:
    QR_PATH = None


def live_detect(frame, conf, iou, model_path):
    """流式摄像头每帧回调"""
    return _safe_live(frame, conf, iou, model_path)


def _live_core(frame, conf, iou, model_path, log=True):
    img = np.ascontiguousarray(frame[:, :, ::-1])          # RGB -> BGR
    global _PT, _PT_PATH, _DET, _DET_PATH

    if str(model_path).lower().endswith(".pt"):
        from ultralytics import YOLO
        if _PT is None or _PT_PATH != str(model_path):
            _PT = YOLO(str(model_path))
            _PT_PATH = str(model_path)
        t0 = time.time()
        r = _PT.predict(img, conf=float(conf), iou=float(iou), imgsz=640,
                        device=0, verbose=False)[0]
        vis = r.plot()
        dt = (time.time() - t0) * 1000
        cnt = {}
        for b in r.boxes:
            nm = r.names[int(b.cls[0])]
            cnt[nm] = cnt.get(nm, 0) + 1
        msg = (f"模型 {Path(model_path).name} (PyTorch/GPU) | {dt:.0f} ms/帧 "
               f"≈ {1000/max(dt,1):.0f} FPS | 本帧: " +
               ("、".join(f"{k}×{v}" for k, v in cnt.items()) if cnt else "无目标"))
        _n[0] += 1
        if log:
            _loglive(f"frame#{_n[0]} {img.shape} GPU {dt:.0f}ms det={len(r.boxes)} conf={conf}")
        msg = f"已收到 {_n[0]} 帧\n" + msg
        return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), msg

    from infer_onnx import Detector, draw, CLASS_NAMES
    if _DET is None or _DET_PATH != str(model_path):
        _DET = Detector(str(model_path), 640, float(conf), float(iou))
        _DET_PATH = str(model_path)
    _DET.conf, _DET.iou = float(conf), float(iou)
    t0 = time.time()
    dets, _ = _DET(img)
    dt = (time.time() - t0) * 1000
    vis = draw(img.copy(), dets)
    cnt = {}
    for d in dets:
        nm = CLASS_NAMES[d["cls"]] if d["cls"] < len(CLASS_NAMES) else str(d["cls"])
        cnt[nm] = cnt.get(nm, 0) + 1
    msg = (f"模型 {Path(model_path).name} (ONNX/CPU) | {dt:.0f} ms/帧 "
           f"≈ {1000/max(dt,1):.0f} FPS | 本帧: " +
           ("、".join(f"{k}×{v}" for k, v in cnt.items()) if cnt else "无目标"))
    _n[0] += 1
    if log:
        _loglive(f"frame#{_n[0]} {img.shape} CPU {dt:.0f}ms det={len(dets)} conf={conf}")
    msg = f"已收到 {_n[0]} 帧\n" + msg
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), msg


# ---------------------------------------------------------------- 界面

with gr.Blocks(title="YOLO26 工作台") as demo:
    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown(f"<div id='buildstamp'>版本 {BUILD_STAMP}</div>"
                        f"### 📱 手机扫这个码，或手动输入 `{PHONE_URL}`"
                        f"\n⚠️ 电脑和手机必须连**同一个** Wi-Fi/热点；换网络后地址会变，"
                        f"重启本程序并刷新页面即可")
        with gr.Column(scale=1):
            if QR_PATH is not None:
                gr.Image(value=str(QR_PATH), label="扫码打开", height=170, container=False)
    gr.Markdown(f"<div id='buildstamp'>版本 {BUILD_STAMP} ｜ 手机访问：https://{lan_ip()}:7870</div>")
    gr.Markdown("# 🎯 YOLO26 工作台\n"
                "采集 → 自动标注 → 人工校对 → 训练 → 推理，全程图形界面。\n\n"
                "**TensorBoard 训练曲线**：<http://127.0.0.1:6006/> 　"
                "**人工校对**：用 X-AnyLabeling 打开数据集目录")

    state = gr.State("")

    with gr.Tab("① 采集图片"):
        gr.Markdown("两条来源：**自己摄像头拍的视频**（强烈推荐，占比要大）和 **联网搜图**（扩类用，域差异大）")
        with gr.Row():
            with gr.Column():
                video = gr.Textbox(label="视频文件或目录", value=str(ROOT / "sample_raw"))
                dst1 = gr.Textbox(label="输出图片目录", value=str(ROOT / "raw_cam"))
                stride = gr.Slider(1, 60, value=5, step=1, label="每 N 帧取一张")
                blur = gr.Slider(0, 200, value=40, step=5, label="糊图阈值（越小收得越多）")
                btn1 = gr.Button("从视频抽帧", variant="primary")
            with gr.Column():
                queries = gr.Textbox(label="联网搜图关键词（逗号分隔）",
                                     value="robocup junior line follower robot, 智能小车 循迹 赛道")
                dst2 = gr.Textbox(label="输出图片目录", value=str(ROOT / "raw_web"))
                perq = gr.Slider(5, 200, value=40, step=5, label="每个关键词最多下载")
                mins = gr.Slider(64, 1024, value=320, step=32, label="短边最小像素")
                btn2 = gr.Button("联网搜图并下载", variant="primary")
        log1 = gr.Textbox(label="运行日志", lines=12)
        gal1 = gr.Gallery(label="已采到的图", columns=8, height=260)
        btn1.click(do_extract, [video, dst1, stride, blur], [log1, gal1])
        btn2.click(do_fetch, [queries, dst2, perq, mins], [log1, gal1])

    with gr.Tab("② 自动标注"):
        gr.Markdown("用模型自动打标生成 YOLO 数据集骨架。**打出来的框必须人工过一遍**。\n\n"
                    "类别已核验：**0 = 篮球，1 = 排球**。两个都要就选「保留全部 2 类」。")
        with gr.Row():
            with gr.Column():
                src2 = gr.Textbox(label="原始图片目录", value=str(ROOT / "raw_cam"))
                dst3 = gr.Textbox(label="数据集目录", value=str(ROOT / "dataset_cam"))
                backend = gr.Radio(["自己的 best.onnx（最准，推荐）", "开放词汇 YOLO-World（新类，需 CLIP）"],
                                   value="自己的 best.onnx（最准，推荐）", label="标注后端")
                prompt = gr.Textbox(label="开放词汇提示词（选 World 时用；写「长相」别写概念，如 toy car）",
                                    value="toy car")
                cls_mode = gr.Radio(["保留全部 2 类（篮球+排球，推荐）", "只保留「排球」（单类）"],
                                    value="保留全部 2 类（篮球+排球，推荐）", label="输出类别")
                conf2 = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="置信度阈值（自动标注宁低勿高）")
                valr = gr.Slider(0.05, 0.5, value=0.2, step=0.05, label="验证集比例")
                btn3 = gr.Button("开始自动标注", variant="primary")
            with gr.Column():
                out2 = gr.Textbox(label="数据集信息", lines=14)
        log2 = gr.Textbox(label="运行日志", lines=10)
        btn3.click(do_label, [src2, dst3, backend, conf2, prompt, valr, cls_mode], [log2, out2])

    with gr.Tab("③ 训练"):
        gr.Markdown("**这一页默认就是「在已有模型基础上继续训练」，不是从零开始。**\n"
                    "- **继续训练（推荐）**：起始权重填上一版的 `best.pt`（例如 `runs/ball_v11/weights/best.pt`），**lr0 必须调小到 0.001**，否则会把已经学到的冲掉；本次训练名换一个新的（例如 ball_v12）。\n"
                    "- **从零开始**：起始权重填官方预训练 `yolo26n.pt`，lr0 用 0.01，并且数据集里要放**全部**数据。\n"
                    "- 拿不准就用继续训练：数据量没到几万张时，继练永远比从零划算。\n"
                    "- 练完导出 ONNX：`python train.py --export-only runs/ball_v12/weights/best.pt`")
        _busy0, _who0 = training_in_progress()
        with gr.Row():
            with gr.Column():
                data3 = gr.Textbox(label="data.yaml", value=str(ROOT / "dataset_ball_v11" / "data.yaml"),
                                   info="dataset_ball_v11=最新(3类+修掉89个错标) / v10=旧 / dataset_ball_v4=2类")
                model3 = gr.Textbox(label="起始权重", value="runs/ball_v11/weights/best.pt",
                                    info="接着上次练=最佳 / yolo26n.pt=官方预训练 / best_recovered.pt=原始权重")
                with gr.Row():
                    epochs = gr.Number(value=80, label="epochs", precision=0)
                    batch = gr.Number(value=16, label="batch", precision=0)
                    imgsz = gr.Number(value=640, label="imgsz", precision=0)
                    lr0 = gr.Number(value=0.001, label="lr0 学习率",
                                    info="接着上次练用 0.001；从官方预训练起手用 0.01")
                name3 = gr.Textbox(label="本次训练名", value="ball_v12")
                single = gr.Checkbox(value=False, label="受限/沙箱环境模式（--single-process）")
                extra = gr.Textbox(label="额外参数（可留空）", value="")
                btn4 = gr.Button("开始训练", variant="primary", interactive=not _busy0)
            with gr.Column():
                tip = gr.Textbox(label="状态", lines=8,
                                 value=(f"⚠️ 检测到已有训练在跑：{_who0}\n按钮已置灰，等它结束会自动恢复。"
                                        if _busy0 else
                                        "显存 8GB 建议：batch=16；OOM 就 8\n"
                                        "接着上次的模型练 → lr0 必须调小(0.001)，否则会把学到的冲掉\n"
                                        "训练曲线去 TensorBoard: http://127.0.0.1:6006/\n"
                                        "⚠️ 训练页不导出 ONNX：练完用命令行 python train.py --export-only <best.pt>"))
                logpath = gr.Textbox(label="日志文件", value="")
        log3 = gr.Textbox(label="训练日志（实时）", lines=22)
        btn4.click(start_train, [data3, model3, epochs, batch, imgsz, name3, single, extra, lr0],
                   [tip, logpath, btn4]).then(poll_train, [logpath, tip], [log3, tip, btn4])

    with gr.Tab("④ 推理验证 / 手机实时测试"):
        gr.Markdown(f"### 📱 手机实时识别（版本 {BUILD_STAMP}）\\n"
                    f"**本机当前手机访问地址**：`https://{lan_ip()}:7870`"
                    f"（用 `手机实时测试.bat` 启动时是 7870；换 Wi-Fi / 热点后 IP 会变，刷新页面看这里）\\n\\n"
                    "**⚠️ 重要：手机请用系统浏览器（Chrome / Safari）打开，不要用微信内置浏览器**"
                    "——微信 WebView 经常会挡住摄像头和实时通道。")
        with gr.Row():
            with gr.Column(scale=1):
                cam = gr.Image(label="① 摄像头（点这里开）", sources=["webcam"],
                               streaming=True, type="numpy", height=300,
                               )
                with gr.Row():
                    ping_btn = gr.Button("先点这个：通道自检", variant="secondary", size="sm")
                    grab_btn = gr.Button("抓一帧检测", variant="secondary", size="sm")
            with gr.Column(scale=1):
                out4 = gr.Image(label="② 识别结果（框画在这里）", height=300)
                live_msg = gr.Textbox(label="③ 实时状态 / 自检结果", lines=4, value="等待画面…")
        with gr.Accordion("④ 上传视频检测（最稳，不依赖浏览器摄像头）", open=True):
            gr.Markdown("手机先**用相机 App 录一段**，再把视频传进来 → 逐帧检测 → 直接给你画好框的视频。"
                        "这条路只用到「上传文件」，几乎不会失败。")
            with gr.Row():
                vid = gr.Video(label="上传视频（手机录的就行）", height=240)
                vid_out = gr.Video(label="识别结果视频", height=240)
            with gr.Row():
                vid_btn = gr.Button("开始逐帧检测", variant="primary")
                vid_msg = gr.Textbox(label="视频检测结果", lines=4)
        with gr.Accordion("模型与参数", open=False):
            model4 = gr.Textbox(label="模型（.onnx 或 .pt）",
                                value=str(ROOT / "runs" / "ball_v11" / "weights" / "best.onnx"),
                                info="换成 .pt 走 GPU 更快：runs\\ball_v11\\weights\\best.pt")
            conf4 = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="置信度")
            iou4 = gr.Slider(0.1, 0.9, value=0.45, step=0.05, label="NMS IoU")
            every = gr.Slider(0.05, 1.0, value=0.2, step=0.05, label="实时刷新间隔(秒)")
        with gr.Accordion("上传图片检测", open=False):
            with gr.Row():
                inp = gr.Image(label="上传图片", type="filepath", height=260)
                with gr.Column():
                    btn5 = gr.Button("检测这张图", variant="primary")
                    msg4 = gr.Textbox(label="检测结果", lines=10)
            out5 = gr.Image(label="检测结果图", height=260)
        btn5.click(do_predict, [inp, conf4, iou4, model4], [out5, msg4])
        ping_btn.click(ping_test, None, [live_msg])
        grab_btn.click(grab_once, [cam, conf4, iou4, model4], [out4, live_msg])
        cam.change(on_cam_change, [cam, conf4, iou4, model4], [out4, live_msg])
        cam.stream(live_detect, [cam, conf4, iou4, model4], [out4, live_msg],
                   stream_every=0.2, time_limit=None, concurrency_limit=1)
        vid_btn.click(detect_video, [vid, conf4, iou4, model4], [vid_out, vid_msg])

    gr.Markdown("---\n**典型流程**：①抽帧/搜图 → ②自动标注 → 用 X-AnyLabeling 人工校对 → ③训练 → ④推理验证 → "
                "`python train.py --export-only <best.pt>` 导出 ONNX")


if __name__ == "__main__":
    import argparse
    import subprocess
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 = 局域网/手机也能访问")
    ap.add_argument("--browser", action="store_true",
                    help="服务就绪后自动打开浏览器（双击 bat 启动时用）")
    ap.add_argument("--ssl", action="store_true",
                    help="用 HTTPS 启动（**手机浏览器开摄像头必须 HTTPS**），首次会自动生成自签证书")
    ap.add_argument("--lan", action="store_true", help="等同 --host 0.0.0.0 --ssl（手机测试一条龙）")
    a = ARGS
    ssl_kw = {}
    if a.ssl:
        certdir = ROOT / "certs"
        certdir.mkdir(exist_ok=True)
        key, crt = certdir / "key.pem", certdir / "cert.pem"
        if not (key.exists() and crt.exists()):
            ip = lan_ip()
            print(f">>> 首次生成自签证书（含 IP {ip} 与 localhost）…")
            # 优先用 Python 的 cryptography；没有就用 openssl
            made = False
            try:
                import datetime
                from cryptography import x509
                from cryptography.hazmat.primitives import hashes, serialization
                from cryptography.hazmat.primitives.asymmetric import rsa
                from cryptography.x509.oid import NameOID
                k = rsa.generate_private_key(public_exponent=65537, key_size=2048)
                name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, ip)])
                san = x509.SubjectAlternativeName([x509.DNSName("localhost"),
                                                   x509.IPAddress(__import__("ipaddress").ip_address(ip)),
                                                   x509.IPAddress(__import__("ipaddress").ip_address("127.0.0.1"))])
                now = datetime.datetime.utcnow()
                cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
                        .public_key(k.public_key()).serial_number(x509.random_serial_number())
                        .not_valid_before(now - datetime.timedelta(days=1))
                        .not_valid_after(now + datetime.timedelta(days=3650))
                        .add_extension(san, critical=False)
                        .sign(k, hashes.SHA256()))
                key.write_bytes(k.private_bytes(serialization.Encoding.PEM,
                                                serialization.PrivateFormat.TraditionalOpenSSL,
                                                serialization.NoEncryption()))
                crt.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
                made = True
            except Exception as e:
                print(f"    cryptography 不可用({type(e).__name__})，改用 openssl")
            if not made:
                import glob as _glob
                import shutil
                ossl = shutil.which("openssl") or r"D:\Anconda\Library\bin\openssl.EXE"
                # 有些环境下系统 OPENSSL_CONF 指向不存在的路径，必须显式 -config
                cnf = next((p for p in _glob.glob(r"D:\Anconda\Library\ssl\openssl.cnf")
                            + _glob.glob(r"D:\Anconda\pkgs\openssl-*\Library\ssl\openssl.cnf")
                            + _glob.glob("/usr/lib/ssl/openssl.cnf")), None)
                cmd = [ossl, "req", "-x509", "-newkey", "rsa:2048", "-nodes"]
                if cnf:
                    cmd += ["-config", cnf]
                cmd += ["-keyout", str(key), "-out", str(crt), "-days", "3650",
                        "-subj", f"/CN={lan_ip()}",
                        "-addext", f"subjectAltName=DNS:localhost,IP:{lan_ip()},IP:127.0.0.1"]
                r = subprocess.run(cmd, capture_output=True, text=True)
                if r.returncode != 0 or not crt.exists():
                    print("    openssl 失败:", (r.stderr or "")[-300:])
                    raise SystemExit("证书生成失败，无法启用 HTTPS")
        ssl_kw = {"ssl_keyfile": str(key), "ssl_certfile": str(crt), "ssl_verify": False}
        print(f"\n>>> 手机访问（同一 Wi-Fi）: https://{lan_ip()}:{a.port}/")
        print(">>> 浏览器会提示证书不安全 —— 点「高级 / 继续访问」即可（自签证书正常现象）")
        print(">>> 打开后点摄像头框 → 允许 → 开始实时识别\n")
    print(f"\n>>> 工作台地址: http://{'127.0.0.1' if a.host == '0.0.0.0' else a.host}:{a.port}/")
    if a.host == "0.0.0.0":
        print(f">>> 局域网地址: http://{lan_ip()}:{a.port}/  （手机直接输这个也能用，但摄像头需要 HTTPS）")
    print(">>> 关闭本窗口 = 停止服务\n")
    # Gradio 6: theme 等移到 launch()；不再支持 show_api
    demo.launch(server_name=a.host, server_port=a.port, inbrowser=a.browser,
                theme=gr.themes.Soft(), show_error=True, css=CUSTOM_CSS, **ssl_kw)
