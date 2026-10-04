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
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

ROOT = Path(__file__).parent.resolve()
PY = sys.executable
LOGS = ROOT / "gui_logs"
LOGS.mkdir(exist_ok=True)

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
os.environ.setdefault("PYTHONPATH", str(ROOT))


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
            cmd += ["--keep", "1"]          # 1 = 机器人（class0 是另一场比赛，不用管）
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

def start_train(data, model, epochs, batch, imgsz, name, single, extra):
    if not data or not Path(data).exists():
        return "❌ data.yaml 不存在", None
    cmd = [PY, "train.py", "--data", data, "--model", model, "--epochs", str(int(epochs)),
           "--batch", str(int(batch)), "--imgsz", str(int(imgsz)), "--name", name,
           "--project", str(ROOT / "runs"), "--workers", "0"]
    if single:
        cmd.append("--single-process")
    if extra.strip():
        cmd += extra.split()
    p, fh, log = _run(cmd, "train", wait=False)
    return f"✅ 已启动训练（PID {p.pid}）\n日志: {log}\n\nTensorBoard: http://127.0.0.1:6006/", str(log)


def poll_train(logfile, state):
    """轮询日志文件，把新增内容吐给界面（不用管道）"""
    if not logfile:
        yield "（还没开始训练）", state
        return
    yield _tail(logfile, 3000), state
    for _ in range(1800):          # 最多盯 1 小时
        time.sleep(2)
        txt = _tail(logfile, 3000)
        done = ("epochs completed" in txt) or ("Traceback" in txt) or ("Error" in txt and "error:" in txt)
        yield txt, ("✅ 结束" if done else state)
        if done:
            break


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


# ---------------------------------------------------------------- 界面

with gr.Blocks(title="YOLO26 工作台") as demo:
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
                    "提示：class0 是另一场比赛的类别、你不用管，所以推荐选「只保留机器人（单类）」。")
        with gr.Row():
            with gr.Column():
                src2 = gr.Textbox(label="原始图片目录", value=str(ROOT / "raw_cam"))
                dst3 = gr.Textbox(label="数据集目录", value=str(ROOT / "dataset_cam"))
                backend = gr.Radio(["自己的 best.onnx（最准，推荐）", "开放词汇 YOLO-World（新类，需 CLIP）"],
                                   value="自己的 best.onnx（最准，推荐）", label="标注后端")
                prompt = gr.Textbox(label="开放词汇提示词（选 World 时用；写「长相」别写概念，如 toy car）",
                                    value="toy car")
                cls_mode = gr.Radio(["只保留「机器人」（单类，推荐）", "保留全部 2 类"],
                                    value="只保留「机器人」（单类，推荐）", label="输出类别")
                conf2 = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="置信度阈值（自动标注宁低勿高）")
                valr = gr.Slider(0.05, 0.5, value=0.2, step=0.05, label="验证集比例")
                btn3 = gr.Button("开始自动标注", variant="primary")
            with gr.Column():
                out2 = gr.Textbox(label="数据集信息", lines=14)
        log2 = gr.Textbox(label="运行日志", lines=10)
        btn3.click(do_label, [src2, dst3, backend, conf2, prompt, valr, cls_mode], [log2, out2])

    with gr.Tab("③ 训练"):
        with gr.Row():
            with gr.Column():
                data3 = gr.Textbox(label="data.yaml", value=str(ROOT / "dataset_demo" / "data.yaml"))
                model3 = gr.Textbox(label="起始权重", value="yolo26n.pt",
                                    info="官方预训练 / runs\\recovered\\best_recovered.pt / yolo26n.yaml")
                with gr.Row():
                    epochs = gr.Number(value=100, label="epochs", precision=0)
                    batch = gr.Number(value=16, label="batch", precision=0)
                    imgsz = gr.Number(value=640, label="imgsz", precision=0)
                name3 = gr.Textbox(label="本次训练名", value="yolo26n_v1")
                single = gr.Checkbox(value=False, label="受限/沙箱环境模式（--single-process）")
                extra = gr.Textbox(label="额外参数（可留空）", value="")
                btn4 = gr.Button("开始训练", variant="primary")
            with gr.Column():
                tip = gr.Textbox(label="状态", lines=8,
                                 value="显存 8GB 建议：batch=16；OOM 就 8；数据集小加 --cache ram（默认已开）\n"
                                       "训练曲线去 TensorBoard: http://127.0.0.1:6006/")
                logpath = gr.Textbox(label="日志文件", value="")
        log3 = gr.Textbox(label="训练日志（实时）", lines=22)
        btn4.click(start_train, [data3, model3, epochs, batch, imgsz, name3, single, extra],
                   [tip, logpath]).then(poll_train, [logpath, tip], [log3, tip])

    with gr.Tab("④ 推理验证"):
        gr.Markdown("拿**没参与训练的实拍图**来验。训练完先用 `train.py --export-only` 导出 ONNX，"
                    "再把下面的模型路径指过去就能对比新旧模型。")
        with gr.Row():
            inp = gr.Image(label="上传图片", type="filepath")
            with gr.Column():
                model4 = gr.Textbox(label="模型（ONNX）", value=str(ROOT / "best.onnx"),
                                    info="也可以填 runs\\exp_mix\\weights\\best.onnx 之类")
                conf4 = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="置信度")
                iou4 = gr.Slider(0.1, 0.9, value=0.45, step=0.05, label="NMS IoU")
                btn5 = gr.Button("检测", variant="primary")
                msg4 = gr.Textbox(label="结果", lines=8)
        out4 = gr.Image(label="检测结果")
        btn5.click(do_predict, [inp, conf4, iou4, model4], [out4, msg4])

    gr.Markdown("---\n**典型流程**：①抽帧/搜图 → ②自动标注 → 用 X-AnyLabeling 人工校对 → ③训练 → ④推理验证 → "
                "`python train.py --export-only <best.pt>` 导出 ONNX")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    # Gradio 6: theme 等移到 launch()；不再支持 show_api
    demo.launch(server_name=a.host, server_port=a.port, inbrowser=False,
                theme=gr.themes.Soft(), show_error=True)
