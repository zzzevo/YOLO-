#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLO26 ONNX 推理 / 自动标注工具（纯 CPU，无需 torch）

用法:
  python infer_onnx.py 图片.jpg
  python infer_onnx.py 视频.mp4 --stride 5 --out out/frames
  python infer_onnx.py 图片目录/ --save-labels        # 生成 YOLO 格式 txt，用于打标预处理

针对的模型: best.onnx  (Ultralytics YOLO26n, input 1x3x640x640, output 1x6x8400 = 4 box + 2 cls)
"""
import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

# 模型的 2 个类别，顺序必须与原训练一致（已用裁剪图肉眼核验）：
#   0 = 篮球（橙色）  1 = 排球（蓝黄白 Mikasa）
CLASS_NAMES = ["篮球", "排球"]
COLORS = [(0, 200, 0), (0, 128, 255), (255, 128, 0), (0, 0, 255)]


def letterbox(img, new_shape=640, color=114):
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_shape, new_shape, 3), color, dtype=np.uint8)
    top, left = (new_shape - nh) // 2, (new_shape - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, left, top


def nms(boxes_xywh, scores, iou_thr=0.45):
    if len(boxes_xywh) == 0:
        return []
    xywh = np.asarray(boxes_xywh, dtype=np.float32)
    rects = [[float(x - w / 2), float(y - h / 2), float(w), float(h)] for x, y, w, h in xywh]
    idx = cv2.dnn.NMSBoxes(rects, [float(s) for s in scores], 0.0, iou_thr)
    return [] if len(idx) == 0 else np.asarray(idx).reshape(-1).tolist()


class Detector:
    def __init__(self, model_path, imgsz=640, conf=0.25, iou=0.45, threads=0):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads > 0:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0]
        self.out_name = self.sess.get_outputs()[0].name
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        print(f"[model] input={self.inp.name} {self.inp.shape}  output={self.out_name} "
              f"{self.sess.get_outputs()[0].shape}")

    def __call__(self, bgr):
        img, r, dx, dy = letterbox(bgr, self.imgsz)
        blob = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[None])

        t0 = time.perf_counter()
        out = self.sess.run([self.out_name], {self.inp.name: blob})[0]
        dt = (time.perf_counter() - t0) * 1000

        pred = out[0]                      # (6, 8400)
        pred = pred.T                      # (8400, 6) -> 4 box + nc 类分数
        boxes_xywh = pred[:, :4]
        cls_scores = pred[:, 4:]
        cls_ids = cls_scores.argmax(1)
        confs = cls_scores[np.arange(len(cls_scores)), cls_ids]

        keep = confs >= self.conf
        boxes_xywh, confs, cls_ids = boxes_xywh[keep], confs[keep], cls_ids[keep]
        picked = nms(boxes_xywh, confs, self.iou)

        dets = []
        for i in picked:
            x, y, w, h = boxes_xywh[i]
            x = (x - dx) / r
            y = (y - dy) / r
            w, h = w / r, h / r
            dets.append({
                "xyxy": (x - w / 2, y - h / 2, x + w / 2, y + h / 2),
                "conf": float(confs[i]),
                "cls": int(cls_ids[i]),
            })
        return dets, dt


def draw(img, dets):
    for d in dets:
        x1, y1, x2, y2 = [int(v) for v in d["xyxy"]]
        c = COLORS[d["cls"] % len(COLORS)]
        name = CLASS_NAMES[d["cls"]] if d["cls"] < len(CLASS_NAMES) else f"id{d['cls']}"
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
        tag = f"{name} {d['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), c, -1)
        cv2.putText(img, tag, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def to_yolo_lines(dets, w, h):
    lines = []
    for d in dets:
        x1, y1, x2, y2 = d["xyxy"]
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        lines.append(f"{d['cls']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="图片 / 视频 / 目录")
    ap.add_argument("--model", default=str(Path(__file__).with_name("best.onnx")))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--stride", type=int, default=10, help="视频抽帧间隔")
    ap.add_argument("--out", default="out", help="结果输出目录")
    ap.add_argument("--save-labels", action="store_true", help="同时写 YOLO 格式 txt（自动标注）")
    args = ap.parse_args()

    det = Detector(args.model, args.imgsz, args.conf, args.iou)
    src = Path(args.source)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.save_labels:
        (out / "labels").mkdir(parents=True, exist_ok=True)

    files = []
    if src.is_dir():
        files = [p for p in sorted(src.rglob("*")) if p.suffix.lower() in IMG_EXTS]
    else:
        files = [src]

    total_frames = total_dets = 0
    for f in files:
        if f.suffix.lower() in VID_EXTS:
            cap = cv2.VideoCapture(str(f))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"[video] {f.name}: {n} 帧, 每 {args.stride} 帧取一张")
            i = saved = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if i % args.stride == 0:
                    dets, dt = det(frame)
                    total_frames += 1
                    total_dets += len(dets)
                    vis = draw(frame.copy(), dets)
                    cv2.putText(vis, f"{dt:.0f} ms", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    name = f"{f.stem}_f{i:05d}"
                    cv2.imwrite(str(out / f"{name}.jpg"), vis)
                    if args.save_labels:
                        h, w = frame.shape[:2]
                        (out / "labels" / f"{name}.txt").write_text(
                            "\n".join(to_yolo_lines(dets, w, h)), encoding="utf-8")
                    saved += 1
                    if dets:
                        print(f"   frame {i:5d}: {len(dets)} 个目标 "
                              f"{[(CLASS_NAMES[d['cls']], round(d['conf'], 2)) for d in dets]}")
                i += 1
            cap.release()
            print(f"   保存 {saved} 张 -> {out}")
        else:
            img = cv2.imread(str(f))
            if img is None:
                print(f"[skip] 读不到 {f}")
                continue
            dets, dt = det(img)
            total_frames += 1
            total_dets += len(dets)
            h, w = img.shape[:2]
            print(f"[image] {f.name} {w}x{h} -> {len(dets)} 个目标, {dt:.0f} ms")
            for d in dets:
                print(f"    {CLASS_NAMES[d['cls']] if d['cls'] < len(CLASS_NAMES) else d['cls']} "
                      f"conf={d['conf']:.3f} xyxy={[round(v, 1) for v in d['xyxy']]}")
            cv2.imwrite(str(out / f"{f.stem}_pred.jpg"), draw(img.copy(), dets))
            if args.save_labels:
                (out / "labels" / f"{f.stem}.txt").write_text(
                    "\n".join(to_yolo_lines(dets, w, h)), encoding="utf-8")

    print(f"\n[done] {total_frames} 帧, 共 {total_dets} 个目标, 输出目录 {out.resolve()}")


if __name__ == "__main__":
    sys.exit(main())
