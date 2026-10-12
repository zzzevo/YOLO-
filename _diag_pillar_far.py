# -*- coding: utf-8 -*-
"""
诊断"远距离定位柱"到底难在哪：按柱子的像素尺寸分档，统计
  (a) 颜色标注器能不能找到  (b) 当前模型能不能找到
用法: python _diag_pillar_far.py --src raw_pillar_far --model runs/ball_v9/weights/best.onnx
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, ".")
import cv2
import numpy as np

from auto_label_pillar import detect_pillar
from infer_onnx import CLASS_NAMES, Detector, IMG_EXTS


def bucket(h):
    if h < 30:
        return "1) 高<30px"
    if h < 60:
        return "2) 30~60px"
    if h < 120:
        return "3) 60~120px"
    if h < 250:
        return "4) 120~250px"
    return "5) 高>250px"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="raw_pillar_far")
    ap.add_argument("--model", default="runs/ball_v9/weights/best.onnx")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    det = Detector(args.model, 640, args.conf)
    files = [p for p in sorted(Path(args.src).rglob("*")) if p.suffix.lower() in IMG_EXTS]
    stat = defaultdict(lambda: {"n": 0, "color": 0, "model": 0, "model_conf": [], "px": []})
    miss_color, miss_model = [], []

    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        cb = detect_pillar(img)                       # 颜色标注器
        mb = [d for d in det(img)[0] if CLASS_NAMES[d["cls"]] == "定位柱"]
        # 以"颜色标注器的框"为准分档（它更接近真值）；两套都没有就跳过
        ref = None
        if cb:
            x1, y1, x2, y2, _, _ = cb[0]
            ref = (x2 - x1, y2 - y1, "color")
        elif mb:
            x1, y1, x2, y2 = mb[0]["xyxy"]
            ref = (x2 - x1, y2 - y1, "model")
        if ref is None:
            continue
        bw, bh, src = ref
        b = bucket(bh)
        s = stat[b]
        s["n"] += 1
        s["px"].append(int(bh))
        s["color"] += int(bool(cb))
        s["model"] += int(bool(mb))
        if mb:
            s["model_conf"].append(max(d["conf"] for d in mb))
        if not cb:
            miss_color.append(p.name)
        if not mb:
            miss_model.append(p.name)

    print(f"共 {len(files)} 帧，其中有柱子参考的 {sum(v['n'] for v in stat.values())} 帧\n")
    print(f"{'柱子高度':14s} {'帧数':>5s} {'颜色标注':>9s} {'v9 模型':>9s} {'模型平均置信度':>14s}")
    print("-" * 60)
    for b in sorted(stat):
        s = stat[b]
        conf = f"{np.mean(s['model_conf']):.2f}" if s["model_conf"] else "—"
        print(f"{b:14s} {s['n']:5d} {s['color']/s['n']*100:8.0f}% {s['model']/s['n']*100:8.0f}% {conf:>14s}")
    print(f"\n颜色标注器漏掉 {len(miss_color)} 帧，v9 漏掉 {len(miss_model)} 帧")
    if miss_model:
        print("v9 漏检的帧（前 12 个）:", ", ".join(miss_model[:12]))
    # 所有柱子高度分布
    allpx = [p for v in stat.values() for p in v["px"]]
    if allpx:
        print(f"\n柱子像素高度：最小 {min(allpx)} / 中位 {int(np.median(allpx))} / 最大 {max(allpx)}")
        print(f"  高度 <60px 的帧占比: {sum(1 for p in allpx if p < 60)/len(allpx)*100:.0f}%")


if __name__ == "__main__":
    main()
