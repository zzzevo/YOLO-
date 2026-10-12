# -*- coding: utf-8 -*-
"""
按**目标像素尺寸**分档评估召回率——专门用来衡量"远距离/小目标"能力。

为什么需要：总体 mAP 会掩盖小目标的短板。本工具把目标按像素尺寸分档，
分别报告召回，一眼看出"多远开始失效"。还能对比不同推理方式（整图 / 高分辨率 / 分块放大）。

用法:
  # 基础：按尺寸分档看召回
  python eval_by_size.py --src raw_pillar_far --model runs/ball_v10/weights/best.onnx --cls 定位柱
  # 对比三种推理方式（--model960 传 960 输入的 onnx）
  python eval_by_size.py --src raw_pillar_far --model runs/ball_v10/weights/best.onnx `
      --model960 runs/ball_v10/weights/best_960.onnx --cls 定位柱 --compare
  # 参考真值用颜色标注器（对定位柱很准）；也可 --gt-labels 指定 YOLO 标签目录
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import cv2
import numpy as np

from infer_onnx import CLASS_NAMES, Detector, IMG_EXTS
from infer_tiled import detect_tiled


def buckets():
    return [(0, 32, "1) <32px"), (32, 64, "2) 32~64px"), (64, 128, "3) 64~128px"),
            (128, 256, "4) 128~256px"), (256, 512, "5) 256~512px"), (512, 10 ** 9, "6) >512px")]


def which_bucket(h):
    for lo, hi, name in buckets():
        if lo <= h < hi:
            return name
    return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="图片目录")
    ap.add_argument("--model", required=True, help="640 输入的 ONNX")
    ap.add_argument("--model960", help="960 输入的 ONNX（用于对比）")
    ap.add_argument("--cls", default="定位柱", help="类别名，逗号分隔")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--gt", default="color", choices=["color", "labels"],
                    help="真值来源：color=颜色标注器（定位柱专用）/ labels=YOLO 标签目录")
    ap.add_argument("--gt-labels", help="--gt labels 时的标签目录")
    ap.add_argument("--compare", action="store_true", help="对比 整图640 / 整图960 / 分块放大")
    args = ap.parse_args()

    want = {CLASS_NAMES.index(c.strip()) for c in args.cls.split(",") if c.strip() in CLASS_NAMES}
    files = [p for p in sorted(Path(args.src).rglob("*")) if p.suffix.lower() in IMG_EXTS]
    if not files:
        raise SystemExit(f"{args.src} 里没有图片")

    # 真值
    if args.gt == "color":
        from auto_label_pillar import detect_pillar
        gt = {}
        for p in files:
            img = cv2.imread(str(p))
            if img is None:
                continue
            h, w = img.shape[:2]
            bs = detect_pillar(img)
            if bs:
                gt[p.name] = [(bs[0][0] / w, bs[0][1] / h, bs[0][2] / w, bs[0][3] / h)]
    else:
        gt = {}
        for p in files:
            lab = Path(args.gt_labels) / (p.stem + ".txt")
            if not lab.exists():
                continue
            h, w = cv2.imread(str(p)).shape[:2]
            boxes = []
            for l in lab.read_text(encoding="utf-8").splitlines():
                if not l.strip():
                    continue
                c, cx, cy, bw, bh = l.split()
                if int(c) in want:
                    boxes.append((float(cx) - float(bw) / 2, float(cy) - float(bh) / 2,
                                  float(cx) + float(bw) / 2, float(cy) + float(bh) / 2))
            if boxes:
                gt[p.name] = boxes

    def iou(a, b):
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / ua if ua > 0 else 0

    modes = [("整图 640", args.model, 640, False)]
    if args.compare:
        if args.model960:
            modes.append(("整图 960", args.model960, 960, False))
        modes.append(("分块 2x2 放大", args.model, 640, True))

    print(f"共 {len(files)} 张，有真值的 {len(gt)} 张；类别: {','.join(CLASS_NAMES[i] for i in sorted(want))}")
    for tag, mpath, imgsz, tiled in modes:
        det = Detector(mpath, imgsz, args.conf)
        stat = defaultdict(lambda: [0, 0])
        for p in files:
            if p.name not in gt:
                continue
            img = cv2.imread(str(p))
            h, w = img.shape[:2]
            dets = (detect_tiled(det, img, tiles=2, overlap=0.25, cls_filter=want) if tiled
                    else [d for d in det(img)[0] if d["cls"] in want])
            for g in gt[p.name]:
                gh = (g[3] - g[1]) * h                      # 真值高度（像素）
                b = which_bucket(gh)
                hit = any(iou(g, (d["xyxy"][0] / w, d["xyxy"][1] / h,
                                 d["xyxy"][2] / w, d["xyxy"][3] / h)) >= 0.3 for d in dets)
                stat[b][1] += 1
                stat[b][0] += int(hit)
        print(f"\n【{tag}】")
        print(f"  {'目标高度':14s} {'样本':>5s} {'召回':>7s}")
        tot = [0, 0]
        for _, _, name in buckets():
            if name in stat and stat[name][1]:
                hit, n = stat[name]
                tot[0] += hit
                tot[1] += n
                flag = "  ← 短板" if hit / n < 0.9 else ""
                print(f"  {name:14s} {n:5d} {hit/n*100:6.0f}%{flag}")
        if tot[1]:
            print(f"  {'合计':14s} {tot[1]:5d} {tot[0]/tot[1]*100:6.0f}%")


if __name__ == "__main__":
    main()
