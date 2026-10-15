# -*- coding: utf-8 -*-
"""
专门回答："哪个模型区分篮球和排球最好？"

度量方式（比总体 mAP 更能反映"分不分得清两类"）：
  * 把每个真值框与预测框按 IoU 匹配，统计 真值类别 -> 预测类别
  * 类间判错率 = (篮球判成排球 + 排球判成篮球) / 匹配上的球框总数
  * 同时给每类的召回（篮球对不对、排球对不对）以及 ultralytics 的 per-class mAP50

用法: python compare_class_models.py [--conf 0.25]
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")
from train import install_single_process_pool

install_single_process_pool()

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

CANDIDATES = [
    ("ball_v4（2类·绿球修复）", "runs/ball_v4/weights/best.pt"),
    ("ball_v5（首个3类）", "runs/ball_v5/weights/best.pt"),
    ("ball_v8（+4段视频+网图）", "runs/ball_v8/weights/best.pt"),
    ("ball_v11（修掉89个错标）", "runs/ball_v11/weights/best.pt"),
    ("ball_v13（+259网图）", "runs/ball_v13/weights/best.pt"),
    ("ball_v14（+445网图·默认）", "runs/ball_v14/weights/best.pt"),
    ("ball_v16（+合成数据）", "runs/ball_v16/weights/best.pt"),
]


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0


def load_frames():
    """把两个验证集（人工修过的球集 + 网图集）合并，得到一个"两类都多"的测试集"""
    frames = []
    for yml in ("data_eval_ball11.yaml", "data_eval_web.yaml"):
        d = yaml.safe_load(Path(yml).read_text(encoding="utf-8"))
        root = Path(d["path"])
        lst = root / str(d["val"])
        for line in lst.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            imgp = Path(line.strip())
            lab = root / "labels" / imgp.parent.name / (imgp.stem + ".txt")
            if not lab.exists():
                continue
            gts = []
            for l in lab.read_text(encoding="utf-8").splitlines():
                if l.strip():
                    c, cx, cy, bw, bh = l.split()
                    c = int(float(c))
                    if c in (0, 1):
                        gts.append((c, (float(cx) - float(bw) / 2, float(cy) - float(bh) / 2,
                                        float(cx) + float(bw) / 2, float(cy) + float(bh) / 2)))
            if gts:
                frames.append((imgp, gts))
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou-thr", type=float, default=0.45)
    args = ap.parse_args()

    frames = load_frames()
    n_cls = Counter(c for _, gts in frames for c, _ in gts)
    print(f"测试集: {len(frames)} 帧, 球框 {sum(n_cls.values())} 个"
          f"（篮球 {n_cls[0]} / 排球 {n_cls[1]}）")
    print(f"匹配: IoU>={args.iou_thr}, 预测 conf>={args.conf}\n")

    rows = []
    for tag, mp in CANDIDATES:
        if not Path(mp).exists():
            continue
        m = YOLO(mp)
        cm = Counter()
        for imgp, gts in frames:
            r = m.predict(str(imgp), conf=args.conf, iou=0.45, imgsz=640, device=0, verbose=False)[0]
            preds = [(tuple(float(v) for v in b.xyxyn[0]), int(b.cls[0]))
                     for b in r.boxes if int(b.cls[0]) in (0, 1)]
            for gc, gbox in gts:
                best, bc = 0.0, None
                for pbox, pc in preds:
                    v = iou(gbox, pbox)
                    if v > best:
                        best, bc = v, pc
                if bc is None or best < args.iou_thr:
                    cm[(gc, "漏检")] += 1
                else:
                    cm[(gc, bc)] += 1
        b_ok, b_bad = cm[(0, 0)], cm[(0, 1)]
        v_ok, v_bad = cm[(1, 1)], cm[(1, 0)]
        b_miss, v_miss = cm[(0, "漏检")], cm[(1, "漏检")]
        matched = b_ok + b_bad + v_ok + v_bad
        wrong = b_bad + v_bad
        rows.append((tag, b_ok, b_bad, b_miss, v_ok, v_bad, v_miss,
                     wrong / max(1, matched), (b_ok + v_ok) / max(1, matched + b_miss + v_miss)))

    print(f"{'模型':26s} {'篮球对':>6s} {'篮→排':>6s} {'篮漏':>5s} "
          f"{'排球对':>6s} {'排→篮':>6s} {'排漏':>5s} {'类间判错率':>10s} {'总正确率':>8s}")
    print("-" * 96)
    for (tag, b_ok, b_bad, b_miss, v_ok, v_bad, v_miss, cr, acc) in rows:
        print(f"{tag:26s} {b_ok:6d} {b_bad:6d} {b_miss:5d} {v_ok:6d} {v_bad:6d} {v_miss:5d} "
              f"{cr*100:9.2f}% {acc*100:7.2f}%")
    best = min(rows, key=lambda x: (x[7], -x[8]))
    print(f"\n=> 类间判错率最低: {best[0]}（{best[7]*100:.2f}%）")


if __name__ == "__main__":
    main()
