# -*- coding: utf-8 -*-
"""
"用起来到底好不好"的实测（这些是 mAP 看不出来的）：
  1) 逐帧漏检：连续多少帧完全看不到球（丢球多久）
  2) 误检闪烁：相邻帧之间新出现/消失的框有多少（框一闪一闪 = 体验差）
  3) 置信度阈值的影响：0.3~0.7 下上面指标怎么变
  4) 时序确认的效果：要求连续 N 帧都出现才画框，能消掉多少闪烁、代价是延迟多少帧
数据用**你自己的视频连续帧**（按顺序），不是抽样图。
"""
import sys
from pathlib import Path

sys.path.insert(0, ".")
from train import install_single_process_pool

install_single_process_pool()

import cv2
from ultralytics import YOLO

MODEL = sys.argv[1] if len(sys.argv) > 1 else "runs/ball_2c/weights/best.pt"
m = YOLO(MODEL)
ROOT = Path("dataset_ball_2c")


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0


# 连续帧序列：每个前缀（视频号）按帧号排序
seqs = {}
for lab in sorted((ROOT / "labels" / "val").glob("*.txt")):
    p = ROOT / "images" / "val" / (lab.stem + ".jpg")
    if not p.exists():
        continue
    pref = lab.stem.split("__")[0]
    if "__" not in lab.stem:            # 跳过地板负样本等没有帧序号的
        continue
    seqs.setdefault(pref, []).append(p)
for k in seqs:
    seqs[k].sort(key=lambda p: int(p.stem.split("__")[1]))
seqs = {k: v for k, v in seqs.items() if len(v) >= 20}
print("连续序列:", {k: len(v) for k, v in seqs.items()})

# 预先算好每帧的检测框（按 conf 存多档，避免重复推理）
CONFS = [0.30, 0.40, 0.50, 0.60]
cache = {}          # path -> {conf: [boxes]}
for k, ps in seqs.items():
    for p in ps:
        img = cv2.imread(str(p))
        if img is None:
            continue
        r = m.predict(str(p), conf=0.25, iou=0.45, imgsz=640, device=0, verbose=False)[0]
        allb = [([float(v) for v in b.xyxy[0]], float(b.conf[0])) for b in r.boxes
                if int(b.cls[0]) in (0, 1)]
        cache[p] = {c: [bx for bx, cf in allb if cf >= c] for c in CONFS}
print(f"已推理 {len(cache)} 帧\n")

print(f"{'conf':>5s} {'帧级检出率':>10s} {'最长连续漏检':>12s} {'平均漏检段':>10s} {'闪烁/帧':>8s}")
print("-" * 56)
res = {}
for c in CONFS:
    det_rate, miss_runs, flick = [], [], []
    for k, ps in seqs.items():
        run = 0
        prev = []
        for p in ps:
            boxes = cache.get(p, {}).get(c, [])
            if boxes:
                run = 0
            else:
                run += 1
                miss_runs.append(run)
            det_rate.append(1 if boxes else 0)
            if prev:
                # 上一帧的框在本帧找不到匹配 = 闪烁
                gone = sum(1 for pb in prev if not any(iou(pb, b) > 0.3 for b in boxes))
                flick.append(gone)
            prev = boxes
    longest = max(miss_runs) if miss_runs else 0
    avg_miss = sum(miss_runs) / max(1, len(miss_runs))
    print(f"{c:5.2f} {sum(det_rate)/len(det_rate)*100:9.1f}% {longest:12d} {avg_miss:10.2f} "
          f"{sum(flick)/max(1,len(flick)):8.2f}")
    res[c] = (sum(det_rate)/len(det_rate), longest, sum(flick)/max(1,len(flick)))

print("\n=== 时序确认：要求连续 N 帧都命中才显示（消闪烁）===")
print(f"{'conf':>5s} {'N':>3s} {'留下的框/帧':>11s} {'闪烁/帧':>8s} {'延迟(帧)':>8s}")
for c in (0.40, 0.50):
    for N in (1, 2, 3):
        kept, flick, delay = [], [], []
        for k, ps in seqs.items():
            prev_final = []
            hit_count = {}
            for p in ps:
                boxes = cache.get(p, {}).get(c, [])
                # 与上一帧的确认框匹配，累加命中计数
                newhit = {}
                for b in boxes:
                    matched = None
                    for key in hit_count:
                        if iou(key, b) > 0.3:
                            matched = key
                            break
                    key = matched if matched else tuple(b)
                    newhit[key] = hit_count.get(matched, 0) + 1 if matched else 1
                hit_count = newhit
                final = [list(k2) for k2, n in hit_count.items() if n >= N]
                kept.append(len(final))
                if prev_final:
                    gone = sum(1 for pb in prev_final if not any(iou(pb, b) > 0.3 for b in final))
                    flick.append(gone)
                prev_final = final
        print(f"{c:5.2f} {N:3d} {sum(kept)/len(kept):11.2f} {sum(flick)/max(1,len(flick)):8.2f} "
              f"{N-1:8d}")
