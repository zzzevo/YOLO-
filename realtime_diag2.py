# -*- coding: utf-8 -*-
"""用**真正连续**的帧序列量实时手感：逐帧检出、连续漏检、框闪烁、时序确认的效果。"""
import sys
from pathlib import Path

sys.path.insert(0, ".")
from train import install_single_process_pool

install_single_process_pool()

import cv2
from ultralytics import YOLO

MODEL = sys.argv[1] if len(sys.argv) > 1 else "runs/ball_2c/weights/best.pt"
SRC = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("_seq_frames")
m = YOLO(MODEL)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0


seqs = {}
for p in sorted(SRC.glob("*.jpg")):
    pref = p.stem.split("_")[0]
    seqs.setdefault(pref, []).append(p)
for k in seqs:
    seqs[k].sort(key=lambda p: int("".join(ch for ch in p.stem.split("_")[-1] if ch.isdigit()) or 0))
print("连续序列:", {k: len(v) for k, v in seqs.items()})

CONFS = [0.30, 0.40, 0.50, 0.60]
cache = {}
for k, ps in seqs.items():
    for p in ps:
        r = m.predict(str(p), conf=0.25, iou=0.45, imgsz=640, device=0, verbose=False)[0]
        allb = [([float(v) for v in b.xyxy[0]], float(b.conf[0])) for b in r.boxes
                if int(b.cls[0]) in (0, 1)]
        cache[p] = {c: [bx for bx, cf in allb if cf >= c] for c in CONFS}

print(f"\n{'conf':>5s} {'逐帧检出率':>10s} {'最长连续漏检(帧)':>14s} {'平均连续漏检':>12s} {'闪烁/帧':>8s} {'框/帧':>7s}")
print("-" * 66)
for c in CONFS:
    dr, runs, flick, nb = [], [], [], []
    for k, ps in seqs.items():
        run, prev = 0, []
        for p in ps:
            boxes = cache.get(p, {}).get(c, [])
            run = 0 if boxes else run + 1
            if not boxes:
                runs.append(run)
            dr.append(1 if boxes else 0)
            nb.append(len(boxes))
            if prev:
                flick.append(sum(1 for pb in prev if not any(iou(pb, b) > 0.3 for b in boxes)))
            prev = boxes
    print(f"{c:5.2f} {sum(dr)/len(dr)*100:9.1f}% {max(runs) if runs else 0:14d} "
          f"{sum(runs)/max(1,len(runs)):12.2f} {sum(flick)/max(1,len(flick)):8.2f} "
          f"{sum(nb)/len(nb):7.2f}")

print("\n=== 时序确认（连续 N 帧都命中才显示）===")
print(f"{'conf':>5s} {'N':>3s} {'框/帧':>7s} {'闪烁/帧':>8s} {'延迟':>5s}")
for c in (0.40, 0.50):
    for N in (1, 2, 3):
        kept, flick = [], []
        for k, ps in seqs.items():
            prev_final, hits = [], {}
            for p in ps:
                boxes = cache.get(p, {}).get(c, [])
                newhit = {}
                for b in boxes:
                    mk = None
                    for key in hits:
                        if iou(key, b) > 0.3:
                            mk = key
                            break
                    newhit[mk if mk else tuple(b)] = hits.get(mk, 0) + 1 if mk else 1
                hits = newhit
                final = [list(k2) for k2, n in hits.items() if n >= N]
                kept.append(len(final))
                if prev_final:
                    flick.append(sum(1 for pb in prev_final
                                     if not any(iou(pb, b) > 0.3 for b in final)))
                prev_final = final
        print(f"{c:5.2f} {N:3d} {sum(kept)/len(kept):7.2f} {sum(flick)/max(1,len(flick)):8.2f} {N-1:4d}帧")
