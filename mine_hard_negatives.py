# -*- coding: utf-8 -*-
"""
挖"硬负样本"：让当前模型在真实帧上自己犯误检，把误检区域挖出来当负样本（空标签）。
—— 这是治"地板被框成一大块排球"的正解：负样本的尺度/纹理/光照都和误检现场一致。

要点:
  * 只要预测框与任何真值框 IoU < 0.2 才算误检
  * 裁剪时留余量(1.4~2.0 倍)，但**裁剪区内不能含任何真值框**（否则空标签会害了真球）
  * 记录误检的尺寸分布，方便判断"大色块"问题是否被覆盖
用法: python mine_hard_negatives.py --model runs/ball_2c/weights/best.pt --src raw_v19 raw_v20 raw_v21 --out hard_neg
"""
import argparse
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")
import cv2
import numpy as np
from ultralytics import YOLO

LABDIRS = [Path("v19_labels/labels"), Path("v20_labels/labels"), Path("v21_labels/labels"),
           Path("v17_labels/labels"), Path("dataset_ball_2c/labels/val"),
           Path("dataset_ball_2c/labels/train")]


def find_labels(stem):
    for d in LABDIRS:
        f = d / (stem + ".txt")
        if f.exists():
            return f
    return None


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/ball_2c/weights/best.pt")
    ap.add_argument("--src", nargs="+", default=["raw_v19", "raw_v20", "raw_v21"])
    ap.add_argument("--out", default="hard_neg")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--target", type=int, default=900)
    ap.add_argument("--min-crop", type=int, default=256)
    ap.add_argument("--max-crop", type=int, default=640)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    m = YOLO(args.model)
    out = Path(args.out)
    for sp in ("train", "val"):
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sp).mkdir(parents=True, exist_ok=True)

    frames = []
    for d in args.src:
        frames += sorted(Path(d).glob("*.jpg"))
    random.shuffle(frames)
    print(f"候选帧 {len(frames)} 张，目标硬负样本 {args.target} 个")

    made = 0
    sizes = Counter()
    cls_dist = Counter()
    tiles = []
    for p in frames:
        if made >= args.target:
            break
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        gts = []
        lf = find_labels(p.stem)
        if lf is not None:
            for l in lf.read_text(encoding="utf-8").splitlines():
                if l.strip():
                    c, cx, cy, bw, bh = l.split()
                    gts.append(((float(cx) - float(bw) / 2) * w, (float(cy) - float(bh) / 2) * h,
                                (float(cx) + float(bw) / 2) * w, (float(cy) + float(bh) / 2) * h))
        r = m.predict(str(p), conf=args.conf, iou=0.45, imgsz=640, device=0, verbose=False)[0]
        for b in r.boxes:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            cid = int(b.cls[0])
            if any(iou((x1, y1, x2, y2), g) > 0.2 for g in gts):
                continue                     # 命中了真球 -> 不是误检
            # 放大裁剪范围，但要保证裁剪区内没有真值框
            bw_, bh_ = x2 - x1, y2 - y1
            for scale in (2.0, 1.6, 1.4, 1.2, 1.0):
                cw = int(max(args.min_crop, min(args.max_crop, bw_ * scale)))
                ch = int(max(args.min_crop, min(args.max_crop, bh_ * scale)))
                cx_, cy_ = (x1 + x2) / 2, (y1 + y2) / 2
                X1 = int(max(0, min(w - cw, cx_ - cw / 2)))
                Y1 = int(max(0, min(h - ch, cy_ - ch / 2)))
                X2, Y2 = X1 + cw, Y1 + ch
                if X2 > w or Y2 > h or cw < 64 or ch < 64:
                    continue
                if any(iou((X1, Y1, X2, Y2), g) > 0.02 for g in gts):
                    continue                 # 裁到真球了，换个尺度
                crop = img[Y1:Y2, X1:X2]
                sp = "val" if made % 8 == 0 else "train"
                name = f"hneg_{made:05d}"
                cv2.imwrite(str(out / "images" / sp / f"{name}.jpg"), crop,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                (out / "labels" / sp / f"{name}.txt").write_text("", encoding="utf-8")
                made += 1
                frac = (bw_ * bh_) / (w * h)
                sizes["大(>8%画面)" if frac > 0.08 else ("中(2~8%)" if frac > 0.02 else "小(<2%)")] += 1
                cls_dist["篮球" if cid == 0 else "排球"] += 1
                if len(tiles) < 12:
                    t = cv2.resize(crop, (200, 200))
                    cv2.putText(t, f"{'篮' if cid==0 else '排'} {b.conf[0]:.2f} {int(frac*100)}%",
                                (4, 194), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    tiles.append(t)
                break
    if tiles:
        while len(tiles) % 4:
            tiles.append(np.full((200, 200, 3), 25, np.uint8))
        cv2.imwrite("hard_neg_examples.jpg",
                    np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        print("误检样例 -> hard_neg_examples.jpg")
    print(f"\n挖到硬负样本 {made} 个")
    print("  误检框尺寸分布:", dict(sizes))
    print("  被误判成什么类:", dict(cls_dist))


if __name__ == "__main__":
    main()
