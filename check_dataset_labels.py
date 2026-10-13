# -*- coding: utf-8 -*-
"""
体检数据集：把每个类别里抽样的**标注框**裁出来放大拼图，人眼一眼看出标签对不对。

用法: python check_dataset_labels.py --data dataset_ball_v10 --per-class 24 --out-prefix label_check
"""
import argparse
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")
import cv2
import numpy as np

from infer_onnx import CLASS_NAMES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_ball_v10")
    ap.add_argument("--per-class", type=int, default=24)
    ap.add_argument("--out-prefix", default="label_check")
    ap.add_argument("--tile", type=int, default=170)
    ap.add_argument("--cols", type=int, default=6)
    args = ap.parse_args()

    root = Path(args.data)
    items = []          # (cls, img_path, box_norm)
    for sp in ("train", "val"):
        for lab in sorted((root / "labels" / sp).glob("*.txt")):
            img = root / "images" / sp / (lab.stem + ".jpg")
            if not img.exists():
                continue
            for l in lab.read_text(encoding="utf-8").splitlines():
                if not l.strip():
                    continue
                c, cx, cy, bw, bh = l.split()
                items.append((int(float(c)), img, (float(cx), float(cy), float(bw), float(bh))))

    print("数据集:", root, "| 总框数:", len(items))
    print("类别分布:", {CLASS_NAMES[k]: v for k, v in sorted(Counter(i[0] for i in items).items())})

    random.seed(1)
    for cls in sorted({i[0] for i in items}):
        pool = [i for i in items if i[0] == cls]
        print(f"  {CLASS_NAMES[cls]}: 训练集里 {len(pool)} 个框", end="")
        # 各类别框的宽高比统计（排球/篮球都应接近 1:1；如果混进细长物体会露馅）
        ars = [i[2][2] / max(1e-6, i[2][3]) for i in pool]
        print(f"，宽高比 中位 {np.median(ars):.2f} / 5%分位 {np.percentile(ars,5):.2f} / 95%分位 {np.percentile(ars,95):.2f}")

        sample = random.sample(pool, min(args.per_class, len(pool)))
        tiles = []
        for c, imgp, (cx, cy, bw, bh) in sample:
            img = cv2.imread(str(imgp))
            if img is None:
                continue
            h, w = img.shape[:2]
            x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            pad = int(max(x2 - x1, y2 - y1) * 0.25)
            cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
            cx2, cy2 = min(w, x2 + pad), min(h, y2 + pad)
            crop = img[cy1:cy2, cx1:cx2].copy()
            if crop.size == 0:
                continue
            s = args.tile / max(crop.shape[:2])
            crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))))
            canvas = np.full((args.tile, args.tile, 3), 25, np.uint8)
            canvas[(args.tile - crop.shape[0]) // 2:(args.tile - crop.shape[0]) // 2 + crop.shape[0],
                   (args.tile - crop.shape[1]) // 2:(args.tile - crop.shape[1]) // 2 + crop.shape[1]] = crop
            cv2.rectangle(canvas, (2, 2), (args.tile - 3, args.tile - 3), (60, 60, 60), 1)
            cv2.putText(canvas, imgp.name[:16], (4, args.tile - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)
            tiles.append(canvas)
        while len(tiles) % args.cols:
            tiles.append(np.full((args.tile, args.tile, 3), 25, np.uint8))
        sheet = np.vstack([np.hstack(tiles[i:i + args.cols]) for i in range(0, len(tiles), args.cols)])
        out = f"{args.out_prefix}_cls{cls}_{ {0:"basketball",1:"volleyball",2:"pillar"}[cls] }.jpg"
        cv2.imwrite(out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"     -> {out}  ({len(tiles)} 个样本)")


if __name__ == "__main__":
    main()
