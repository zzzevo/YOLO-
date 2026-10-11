#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自动标注「蓝-绿-蓝定位柱」（或任何"多色块拼成的标记板"）。

原理：定位柱是**强饱和的蓝色+绿色块**，和灰色地面区分度极高。做法：
  1) 分别取蓝、绿的 HSV 掩码
  2) 合并 + 膨胀，让蓝绿段连成一整块
  3) 每个大连通块必须**同时含有足够的蓝和足够的绿**（这就是"定位柱"的签名）
  4) 取该连通块的外接矩形作为标注框（整块板，含两端蓝色）

用法:
  python auto_label_pillar.py --src raw_pillar --out pillar_labels --class-id 2 --sheet _pillar_check.jpg
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from infer_onnx import IMG_EXTS


def detect_pillar(img, blue_lo=(95, 70, 50), blue_hi=(140, 255, 255),
                  green_lo=(35, 70, 50), green_hi=(90, 255, 255),
                  min_area_frac=0.0025, min_each_frac=0.12):
    """返回 [(x1,y1,x2,y2, blue_frac, green_frac)]"""
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mb = cv2.inRange(hsv, np.array(blue_lo), np.array(blue_hi))
    mg = cv2.inRange(hsv, np.array(green_lo), np.array(green_hi))
    k = np.ones((9, 9), np.uint8)
    mb = cv2.morphologyEx(mb, cv2.MORPH_OPEN, k)
    mg = cv2.morphologyEx(mg, cv2.MORPH_OPEN, k)

    both = cv2.bitwise_or(mb, mg)
    # 膨胀把蓝绿段之间的缝隙连起来（板子中间没有别的颜色，缝隙很小）
    both = cv2.dilate(both, np.ones((13, 13), np.uint8), iterations=2)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(both, 8)

    out = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < min_area_frac * h * w:
            continue
        m = (lab[y:y + bh, x:x + bw] == i)
        bcnt = int(np.count_nonzero(m & (mb[y:y + bh, x:x + bw] > 0)))
        gcnt = int(np.count_nonzero(m & (mg[y:y + bh, x:x + bw] > 0)))
        tot = max(1, bcnt + gcnt)
        bf, gf = bcnt / tot, gcnt / tot
        # 签名：蓝绿都要有相当比例（定位柱 = 蓝30 + 绿40 + 蓝30）
        if bf < min_each_frac or gf < min_each_frac:
            continue
        # 长条形状：定位柱是 1:5 的长条（透视下会更长或更短，放宽到 1:1.5~1:12）
        ar = max(bw, bh) / max(1, min(bw, bh))
        if not (1.3 <= ar <= 14):
            continue
        out.append((x, y, x + bw, y + bh, round(bf, 2), round(gf, 2)))
    out.sort(key=lambda t: -((t[2] - t[0]) * (t[3] - t[1])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", help="YOLO txt 输出目录")
    ap.add_argument("--class-id", type=int, default=2, help="定位柱的类别 id")
    ap.add_argument("--max-boxes", type=int, default=1)
    ap.add_argument("--sheet", default="_pillar_check.jpg")
    ap.add_argument("--sheet-n", type=int, default=12)
    ap.add_argument("--tile", type=int, default=300)
    ap.add_argument("--cols", type=int, default=4)
    args = ap.parse_args()

    src = Path(args.src)
    files = [p for p in sorted(src.rglob("*")) if p.suffix.lower() in IMG_EXTS]
    if not files:
        raise SystemExit(f"{src} 里没有图片")
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)

    hit = 0
    tiles = []
    step = max(1, len(files) // args.sheet_n)
    for i, p in enumerate(files):
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        boxes = detect_pillar(img)[: args.max_boxes]
        if boxes:
            hit += 1
        if args.out:
            (Path(args.out) / f"{p.stem}.txt").write_text(
                "\n".join(f"{args.class_id} {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} "
                          f"{(x2-x1)/w:.6f} {(y2-y1)/h:.6f}" for x1, y1, x2, y2, _, _ in boxes),
                encoding="utf-8")
        if args.sheet and len(tiles) < args.sheet_n and i % step == 0:
            vis = img.copy()
            for x1, y1, x2, y2, bf, gf in boxes:
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 3)
                cv2.putText(vis, f"b{bf} g{gf}", (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            s = args.tile / max(h, w)
            vis = cv2.resize(vis, (int(w * s), int(h * s)))
            c = np.full((args.tile, args.tile, 3), 25, np.uint8)
            c[(args.tile - vis.shape[0]) // 2:(args.tile - vis.shape[0]) // 2 + vis.shape[0],
              (args.tile - vis.shape[1]) // 2:(args.tile - vis.shape[1]) // 2 + vis.shape[1]] = vis
            cv2.putText(c, f"{p.name[:14]}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            cv2.putText(c, f"{len(boxes)} det", (4, args.tile - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0) if boxes else (0, 0, 255), 1)
            tiles.append(c)

    print(f"共 {len(files)} 张：检出定位柱 {hit} 张（{hit/max(1,len(files))*100:.0f}%）")
    if args.out:
        print(f"标签写到 {args.out}")
    if args.sheet and tiles:
        while len(tiles) % args.cols:
            tiles.append(np.full((args.tile, args.tile, 3), 25, np.uint8))
        rows = [np.hstack(tiles[i:i + args.cols]) for i in range(0, len(tiles), args.cols)]
        cv2.imwrite(args.sheet, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"检查表: {args.sheet}")


if __name__ == "__main__":
    main()
