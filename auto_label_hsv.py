#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
用传统颜色分割自动标注「颜色鲜明」的球（HSV 阈值 + 轮廓 + 圆度过滤）。

什么时候用：神经网络对新颜色/新外观的球"失明"时（本项目实测：模型只见过橙色篮球，
对绿色 Wilson 篮球 192 帧里 110 帧零检出，连大球也看不见；尺度空间和 YOLO-World 都救不了）。
这种纯颜色偏见，用颜色分割反而最可靠。

用法:
  # 先看效果（生成一张带框的联系表，人眼确认后再批量标）
  python auto_label_hsv.py --src raw_bb2 --class-id 0 --sheet _hsv_check.jpg --limit 12
  # 批量标注（写 YOLO txt 到 --out，再人工抽查）
  python auto_label_hsv.py --src raw_bb2 --class-id 0 --out hsv_labels --min-area 0.003

注意: 一定要人眼抽查！颜色分割会被"背景里同色的东西"骗（绿墙、绿衣服、地板反光）。
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from infer_onnx import IMG_EXTS


def detect_color_ball(img, hsv_lo, hsv_hi, min_area_frac=0.003, max_boxes=2):
    """返回 [(x1,y1,x2,y2,fill)]，按面积从大到小"""
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lo), np.array(hsv_hi))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((17, 17), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    out = []
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:max_boxes]:
        a = cv2.contourArea(c)
        if a < min_area_frac * h * w:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        fill = a / (bw * bh + 1e-6)          # 外接矩形填充率：圆≈0.785
        ar = bw / (bh + 1e-6)
        if not (0.45 < fill < 0.98):          # 太瘦长/太方正都不是球
            continue
        if not (0.55 < ar < 1.8):
            continue
        out.append((x, y, x + bw, y + bh, round(fill, 3)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", help="YOLO txt 输出目录（不给就只做检查表）")
    ap.add_argument("--class-id", type=int, default=0)
    ap.add_argument("--hsv-lo", default="30,40,40", help="HSV 下限 H,S,V（OpenCV H 0~180）")
    ap.add_argument("--hsv-hi", default="100,255,230")
    ap.add_argument("--min-area", type=float, default=0.003, help="最小面积占整图比例")
    ap.add_argument("--max-boxes", type=int, default=2)
    ap.add_argument("--sheet", default="_hsv_check.jpg", help="检查表输出路径（留空则不生成）")
    ap.add_argument("--sheet-n", type=int, default=12)
    ap.add_argument("--tile", type=int, default=300)
    ap.add_argument("--cols", type=int, default=4)
    args = ap.parse_args()

    lo = tuple(int(x) for x in args.hsv_lo.split(","))
    hi = tuple(int(x) for x in args.hsv_hi.split(","))
    src = Path(args.src)
    files = [p for p in sorted(src.rglob("*")) if p.suffix.lower() in IMG_EXTS]
    if not files:
        raise SystemExit(f"{src} 里没有图片")

    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)

    hit = miss = 0
    tiles = []
    for i, p in enumerate(files):
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        boxes = detect_color_ball(img, lo, hi, args.min_area, args.max_boxes)
        if boxes:
            hit += 1
        else:
            miss += 1
        if args.out:
            lines = []
            for x1, y1, x2, y2, _ in boxes:
                cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                bw, bh = (x2 - x1) / w, (y2 - y1) / h
                lines.append(f"{args.class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            (Path(args.out) / f"{p.stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        # 检查表抽样
        if args.sheet and len(tiles) < args.sheet_n and (i % max(1, len(files) // args.sheet_n) == 0):
            vis = img.copy()
            for x1, y1, x2, y2, f in boxes:
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(vis, f"fill {f}", (x1, max(18, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            s = args.tile / max(h, w)
            vis = cv2.resize(vis, (int(w * s), int(h * s)))
            canvas = np.full((args.tile, args.tile, 3), 25, np.uint8)
            canvas[(args.tile - vis.shape[0]) // 2:(args.tile - vis.shape[0]) // 2 + vis.shape[0],
                   (args.tile - vis.shape[1]) // 2:(args.tile - vis.shape[1]) // 2 + vis.shape[1]] = vis
            cv2.putText(canvas, f"{p.name[:16]} {len(boxes)}det", (4, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            tiles.append(canvas)

    print(f"共 {len(files)} 张：检出 {hit} / 未检出 {miss}")
    if args.out:
        print(f"标签已写到 {args.out}")
    if args.sheet and tiles:
        while len(tiles) % args.cols:
            tiles.append(np.full((args.tile, args.tile, 3), 25, np.uint8))
        rows = [np.hstack(tiles[i:i + args.cols]) for i in range(0, len(tiles), args.cols)]
        cv2.imwrite(args.sheet, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"检查表: {args.sheet}（人眼确认后再用）")


if __name__ == "__main__":
    main()
