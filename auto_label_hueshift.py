#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
"色相平移"自动标注：专治模型的**颜色偏见**。

原理：模型只见过橙色篮球，看到粉/紫/蓝/绿球就不认。但把图片的色相整体旋转，
让球变成橙色，模型立刻又能认出来 —— 而**色相旋转不改变任何位置**，
所以检测到的框可以直接用回原图。

做法：对每张图尝试多个色相偏移(0/30/60/90/120/150/180 度)，把"最有把握的那个框"留下，
跨偏移做 NMS 去重。

用法:
  python auto_label_hueshift.py --src raw_web_bb --out hue_labels --conf 0.45 --sheet _hue_check.jpg
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from infer_onnx import CLASS_NAMES, Detector, IMG_EXTS, nms, to_yolo_lines


def hue_shift(img, deg):
    if deg == 0:
        return img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0].astype(np.int16) + int(deg / 2)) % 180   # OpenCV H: 0~180
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def looks_like_ball(img, box, min_area_frac=0.02):
    """严格过滤：这个框里像不像一个"单一颜色的圆球"？
    色相平移会把彩虹/绿植/文字也变成"橙色球状物"，必须用形状+颜色一致性挡掉。
    返回 (是否通过, 说明)
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(max(0, min(v, d))) for v, d in zip(box, (w, h, w, h))]
    bw, bh = x2 - x1, y2 - y1
    if bw < 12 or bh < 12:
        return False, "太小"
    if (bw * bh) < min_area_frac * w * h:
        return False, "面积太小"
    ar = bw / bh
    if not (0.6 < ar < 1.7):
        return False, "不方正"
    crop = img[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hh, ss, vv = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    if float(np.median(vv)) < 25:
        return False, "太暗"
    # 主色占比：真球大部分是同一个色相；彩虹/花纹/文字都不满足
    valid = ss > 35
    if valid.sum() < 0.15 * valid.size:
        return False, "饱和度太低"
    hist = np.bincount(hh[valid].ravel(), minlength=180)
    # 以主色相为中心 ±12 的窗口
    peak = int(hist.argmax())
    win = sum(hist[(peak + d) % 180] for d in range(-12, 13))
    frac = win / max(1, valid.sum())
    if frac < 0.45:
        return False, f"颜色太杂({frac:.2f})"
    # 圆度：主色掩码的最大连通块的填充率
    lo = int(max(0, peak - 12)); hi = int(min(179, peak + 12))
    mask = cv2.inRange(hsv, (lo, 35, 20), (hi, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return False, "无连通块"
    big = 1 + int(np.argmax(stats[1:, 4]))
    x, y, bw2, bh2, area = stats[big]
    fill = area / (bw2 * bh2 + 1e-6)
    if fill < 0.55:
        return False, f"不圆({fill:.2f})"
    if not (0.55 < bw2 / (bh2 + 1e-6) < 1.8):
        return False, "长宽比不对"
    if area < 0.35 * (bw * bh):
        return False, "球体占比太小"
    return True, f"ok(fill={fill:.2f},主色{frac:.2f})"


def detect_multihue(det, img, shifts=(0, 30, 60, 90, 120, 150, 180), iou_thr=0.5):
    boxes, scores, classes = [], [], []
    for s in shifts:
        d, _ = det(hue_shift(img, s))
        for x in d:
            boxes.append(x["xyxy"])
            scores.append(x["conf"])
            classes.append(x["cls"])
    if not boxes:
        return []
    out = []
    for c in sorted(set(classes)):
        idx = [i for i, cc in enumerate(classes) if cc == c]
        xywh = [[(boxes[i][0] + boxes[i][2]) / 2, (boxes[i][1] + boxes[i][3]) / 2,
                 boxes[i][2] - boxes[i][0], boxes[i][3] - boxes[i][1]] for i in idx]
        for j in nms(xywh, [scores[i] for i in idx], iou_thr):
            i = idx[j]
            out.append({"xyxy": boxes[i], "conf": scores[i], "cls": c})
    out.sort(key=lambda x: -x["conf"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", help="YOLO txt 输出目录")
    ap.add_argument("--model", default="runs/ball_v3/weights/best.onnx")
    ap.add_argument("--conf", type=float, default=0.45, help="保留门槛")
    ap.add_argument("--shifts", default="0,30,60,90,120,150,180")
    ap.add_argument("--max-boxes", type=int, default=2)
    ap.add_argument("--force-class", type=int, default=0,
                    help="把保留的框类别强制成这个 id（这批图全是篮球, 默认 0）")
    ap.add_argument("--no-strict", action="store_true", help="关掉'像不像球'的严格过滤（不推荐）")
    ap.add_argument("--sheet", default="_hue_check.jpg")
    ap.add_argument("--sheet-n", type=int, default=12)
    ap.add_argument("--tile", type=int, default=300)
    ap.add_argument("--cols", type=int, default=4)
    args = ap.parse_args()

    shifts = tuple(int(x) for x in args.shifts.split(","))
    det = Detector(args.model, 640, 0.15)
    src = Path(args.src)
    files = [p for p in sorted(src.rglob("*")) if p.suffix.lower() in IMG_EXTS]
    if not files:
        raise SystemExit(f"{src} 里没有图片")
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)

    plain_hit = multi_hit = strict_drop = 0
    drop_reasons = {}
    tiles = []
    step = max(1, len(files) // args.sheet_n)
    for i, p in enumerate(files):
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        base = det(img)[0]
        if any(d["conf"] >= args.conf for d in base):
            plain_hit += 1
        good = [d for d in detect_multihue(det, img, shifts) if d["conf"] >= args.conf]
        reasons = []
        if not args.no_strict:
            kept_strict = []
            for d in good:
                ok, why = looks_like_ball(img, d["xyxy"])
                if ok:
                    kept_strict.append(d)
                else:
                    reasons.append(why)
            good = kept_strict
        good = good[: args.max_boxes]
        if args.force_class is not None:
            good = [dict(d, cls=args.force_class) for d in good]
        if good:
            multi_hit += 1
        strict_drop += len(reasons)
        for r in reasons[:2]:
            drop_reasons[r.split("(")[0]] = drop_reasons.get(r.split("(")[0], 0) + 1
        if args.out:
            (Path(args.out) / f"{p.stem}.txt").write_text(
                "\n".join(to_yolo_lines(good, w, h)), encoding="utf-8")
        if args.sheet and len(tiles) < args.sheet_n and i % step == 0:
            vis = img.copy()
            for d in good:
                x1, y1, x2, y2 = [int(v) for v in d["xyxy"]]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(vis, f"{CLASS_NAMES[d['cls']]} {d['conf']:.2f}", (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            s = args.tile / max(h, w)
            vis = cv2.resize(vis, (int(w * s), int(h * s)))
            c = np.full((args.tile, args.tile, 3), 25, np.uint8)
            c[(args.tile - vis.shape[0]) // 2:(args.tile - vis.shape[0]) // 2 + vis.shape[0],
              (args.tile - vis.shape[1]) // 2:(args.tile - vis.shape[1]) // 2 + vis.shape[1]] = vis
            cv2.putText(c, f"{p.name[:14]}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            cv2.putText(c, f"{len(good)} det", (4, args.tile - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0) if good else (0, 0, 255), 1)
            tiles.append(c)

    print(f"共 {len(files)} 张")
    print(f"  原图直接检测 ≥{args.conf}: {plain_hit} 张")
    if drop_reasons: print("  被严格过滤拦下:", dict(sorted(drop_reasons.items(), key=lambda x:-x[1])[:5]))
    print(f"  色相平移后检测 ≥{args.conf}: {multi_hit} 张   (+{multi_hit - plain_hit})")
    if args.out:
        print(f"  标签写到 {args.out}")
    if args.sheet and tiles:
        while len(tiles) % args.cols:
            tiles.append(np.full((args.tile, args.tile, 3), 25, np.uint8))
        rows = [np.hstack(tiles[i:i + args.cols]) for i in range(0, len(tiles), args.cols)]
        cv2.imwrite(args.sheet, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"  检查表: {args.sheet}")


if __name__ == "__main__":
    main()
