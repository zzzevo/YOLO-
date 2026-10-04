#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从视频批量抽帧，给 build_dataset.py 当输入。
带模糊过滤和相似帧去重，避免抽出一堆几乎一样的废图污染数据集。

用法:
  python extract_frames.py --src D:\\videos --dst D:\\raw_images
  python extract_frames.py --src a.mp4 --dst raw --stride 5 --blur-thresh 60 --dedup 0.98
  python extract_frames.py --src D:\\videos --dst raw --roi 0,0,448,448   # 只保留 ROI 区域

参数说明:
  --stride      每 N 帧取一张（30fps 视频取 5 ≈ 每秒 6 张）
  --blur-thresh Laplacian 方差下限，低于此值判为糊图丢弃；0=不过滤
  --dedup       与上一张的相似度上限（直方图相关性），>= 该值视为重复丢弃；1.0=不去重
  --roi         x,y,w,h，只裁这块区域（比如 MaixCam 的赛道上半部分）
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

VID_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".m4v"}


def hist_sim(a, b):
    ha = cv2.calcHist([cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)], [0], None, [64], [0, 256])
    hb = cv2.calcHist([cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)], [0], None, [64], [0, 256])
    cv2.normalize(ha, ha); cv2.normalize(hb, hb)
    return cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="视频文件或目录")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--blur-thresh", type=float, default=60.0)
    ap.add_argument("--dedup", type=float, default=0.98)
    ap.add_argument("--roi", default="", help="x,y,w,h")
    ap.add_argument("--max-per-video", type=int, default=0, help="0=不限")
    ap.add_argument("--prefix", default="", help="文件名前缀，默认用视频名")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    vids = [src] if src.is_file() else [p for p in sorted(src.rglob("*")) if p.suffix.lower() in VID_EXTS]
    if not vids:
        raise SystemExit(f"{src} 下没有视频")

    roi = None
    if args.roi:
        x, y, w, h = [int(v) for v in args.roi.split(",")]
        roi = (x, y, w, h)

    grand = 0
    for v in vids:
        cap = cv2.VideoCapture(str(v))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        prefix = args.prefix or v.stem
        print(f"[video] {v.name}  {total} 帧 @{fps:.0f}fps -> 每 {args.stride} 帧取一张")
        i = kept = dropped_blur = dropped_dup = 0
        prev = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % args.stride == 0:
                img = frame[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]] if roi else frame
                if args.blur_thresh > 0:
                    if cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() < args.blur_thresh:
                        dropped_blur += 1
                        i += 1
                        continue
                if prev is not None and args.dedup < 1.0 and hist_sim(prev, img) >= args.dedup:
                    dropped_dup += 1
                    i += 1
                    continue
                cv2.imwrite(str(dst / f"{prefix}_{i:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                prev = img
                kept += 1
                if args.max_per_video and kept >= args.max_per_video:
                    break
            i += 1
        cap.release()
        grand += kept
        print(f"    保留 {kept}  糊图丢弃 {dropped_blur}  重复丢弃 {dropped_dup}")

    print(f"\n[done] 共抽 {grand} 张 -> {dst.resolve()}")
    print(f"  下一步: python build_dataset.py --src {dst} --dst D:\\datasets\\robocup_balls")


if __name__ == "__main__":
    main()
