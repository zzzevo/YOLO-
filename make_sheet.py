# 把一批图片拼成一张联系表(contact sheet)，用于快速人眼检查数据内容
import argparse
import glob
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="图片目录或通配符")
    ap.add_argument("--out", default="_sheet.jpg")
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--tile", type=int, default=220)
    ap.add_argument("--n", type=int, default=24, help="最多取多少张（按文件名均匀取样）")
    args = ap.parse_args()

    files = sorted(glob.glob(str(Path(args.src) / "*.jpg")) if Path(args.src).is_dir() else glob.glob(args.src))
    if not files:
        raise SystemExit("没有图片")
    if len(files) > args.n:
        idx = np.linspace(0, len(files) - 1, args.n).astype(int)
        files = [files[i] for i in idx]

    tiles = []
    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        h, w = img.shape[:2]
        s = args.tile / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)))
        canvas = np.full((args.tile, args.tile, 3), 30, np.uint8)
        y = (args.tile - img.shape[0]) // 2
        x = (args.tile - img.shape[1]) // 2
        canvas[y:y + img.shape[0], x:x + img.shape[1]] = img
        cv2.putText(canvas, Path(f).name[:20], (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(canvas)

    while len(tiles) % args.cols:
        tiles.append(np.full((args.tile, args.tile, 3), 30, np.uint8))
    rows = [np.hstack(tiles[i:i + args.cols]) for i in range(0, len(tiles), args.cols)]
    sheet = np.vstack(rows)
    cv2.imwrite(args.out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"{len(files)} 张 -> {args.out}  ({sheet.shape[1]}x{sheet.shape[0]})")


if __name__ == "__main__":
    main()
