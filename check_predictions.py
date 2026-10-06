# 视觉核验：在验证集上画框 + 顺便和教师模型(原始 onnx)对比，拼成联系表
import argparse
import glob
from pathlib import Path

import cv2
import numpy as np

from train import install_single_process_pool

install_single_process_pool()

from ultralytics import YOLO  # noqa: E402

ROOT = Path(__file__).parent.resolve()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/ball_v1/weights/best.pt")
    ap.add_argument("--imgs", default="dataset_ball/images/val")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default="_sheet_pred.jpg")
    ap.add_argument("--tile", type=int, default=300)
    ap.add_argument("--cols", type=int, default=4)
    args = ap.parse_args()

    files = sorted(glob.glob(str(ROOT / args.imgs / "*.jpg")))
    if not files:
        raise SystemExit("没有验证图片")
    if len(files) > args.n:
        idx = np.linspace(0, len(files) - 1, args.n).astype(int)
        files = [files[i] for i in idx]

    model = YOLO(str(ROOT / args.model))
    tiles = []
    for f in files:
        r = model.predict(f, conf=args.conf, imgsz=640, verbose=False)[0]
        vis = r.plot(line_width=2)
        h, w = vis.shape[:2]
        s = args.tile / max(h, w)
        vis = cv2.resize(vis, (int(w * s), int(h * s)))
        canvas = np.full((args.tile, args.tile, 3), 25, np.uint8)
        y = (args.tile - vis.shape[0]) // 2
        x = (args.tile - vis.shape[1]) // 2
        canvas[y:y + vis.shape[0], x:x + vis.shape[1]] = vis
        tags = [f"{r.names[int(b.cls[0])]} {float(b.conf[0]):.2f}" for b in r.boxes]
        cv2.putText(canvas, f"{Path(f).name[:16]}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, " | ".join(tags)[:34] if tags else "NO DETECTION", (4, args.tile - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0) if tags else (0, 0, 255), 1, cv2.LINE_AA)
        tiles.append(canvas)
        print(f"{Path(f).name}: {len(r.boxes)} -> {tags}")

    while len(tiles) % args.cols:
        tiles.append(np.full((args.tile, args.tile, 3), 25, np.uint8))
    rows = [np.hstack(tiles[i:i + args.cols]) for i in range(0, len(tiles), args.cols)]
    sheet = np.vstack(rows)
    cv2.imwrite(str(ROOT / args.out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"\n已写出 {args.out} ({sheet.shape[1]}x{sheet.shape[0]})")


if __name__ == "__main__":
    main()
