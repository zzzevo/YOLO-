# -*- coding: utf-8 -*-
"""
修正版硬负样本挖掘：**必须 CLIP 也说"这不是球"** 才挖成负样本。
（教训：只按"标签里没有"判断误检会被坑 —— 有些真球的框被前一步的过滤丢掉了，
  那条框就成了"没有标签"，被误当成误检挖走，等于教模型"真球=背景"）
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

LABELS_PROMPT = ["a basketball", "a volleyball", "a floor or wooden surface",
                 "a wall", "a line marking", "a shoe", "a person", "a cable or rope",
                 "a shadow", "a bag or cloth", "other object"]


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/ball_2c/weights/best.pt")
    ap.add_argument("--src", nargs="+", default=["raw_v19", "raw_v20", "raw_v21", "raw_v17"])
    ap.add_argument("--out", default="hard_neg2")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--target", type=int, default=700)
    ap.add_argument("--ball-thr", type=float, default=0.30,
                    help="CLIP 判'是球'的概率低于它才算真负样本（留足余量）")
    ap.add_argument("--min-crop", type=int, default=256)
    ap.add_argument("--max-crop", type=int, default=640)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    random.seed(args.seed)

    import torch
    import clip
    from PIL import Image
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cmodel, cprep = clip.load("ViT-B/32", device=dev, download_root="weights/clip")
    with torch.no_grad():
        tf = cmodel.encode_text(clip.tokenize(LABELS_PROMPT).to(dev))
        tf = tf / tf.norm(dim=-1, keepdim=True)

    def p_ball(crop):
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            f = cmodel.encode_image(cprep(Image.fromarray(rgb)).unsqueeze(0).to(dev))
            f = f / f.norm(dim=-1, keepdim=True)
            p = (100 * f @ tf.T).softmax(dim=-1)[0].cpu().numpy()
        return float(p[0] + p[1]), LABELS_PROMPT[int(p.argmax())]

    m = YOLO(args.model)
    out = Path(args.out)
    for sp in ("train", "val"):
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sp).mkdir(parents=True, exist_ok=True)

    frames = []
    for d in args.src:
        frames += sorted(Path(d).glob("*.jpg"))
    random.shuffle(frames)
    print(f"候选帧 {len(frames)} 张")

    made = rejected_as_ball = 0
    sizes, topos, tiles = Counter(), Counter(), []
    for p in frames:
        if made >= args.target:
            break
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        r = m.predict(str(p), conf=args.conf, iou=0.45, imgsz=640, device=0, verbose=False)[0]
        for b in r.boxes:
            if made >= args.target:
                break
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            bw_, bh_ = x2 - x1, y2 - y1
            for scale in (2.0, 1.6, 1.3, 1.0):
                cw = int(max(args.min_crop, min(args.max_crop, bw_ * scale)))
                ch = int(max(args.min_crop, min(args.max_crop, bh_ * scale)))
                if cw > w or ch > h:
                    continue
                cx_, cy_ = (x1 + x2) / 2, (y1 + y2) / 2
                X1 = int(max(0, min(w - cw, cx_ - cw / 2)))
                Y1 = int(max(0, min(h - ch, cy_ - ch / 2)))
                crop = img[Y1:Y1 + ch, X1:X1 + cw]
                if crop.size < 64 * 64:
                    continue
                pb, top = p_ball(crop)          # 关键：CLIP 说这是不是球
                if pb >= args.ball_thr:
                    rejected_as_ball += 1
                    break                        # 是球 -> 绝不挖成负样本
                sp = "val" if made % 8 == 0 else "train"
                name = f"hneg_{made:05d}"
                cv2.imwrite(str(out / "images" / sp / f"{name}.jpg"), crop,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                (out / "labels" / sp / f"{name}.txt").write_text("", encoding="utf-8")
                made += 1
                frac = (bw_ * bh_) / (w * h)
                sizes["大(>8%画面)" if frac > 0.08 else ("中(2~8%)" if frac > 0.02 else "小(<2%)")] += 1
                topos[top] += 1
                if len(tiles) < 12:
                    t = cv2.resize(crop, (200, 200))
                    cv2.putText(t, f"{top[:16]} {pb:.2f}", (4, 194),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1)
                    tiles.append(t)
                break
    if tiles:
        while len(tiles) % 4:
            tiles.append(np.full((200, 200, 3), 25, np.uint8))
        cv2.imwrite("hard_neg2_examples.jpg",
                    np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        print("负样本样例 -> hard_neg2_examples.jpg")
    print(f"\n挖到真负样本 {made} 个；因 CLIP 判断'是球'而放弃的候选 {rejected_as_ball} 个")
    print("  尺寸分布:", dict(sizes))
    print("  CLIP 认为它们是什么:", dict(sorted(topos.items(), key=lambda x: -x[1])))


if __name__ == "__main__":
    main()
