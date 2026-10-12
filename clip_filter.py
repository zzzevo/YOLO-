# -*- coding: utf-8 -*-
"""
用 CLIP 给"模型检测到的球"做二次语义判别——治网图标注的污染。

背景：模型会把橙子、玫瑰、彩虹、地图、人脸都当成球，形状/颜色启发式区分不了
"橙子"和"橙色篮球"。但 CLIP 是语义模型，它**知道**篮球和橙子的区别。
做法：把检测框裁出来 → CLIP 零样本分类 → 只有"篮球/排球"胜出才保留。

用法: python clip_filter.py --src raw_web_all --out clip_check.jpg
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from infer_onnx import CLASS_NAMES, Detector, IMG_EXTS

LABELS = [
    "a basketball", "a volleyball",
    "an orange fruit", "a flower", "a logo or trademark", "a map", "a person's face",
    "a screenshot with text", "a plant or grass", "an abstract pattern",
    "a circle or dot", "other object",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="raw_web_all")
    ap.add_argument("--model", default="runs/ball_v7/weights/best.onnx")
    ap.add_argument("--conf", type=float, default=0.45)
    ap.add_argument("--out", default="clip_check.jpg")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--clip-root", default="weights/clip")
    args = ap.parse_args()

    import torch
    import clip
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=dev, download_root=args.clip_root)
    with torch.no_grad():
        text = clip.tokenize(LABELS).to(dev)
        tfeat = model.encode_text(text)
        tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)

    det = Detector(args.model, 640, 0.25)
    src = Path(args.src)
    files = [p for p in sorted(src.rglob("*")) if p.suffix.lower() in IMG_EXTS]
    rows = []
    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        for d in det(img)[0]:
            if d["cls"] not in (0, 1) or d["conf"] < args.conf:
                continue
            x1, y1, x2, y2 = [int(v) for v in d["xyxy"]]
            mw, mh = int((x2 - x1) * 0.08), int((y2 - y1) * 0.08)
            crop = img[max(0, y1 - mh):min(h, y2 + mh), max(0, x1 - mw):min(w, x2 + mw)]
            if crop.size == 0:
                continue
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            from PIL import Image
            with torch.no_grad():
                f = model.encode_image(preprocess(Image.fromarray(rgb)).unsqueeze(0).to(dev))
                f = f / f.norm(dim=-1, keepdim=True)
                prob = (100 * f @ tfeat.T).softmax(dim=-1)[0].cpu().numpy()
            ball_p = float(prob[0] + prob[1])
            top = LABELS[int(prob.argmax())]
            rows.append({"file": p.name, "path": str(p), "cls": CLASS_NAMES[d["cls"]],
                         "conf": d["conf"], "ball_p": ball_p, "top": top,
                         "box": (x1, y1, x2, y2), "img": img})

    rows.sort(key=lambda r: -r["ball_p"])
    print(f"候选 {len(rows)} 个（来自 {len(files)} 张图）")
    for thr in (0.5, 0.7, 0.9, 0.95):
        n = sum(1 for r in rows if r["ball_p"] >= thr)
        print(f"  CLIP '是球' 概率 >= {thr:.2f}: {n} 个")
    print("\n概率最高的 5 个:")
    for r in rows[:5]:
        print(f"   {r['ball_p']:.3f} {r['cls']} {r['conf']:.2f} -> CLIP 认为: {r['top']}")
    print("概率最低的 5 个:")
    for r in rows[-5:]:
        print(f"   {r['ball_p']:.3f} {r['cls']} {r['conf']:.2f} -> CLIP 认为: {r['top']}")

    # 出图：按 CLIP 球概率从高到低取 n 张，若判别有效，前几张应全是真球
    tiles = []
    for r in rows[: args.n]:
        img = r["img"]
        x1, y1, x2, y2 = r["box"]
        pad = int(max(x2 - x1, y2 - y1) * 0.9)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        c = img[max(0, cy - pad):cy + pad, max(0, cx - pad):cx + pad]
        s = 300 / max(c.shape[:2])
        c = cv2.resize(c, (int(c.shape[1] * s), int(c.shape[0] * s)))
        cv2.rectangle(c, (int((x1 - (cx - pad)) * s), int((y1 - (cy - pad)) * s)),
                      (int((x2 - (cx - pad)) * s), int((y2 - (cy - pad)) * s)), (0, 255, 0), 2)
        cc = np.full((300, 300, 3), 25, np.uint8)
        cc[(300 - c.shape[0]) // 2:(300 - c.shape[0]) // 2 + c.shape[0],
           (300 - c.shape[1]) // 2:(300 - c.shape[1]) // 2 + c.shape[1]] = c
        cv2.putText(cc, f"ballP{r['ball_p']:.2f} {r['top'][:16]}", (3, 292),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1)
        tiles.append(cc)
    if tiles:
        while len(tiles) % 4:
            tiles.append(np.full((300, 300, 3), 25, np.uint8))
        cv2.imwrite(args.out, np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"已写出 {args.out}（概率最高的 {len(tiles)} 个）")


if __name__ == "__main__":
    main()
