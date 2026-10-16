# -*- coding: utf-8 -*-
"""
针对"白线 + 木地板角落被框成一大块排球"专门造负样本。

思路：误检现场的共同特征是"木地板 + 白色场地线"。所以直接大量采集这种图案的地板裁剪
（低饱和木色 + 含相当比例白色像素 + CLIP 也说不是球 + 不含任何真值框），作为空标签负样本。
比只挖到 9 个误检样本有效得多。
"""
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")
import cv2
import numpy as np

PROMPT = ["a basketball", "a volleyball", "a floor or wooden surface", "a white line",
          "a wall", "a shoe", "a shadow", "a bag or cloth", "other object"]
OUT = Path("hard_neg_line")
TARGET = 420
random.seed(9)

import torch
import clip
from PIL import Image
dev = "cuda" if torch.cuda.is_available() else "cpu"
cmodel, cprep = clip.load("ViT-B/32", device=dev, download_root="weights/clip")
with torch.no_grad():
    tf = cmodel.encode_text(clip.tokenize(PROMPT).to(dev))
    tf = tf / tf.norm(dim=-1, keepdim=True)


def p_ball(crop):
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        f = cmodel.encode_image(cprep(Image.fromarray(rgb)).unsqueeze(0).to(dev))
        f = f / f.norm(dim=-1, keepdim=True)
        p = (100 * f @ tf.T).softmax(dim=-1)[0].cpu().numpy()
    return float(p[0] + p[1]), PROMPT[int(p.argmax())]


LABDIRS = [Path("v19_labels/labels"), Path("v20_labels/labels"), Path("v21_labels/labels"),
           Path("v17_labels/labels"), Path("dataset_ball_2c/labels/val"),
           Path("dataset_ball_2c/labels/train"), Path("hard_neg_v22/labels/train"),
           Path("hard_neg_v22/labels/val")]
for sp in ("train", "val"):
    (OUT / "images" / sp).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / sp).mkdir(parents=True, exist_ok=True)


def boxes_of(stem, w, h):
    for d in LABDIRS:
        f = d / (stem + ".txt")
        if f.exists():
            out = []
            for l in f.read_text(encoding="utf-8").splitlines():
                if l.strip():
                    c, cx, cy, bw, bh = l.split()
                    out.append(((float(cx) - float(bw) / 2) * w, (float(cy) - float(bh) / 2) * h,
                                (float(cx) + float(bw) / 2) * w, (float(cy) + float(bh) / 2) * h))
            return out
    return []


frames = []
for d in ("raw_v22", "raw_v19", "raw_v20", "raw_v21", "raw_v17"):
    frames += sorted(Path(d).glob("*.jpg"))
random.shuffle(frames)
print(f"候选帧 {len(frames)}")

made = 0
topos = Counter()
tiles = []
for p in frames:
    if made >= TARGET:
        break
    img = cv2.imread(str(p))
    if img is None:
        continue
    h, w = img.shape[:2]
    bs = boxes_of(p.stem, w, h)
    for _ in range(10):
        if made >= TARGET:
            break
        size = random.choice([320, 384, 448, 512, 576, 640])
        if size > w or size > h:
            continue
        x = random.randint(0, w - size)
        y = random.randint(0, h - size)
        if any((x + size > bx1 - 15 and x < bx2 + 15 and y + size > by1 - 15 and y < by2 + 15)
               for bx1, by1, bx2, by2 in bs):
            continue                                   # 含有真球，跳过
        crop = img[y:y + size, x:x + size]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        sat = ((S > 80) & (V > 60))
        if sat.sum() / sat.size > 0.08:
            continue                                   # 高饱和 = 可能有球/彩色物
        white = ((V > 165) & (S < 70))                 # 白色场地线
        wfrac = white.sum() / white.size
        wood = ((H > 5) & (H < 30) & (S > 30) & (S < 160) & (V > 80))
        woodfrac = wood.sum() / wood.size
        # 只要"木地板 + 白线"这一类图案
        if woodfrac < 0.35:
            continue
        pb, top = p_ball(crop)
        if pb >= 0.25:
            continue
        sp = "val" if made % 9 == 0 else "train"
        cv2.imwrite(str(OUT / "images" / sp / f"line_{made:04d}.jpg"), crop,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        (OUT / "labels" / sp / f"line_{made:04d}.txt").write_text("", encoding="utf-8")
        made += 1
        topos[f"白线{wfrac*100:.0f}% 木纹{woodfrac*100:.0f}%"] += 1
        if len(tiles) < 12:
            t = cv2.resize(crop, (180, 180))
            cv2.putText(t, f"线{wfrac*100:.0f}% 木{woodfrac*100:.0f}% {pb:.2f}", (3, 174),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1)
            tiles.append(t)

if tiles:
    while len(tiles) % 4:
        tiles.append(np.full((180, 180, 3), 25, np.uint8))
    cv2.imwrite("hard_neg_line_examples.jpg",
                np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]),
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("样例 -> hard_neg_line_examples.jpg")
print(f"造出'白线+地板'负样本 {made} 个")
print("  典型构成:", dict(list(topos.items())[:6]))
