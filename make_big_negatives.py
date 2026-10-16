# -*- coding: utf-8 -*-
"""
针对性造"大地板块"负样本（治 user 说的"地板被框成一大块排球"）：
  A) 低阈值(0.15)挖误检 + CLIP 门槛，**每个误检裁 3 种尺度**(1.0/1.5/2.2) -> 覆盖"一大块"
  B) 直接造大尺度地板负样本: 512~640 的低饱和地板裁剪
  C) 顺便测试: 当前模型在**大尺寸**地板块(640)上的误检率 —— 复现"一大块"问题
"""
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")
import cv2
import numpy as np
from ultralytics import YOLO

OUT = Path("hard_neg3")
PROMPT = ["a basketball", "a volleyball", "a floor or wooden surface", "a wall",
          "a line marking", "a shoe", "a person", "a cable or rope", "a shadow",
          "a bag or cloth", "other object"]

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


for sp in ("train", "val"):
    (OUT / "images" / sp).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / sp).mkdir(parents=True, exist_ok=True)

m = YOLO("runs/ball_2c/weights/best.pt")
random.seed(2)
frames = []
for d in ("raw_v19", "raw_v20", "raw_v21", "raw_v17", "raw_v18" if Path("raw_v18").exists() else "raw_v17"):
    frames += sorted(Path(d).glob("*.jpg"))
frames = list(dict.fromkeys(frames))
random.shuffle(frames)
print(f"候选帧 {len(frames)}")

made = 0
sizes = Counter()
topos = Counter()
tiles = []
# A) 挖误检，每框多尺度
for p in frames:
    if made >= 600:
        break
    img = cv2.imread(str(p))
    if img is None:
        continue
    h, w = img.shape[:2]
    r = m.predict(str(p), conf=0.15, iou=0.45, imgsz=640, device=0, verbose=False)[0]
    for b in r.boxes:
        if made >= 600:
            break
        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
        bw_, bh_ = x2 - x1, y2 - y1
        for scale in (1.0, 1.5, 2.2):
            cw = int(max(256, min(720, bw_ * scale)))
            ch = int(max(256, min(720, bh_ * scale)))
            if cw > w or ch > h:
                continue
            X1 = int(max(0, min(w - cw, (x1 + x2) / 2 - cw / 2)))
            Y1 = int(max(0, min(h - ch, (y1 + y2) / 2 - ch / 2)))
            crop = img[Y1:Y1 + ch, X1:X1 + cw]
            if crop.size < 200 * 200:
                continue
            pb, top = p_ball(crop)
            if pb >= 0.30:
                continue
            sp = "val" if made % 8 == 0 else "train"
            cv2.imwrite(str(OUT / "images" / sp / f"hneg_{made:05d}.jpg"), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            (OUT / "labels" / sp / f"hneg_{made:05d}.txt").write_text("", encoding="utf-8")
            made += 1
            frac = (bw_ * bh_) / (w * h)
            sizes["大(>8%)" if frac > 0.08 else ("中(2~8%)" if frac > 0.02 else "小(<2%)")] += 1
            topos[top] += 1
            if len(tiles) < 16:
                t = cv2.resize(crop, (180, 180))
                cv2.putText(t, f"{top[:14]} {pb:.2f} {int(scale*100)}%", (3, 174),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1)
                tiles.append(t)

# B) 大尺度纯地板负样本（低饱和 + CLIP 也是地板）
extra = 0
for p in frames:
    if extra >= 260:
        break
    img = cv2.imread(str(p))
    if img is None:
        continue
    h, w = img.shape[:2]
    for _ in range(3):
        if extra >= 260:
            break
        size = random.choice([448, 512, 576, 640])
        if size > w or size > h:
            continue
        x = random.randint(0, w - size)
        y = random.randint(0, h - size)
        crop = img[y:y + size, x:x + size]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        sat = (((hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 60)).sum()) / (size * size)
        if sat > 0.10:
            continue                      # 高饱和 = 可能有球
        pb, top = p_ball(crop)
        if pb >= 0.30:
            continue
        sp = "train" if extra % 5 else "val"
        cv2.imwrite(str(OUT / "images" / sp / f"floor_big_{extra:04d}.jpg"), crop,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        (OUT / "labels" / sp / f"floor_big_{extra:04d}.txt").write_text("", encoding="utf-8")
        extra += 1
        topos[top] += 1

if tiles:
    while len(tiles) % 4:
        tiles.append(np.full((180, 180, 3), 25, np.uint8))
    cv2.imwrite("hard_neg3_examples.jpg",
                np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]),
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("样例 -> hard_neg3_examples.jpg")
print(f"负样本: 误检挖掘 {made} + 大尺度地板 {extra} = {made + extra}")
print("  误检尺寸分布:", dict(sizes))
print("  CLIP 认为是什么:", dict(sorted(topos.items(), key=lambda x: -x[1])))
print(f"  大尺寸地板负样本尺寸: 448/512/576/640（{extra} 张）")
