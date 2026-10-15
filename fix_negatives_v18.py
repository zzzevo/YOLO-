# -*- coding: utf-8 -*-
"""修正"负样本"标签：
   之前把"检测器没检出"的帧当成空场 -> 其实里面有球(漏检) -> 空标签是有害的。
   现在用 v17 + CLIP 重新标这些帧，只有**真的没有球**的才留作负样本。
然后构建 v18 = v17（去掉错误负样本、补上正确标注）+ 真负样本。
"""
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")
import cv2
import numpy as np
from infer_onnx import CLASS_NAMES, Detector

LABELS = ["a basketball", "a volleyball", "an orange fruit", "a flower",
          "a logo or trademark", "a map", "a person's face", "a screenshot with text",
          "a plant or grass", "an abstract pattern", "a circle or dot", "other object"]

old = Path("dataset_ball_v17")
new = Path("dataset_ball_v18")
SRC = Path("raw_v17")
LAB = Path("v17_labels/labels")

import torch
import clip
from PIL import Image
dev = "cuda" if torch.cuda.is_available() else "cpu"
cmodel, cprep = clip.load("ViT-B/32", device=dev, download_root="weights/clip")
with torch.no_grad():
    tf = cmodel.encode_text(clip.tokenize(LABELS).to(dev))
    tf = tf / tf.norm(dim=-1, keepdim=True)


def clip_cls(crop):
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        f = cmodel.encode_image(cprep(Image.fromarray(rgb)).unsqueeze(0).to(dev))
        f = f / f.norm(dim=-1, keepdim=True)
        p = (100 * f @ tf.T).softmax(dim=-1)[0].cpu().numpy()
    return float(p[0]), float(p[1])


det = Detector("runs/ball_v17/weights/best.onnx", 640, 0.15)

# 找出 v17 里被当成负样本(空标签)的帧
neg_frames = []
for sp in ("train", "val"):
    for lab in (old / "labels" / sp).glob("*.txt"):
        if not lab.read_text(encoding="utf-8").strip():
            neg_frames.append((sp, lab.stem))
print(f"原负样本帧: {len(neg_frames)}")

fixed, still_empty = 0, 0
for sp, stem in neg_frames:
    imgp = SRC / f"{stem}.jpg"
    img = cv2.imread(str(imgp))
    if img is None:
        continue
    h, w = img.shape[:2]
    dets = [d for d in det(img)[0] if d["cls"] in (0, 1) and d["conf"] >= 0.3]
    lines = []
    for d in dets:
        x1, y1, x2, y2 = [int(v) for v in d["xyxy"]]
        pad = int(max(x2 - x1, y2 - y1) * 0.08)
        crop = img[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)]
        if crop.size == 0:
            continue
        pb, pv = clip_cls(crop)
        if pb + pv < 0.9:
            continue
        c = 0 if pb >= pv else 1
        lines.append(f"{c} {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}")
    (LAB / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
    if lines:
        fixed += 1
    else:
        still_empty += 1
print(f"重新标注后: 补上标注 {fixed} 帧 / 确实无球 {still_empty} 帧")

# 重建：v17 的非负样本帧照搬；负样本帧用刚修好的标签
if new.exists():
    shutil.rmtree(new)
for sp in ("train", "val"):
    (new / "images" / sp).mkdir(parents=True, exist_ok=True)
    (new / "labels" / sp).mkdir(parents=True, exist_ok=True)
negset = {s for _, s in neg_frames}
cnt = Counter()
for sp in ("train", "val"):
    for p in (old / "images" / sp).glob("*.jpg"):
        if p.stem in negset:
            continue
        l = old / "labels" / sp / (p.stem + ".txt")
        if l.exists():
            shutil.copyfile(p, new / "images" / sp / p.name)
            shutil.copyfile(l, new / "labels" / sp / l.name)
            cnt["原样本"] += 1
    for stem in [s for sp2, s in neg_frames if sp2 == sp]:
        src = SRC / f"{stem}.jpg"
        if not src.exists():
            continue
        shutil.copyfile(src, new / "images" / sp / f"{stem}.jpg")
        shutil.copyfile(LAB / f"{stem}.txt", new / "labels" / sp / f"{stem}.txt")
        cnt["修正帧"] += 1
(new / "data.yaml").write_text(
    "# v17 修正版：把'漏检当空场'的错误负样本重新标注（有害标签），只留真无球帧\n"
    f"path: {new.resolve().as_posix()}\ntrain: images/train\nval: images/val\n"
    "names:\n  0: 篮球\n  1: 排球\n  2: 定位柱\n", encoding="utf-8")
print(f"v18: {dict(cnt)} | train {len(list((new/'images'/'train').glob('*.jpg')))} "
      f"val {len(list((new/'images'/'val').glob('*.jpg')))}")
