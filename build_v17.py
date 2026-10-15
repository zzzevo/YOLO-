# -*- coding: utf-8 -*-
"""构建 v17 数据集：
   = v16 的数据（真实+合成）
   + 新视频 455 帧（CLIP 定类，已修 1 个误判）
   + **纯地板负样本**：新视频里"什么都没检出"的帧（空标签）——专门教模型"地板不是球"
"""
import random
import shutil
from pathlib import Path

from infer_onnx import Detector

SRC = Path("raw_v17")
LAB = Path("v17_labels/labels")
BASE = Path("dataset_ball_v16")
OUT = Path("dataset_ball_v17")
N_NEG = 90          # 负样本上限（别太多，免得把"没球"学过头）

# 1) 修掉视频3 里那个被 CLIP 误判成排球的框
n_fix = 0
lab = LAB / "n3__001128.txt"
if lab.exists():
    lines = []
    for l in lab.read_text(encoding="utf-8").splitlines():
        if l.strip() and l.split()[0] == "1":
            lines.append("0 " + " ".join(l.split()[1:]))
            n_fix += 1
        elif l.strip():
            lines.append(l)
    lab.write_text("\n".join(lines), encoding="utf-8")
print(f"修正误判: {n_fix} 个框（排球->篮球）")

# 2) 找"纯地板"帧（整体无检出）
has_lab = {p.stem for p in LAB.glob("*.txt") if p.read_text(encoding="utf-8").strip()}
empties = [p for p in sorted(LAB.glob("*.txt")) if p.stem not in has_lab or not p.read_text(encoding="utf-8").strip()]
print(f"无检出帧: {len(empties)}")

# 3) 组数据集
if OUT.exists():
    shutil.rmtree(OUT)
for sp in ("train", "val"):
    (OUT / "images" / sp).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / sp).mkdir(parents=True, exist_ok=True)
for p in (BASE / "images" / "train").glob("*.jpg"):
    l = BASE / "labels" / "train" / (p.stem + ".txt")
    if l.exists():
        shutil.copyfile(p, OUT / "images" / "train" / p.name)
        shutil.copyfile(l, OUT / "labels" / "train" / l.name)
for p in (BASE / "images" / "val").glob("*.jpg"):
    l = BASE / "labels" / "val" / (p.stem + ".txt")
    if l.exists():
        shutil.copyfile(p, OUT / "images" / "val" / p.name)
        shutil.copyfile(l, OUT / "labels" / "val" / l.name)

# 新帧：80% train / 20% val
random.seed(3)
news = sorted(LAB.glob("*.txt"))
random.shuffle(news)
n_val = int(len(news) * 0.2)
n_ball = n_neg = 0
for i, l in enumerate(news):
    txt = l.read_text(encoding="utf-8").strip()
    if not txt:
        continue          # 空标签的帧走下面的负样本逻辑
    sp = "val" if i < n_val else "train"
    shutil.copyfile(SRC / (l.stem + ".jpg"), OUT / "images" / sp / (l.stem + ".jpg"))
    shutil.copyfile(l, OUT / "labels" / sp / l.name)
    n_ball += 1

# 负样本（空标签）：一半 train 一半 val
random.shuffle(empties)
for i, l in enumerate(empties[:N_NEG]):
    sp = "train" if i % 2 == 0 else "val"
    shutil.copyfile(SRC / (l.stem + ".jpg"), OUT / "images" / sp / (l.stem + ".jpg"))
    (OUT / "labels" / sp / l.name).write_text("", encoding="utf-8")
    n_neg += 1

(OUT / "data.yaml").write_text(
    "# v16 全部数据 + 3 段新视频（CLIP 定类）+ 纯地板负样本\n"
    f"path: {OUT.resolve().as_posix()}\ntrain: images/train\nval: images/val\n"
    "names:\n  0: 篮球\n  1: 排球\n  2: 定位柱\n", encoding="utf-8")
print(f"v17: 新视频带标注帧 {n_ball} + 纯地板负样本 {n_neg}")
print(f"  train {len(list((OUT/'images'/'train').glob('*.jpg')))} / "
      f"val {len(list((OUT/'images'/'val').glob('*.jpg')))}")
