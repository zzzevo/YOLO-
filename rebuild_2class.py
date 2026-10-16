# -*- coding: utf-8 -*-
"""用**已修正好的标签**（v19/v20/v21 都过了形状+CLIP 双重检查）重建纯 2 类数据集。
   = dataset_ball_v18 去掉定位柱
   + v19/v20/v21（新排球视频，标签已核对）
   + 纯地板负样本
"""
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2

BASE = Path("dataset_ball_v18")
OUT = Path("dataset_ball_2c")
N_FLOOR_NEG = 320
random.seed(5)

if OUT.exists():
    shutil.rmtree(OUT)
for sp in ("train", "val"):
    (OUT / "images" / sp).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / sp).mkdir(parents=True, exist_ok=True)

stat = Counter()
# 1) 3类 -> 2类
for sp in ("train", "val"):
    for lab in (BASE / "labels" / sp).glob("*.txt"):
        img = BASE / "images" / sp / (lab.stem + ".jpg")
        if not img.exists():
            continue
        out = [l for l in lab.read_text(encoding="utf-8").splitlines()
               if l.strip() and l.split()[0] != "2"]
        if out:
            shutil.copyfile(img, OUT / "images" / sp / img.name)
            (OUT / "labels" / sp / lab.name).write_text("\n".join(out), encoding="utf-8")
            stat["v18转2类_帧"] += 1
            stat["v18转2类_框"] += len(out)
        elif random.random() < 0.25:
            shutil.copyfile(img, OUT / "images" / sp / img.name)
            (OUT / "labels" / sp / lab.name).write_text("", encoding="utf-8")
            stat["原柱帧当负样本"] += 1

# 2) 新排球视频（标签已核对）
for src_dir, lab_dir, tag in (("raw_v19", "v19_labels/labels", "v19"),
                              ("raw_v20", "v20_labels/labels", "v20"),
                              ("raw_v21", "v21_labels/labels", "v21")):
    n = 0
    for lab in sorted(Path(lab_dir).glob("*.txt")):
        if not lab.read_text(encoding="utf-8").strip():
            continue
        sp = "val" if n % 5 == 0 else "train"
        shutil.copyfile(Path(src_dir) / (lab.stem + ".jpg"), OUT / "images" / sp / (lab.stem + ".jpg"))
        shutil.copyfile(lab, OUT / "labels" / sp / lab.name)
        n += 1
        stat[f"{tag}_框"] += len([l for l in lab.read_text(encoding="utf-8").splitlines() if l.strip()])
    stat[f"{tag}_帧"] = n

# 3) 纯地板负样本
pool = [p for d in ("raw_v19", "raw_v20", "raw_v21", "raw_v17") for p in sorted(Path(d).glob("*.jpg"))]
random.shuffle(pool)
LABDIRS = [Path("v19_labels/labels"), Path("v20_labels/labels"),
           Path("v21_labels/labels"), Path("v17_labels/labels")]
PATCH = 384
made = 0
for p in pool:
    if made >= N_FLOOR_NEG:
        break
    img = cv2.imread(str(p))
    if img is None:
        continue
    h, w = img.shape[:2]
    if w < PATCH or h < PATCH:
        continue
    boxes = []
    for cand in LABDIRS:
        f = cand / (p.stem + ".txt")
        if f.exists():
            for l in f.read_text(encoding="utf-8").splitlines():
                if l.strip():
                    c, cx, cy, bw, bh = l.split()
                    boxes.append((float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h))
            break
    for _ in range(6):
        if made >= N_FLOOR_NEG:
            break
        x = random.randint(0, w - PATCH)
        y = random.randint(0, h - PATCH)
        ok = all((x + PATCH < cx - bw / 2 - 20 or x > cx + bw / 2 + 20 or
                  y + PATCH < cy - bh / 2 - 20 or y > cy + bh / 2 + 20)
                 for cx, cy, bw, bh in boxes)
        if not ok:
            continue
        # 地板块必须**低饱和**（木地板是低饱和暖色；球一定高饱和）——避免把球裁进来
        crop = img[y:y + PATCH, x:x + PATCH]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        satfrac = (((hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 60)).sum()) / (PATCH * PATCH)
        if satfrac > 0.12:
            continue
        sp = "train" if made % 5 else "val"
        cv2.imwrite(str(OUT / "images" / sp / f"floor_{made:04d}.jpg"), crop,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        (OUT / "labels" / sp / f"floor_{made:04d}.txt").write_text("", encoding="utf-8")
        made += 1
stat["纯地板负样本"] = made

(OUT / "data.yaml").write_text(
    "# 纯 2 类（篮球/排球，无定位柱）+ 新排球视频 + 纯地板负样本\n"
    f"path: {OUT.resolve().as_posix()}\ntrain: images/train\nval: images/val\n"
    "names:\n  0: 篮球\n  1: 排球\n", encoding="utf-8")
print("统计:", dict(stat))
print(f"train {len(list((OUT/'images'/'train').glob('*.jpg')))} / "
      f"val {len(list((OUT/'images'/'val').glob('*.jpg')))}")
c = Counter()
for sp in ("train", "val"):
    for t in (OUT / "labels" / sp).glob("*.txt"):
        for l in t.read_text(encoding="utf-8").splitlines():
            if l.strip():
                c["篮球" if l.split()[0] == "0" else "排球"] += 1
print("类别分布:", dict(c))
