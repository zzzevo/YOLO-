# -*- coding: utf-8 -*-
"""
建纯 2 类数据集（篮球/排球，**去掉定位柱**）+ 治地板误检。

步骤:
  A) v20/v21（视频5/6/7，人工确认都是排球）自动标注：v18 检测 + "像不像球"过滤 + 强制排球
     —— 同时把"模型说篮球但置信度高"的帧列出来供核对
  B) 把 dataset_ball_v18（3类）转 2 类：删掉定位柱标签；因此变空的帧当作负样本（它们本来没球）
  C) 合入 v19(4段排球) + v20/v21
  D) **生成纯地板负样本**：从帧里裁不含任何标注框的地板小块，单独作为空标签图片
     （这是干净的真负样本，专门治"木地板被认成球"）
"""
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")
import cv2
import numpy as np
from infer_onnx import Detector

BASE = Path("dataset_ball_v18")
OUT = Path("dataset_ball_2c")
V19 = Path("v19_labels/labels")
N_FLOOR_NEG = 320
random.seed(5)


def ball_like(img, box):
    x1, y1, x2, y2 = [int(max(0, v)) for v in box]
    crop = img[y1:y2, x1:x2]
    if crop.size < 400:
        return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    sat = ((hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 60)).astype(np.uint8)
    if sat.sum() / sat.size < 0.15:
        return False
    sat = cv2.morphologyEx(sat * 255, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(sat, 8)
    if n <= 1:
        return False
    big = 1 + int(np.argmax(stats[1:, 4]))
    x, y, bw, bh, area = stats[big]
    if area < 0.25 * crop.shape[0] * crop.shape[1]:
        return False
    if area / (bw * bh + 1e-6) < 0.5 or not (0.5 < bw / (bh + 1e-6) < 2.0):
        return False
    return True


# ---------- A) 标注 v20 / v21 ----------
det = Detector("runs/ball_v18/weights/best.onnx", 640, 0.2)
for src_dir, out_lab in (("raw_v20", "v20_labels"), ("raw_v21", "v21_labels")):
    src, lab_out = Path(src_dir), Path(out_lab) / "labels"
    lab_out.mkdir(parents=True, exist_ok=True)
    kept = dropped = 0
    hi_conf_basket = []
    for p in sorted(src.glob("*.jpg")):
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        lines = []
        for d in det(img)[0]:
            if d["cls"] not in (0, 1) or d["conf"] < 0.35:
                continue
            if not ball_like(img, d["xyxy"]):
                dropped += 1
                continue
            x1, y1, x2, y2 = d["xyxy"]
            # 视频5/6/7 人工确认都是排球 -> 强制 class 1；但把"模型高置信说篮球"的记下来
            if int(d["cls"]) == 0 and d["conf"] >= 0.6:
                hi_conf_basket.append((p.name, round(float(d["conf"]), 2)))
            lines.append(f"1 {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}")
            kept += 1
        (lab_out / (p.stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")
    print(f"{src_dir}: 保留 {kept} 框（强制排球）/ 不像球剔除 {dropped}")
    if hi_conf_basket:
        print(f"   ⚠️ 模型高置信(>=0.6)说是篮球的 {len(hi_conf_basket)} 个 -> 已强制改排球, 抽样核对: "
              f"{hi_conf_basket[:6]}")

# ---------- B/C) 组 2 类数据集 ----------
if OUT.exists():
    shutil.rmtree(OUT)
for sp in ("train", "val"):
    (OUT / "images" / sp).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / sp).mkdir(parents=True, exist_ok=True)
stat = Counter()
for sp in ("train", "val"):
    for lab in (BASE / "labels" / sp).glob("*.txt"):
        img = BASE / "images" / sp / (lab.stem + ".jpg")
        if not img.exists():
            continue
        out = []
        for l in lab.read_text(encoding="utf-8").splitlines():
            if l.strip() and l.split()[0] != "2":          # 去掉定位柱
                out.append(l)
                stat["原3类->2类"] += 1
        if out:
            shutil.copyfile(img, OUT / "images" / sp / img.name)
            (OUT / "labels" / sp / lab.name).write_text("\n".join(out), encoding="utf-8")
            stat["使用帧"] += 1
        elif random.random() < 0.25:                       # 只剩柱子的帧 -> 少量当负样本
            shutil.copyfile(img, OUT / "images" / sp / img.name)
            (OUT / "labels" / sp / lab.name).write_text("", encoding="utf-8")
            stat["原柱帧当负样本"] += 1

# v19（4 段排球，已修）
n19 = 0
for lab in sorted(V19.glob("*.txt")):
    txt = lab.read_text(encoding="utf-8").strip()
    if not txt:
        continue
    sp = "val" if n19 % 5 == 0 else "train"
    shutil.copyfile(Path("raw_v19") / (lab.stem + ".jpg"), OUT / "images" / sp / (lab.stem + ".jpg"))
    shutil.copyfile(lab, OUT / "labels" / sp / lab.name)
    n19 += 1
stat["v19排球帧"] = n19
# v20 / v21
for src_lab, prefix in (("v20_labels/labels", "raw_v20"), ("v21_labels/labels", "raw_v21")):
    n = 0
    for lab in sorted(Path(src_lab).glob("*.txt")):
        if not lab.read_text(encoding="utf-8").strip():
            continue
        sp = "val" if n % 5 == 0 else "train"
        shutil.copyfile(Path(prefix) / (lab.stem + ".jpg"), OUT / "images" / sp / (lab.stem + ".jpg"))
        shutil.copyfile(lab, OUT / "labels" / sp / lab.name)
        n += 1
    stat[f"{prefix}帧"] = n

# ---------- D) 纯地板负样本 ----------
pool = []
for d in ("raw_v19", "raw_v20", "raw_v21", "raw_v17"):
    for p in sorted(Path(d).glob("*.jpg")):
        pool.append(p)
random.shuffle(pool)
made = 0
PATCH = 384
for p in pool:
    if made >= N_FLOOR_NEG:
        break
    img = cv2.imread(str(p))
    if img is None:
        continue
    h, w = img.shape[:2]
    # 用该帧的标签（可能在 v19_labels / v20_labels / v21_labels / v17_labels 里）
    boxes = []
    for cand in (V19, Path("v20_labels/labels"), Path("v21_labels/labels"), Path("v17_labels/labels")):
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
        if w < PATCH or h < PATCH:
            break
        x = random.randint(0, w - PATCH)
        y = random.randint(0, h - PATCH)
        ok = True
        for cx, cy, bw, bh in boxes:
            if not (x + PATCH < cx - bw / 2 - 20 or x > cx + bw / 2 + 20 or
                    y + PATCH < cy - bh / 2 - 20 or y > cy + bh / 2 + 20):
                ok = False
                break
        if not ok:
            continue
        crop = img[y:y + PATCH, x:x + PATCH]
        sp = "train" if made % 5 else "val"
        name = f"floor_{made:04d}"
        cv2.imwrite(str(OUT / "images" / sp / f"{name}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (OUT / "labels" / sp / f"{name}.txt").write_text("", encoding="utf-8")
        made += 1
stat["纯地板负样本"] = made

(OUT / "data.yaml").write_text(
    "# 纯 2 类（篮球/排球，无定位柱）+ 新排球视频 + 纯地板负样本\n"
    f"path: {OUT.resolve().as_posix()}\ntrain: images/train\nval: images/val\n"
    "names:\n  0: 篮球\n  1: 排球\n", encoding="utf-8")
print("\n数据集统计:", dict(stat))
print(f"train {len(list((OUT/'images'/'train').glob('*.jpg')))} / "
      f"val {len(list((OUT/'images'/'val').glob('*.jpg')))}")
c = Counter()
for sp in ("train", "val"):
    for t in (OUT / "labels" / sp).glob("*.txt"):
        for l in t.read_text(encoding="utf-8").splitlines():
            if l.strip():
                c["篮球" if l.split()[0] == "0" else "排球"] += 1
print("类别分布:", dict(c))
