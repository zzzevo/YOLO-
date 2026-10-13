#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
合成训练数据：把"球/定位柱"抠出来，贴到**你自己场地的空场背景**上，标签由构造过程精确决定。

为什么这招比继续爬网图更值：
  * 网图补的是"外观多样性"，但你自家场地的**光照/地板/相机角度/目标尺度**分布它给不了；
  * 合成数据的标签是**构造出来的**，不存在标错风险（爬图最大的坑）；
  * 数量不受限：想造多少造多少，还能精确控制"远距离小球"的占比（这是之前最难的一档）。

做法:
  1) 抠图源：从已有数据集里取标注框 -> 裁出球(用内切圆遮罩+羽化) / 定位柱(矩形遮罩)
  2) 背景：从你的视频帧里筛**没有任何标注**的帧（空场地）——避免"背景里已有球却没标签"
  3) 随机贴 1~4 个目标：位置随机、尺寸按目标像素高度分布抽样（刻意多造远距离小目标）、
     可选水平翻转/轻微旋转、边缘羽化 + 轻微噪声，让接缝不那么"贴纸"
  4) 标签：直接由粘贴位置算出（精确）

用法:
  python synth_paste.py --data dataset_ball_v15 --bg raw_v7 raw_bb2 raw_pillar_far \
      --out dataset_synth --n 1500
"""
import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

from infer_onnx import CLASS_NAMES


def load_crops(root, split="train"):
    """从数据集里抽取 (类别, 裁剪图, 遮罩类型)"""
    balls, pillars = [], []
    for lab in sorted((root / "labels" / split).glob("*.txt")):
        imgp = root / "images" / split / (lab.stem + ".jpg")
        img = cv2.imread(str(imgp))
        if img is None:
            continue
        h, w = img.shape[:2]
        for l in lab.read_text(encoding="utf-8").splitlines():
            if not l.strip():
                continue
            c, cx, cy, bw, bh = l.split()
            c = int(float(c))
            x1, y1 = int((float(cx) - float(bw) / 2) * w), int((float(cy) - float(bh) / 2) * h)
            x2, y2 = int((float(cx) + float(bw) / 2) * w), int((float(cy) + float(bh) / 2) * h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 < 24 or y2 - y1 < 24:
                continue
            crop = img[y1:y2, x1:x2].copy()
            if c == 2:
                if 20 <= crop.shape[0] <= 700 and 12 <= crop.shape[1] <= 400:
                    pillars.append(crop)
            else:
                if 20 <= min(crop.shape[:2]) and max(crop.shape[:2]) <= 600:
                    balls.append((c, crop))
    return balls, pillars


def ball_mask(crop):
    """球用内切椭圆遮罩 + 羽化（去掉矩形背景，贴上去才不像贴纸）"""
    h, w = crop.shape[:2]
    m = np.zeros((h, w), np.float32)
    cv2.ellipse(m, (w // 2, h // 2), (int(w * 0.47), int(h * 0.47)), 0, 0, 360, 1.0, -1)
    k = max(3, int(min(h, w) * 0.10) | 1)
    m = cv2.GaussianBlur(m, (k, k), 0)
    return m


def rect_mask(crop):
    h, w = crop.shape[:2]
    m = np.ones((h, w), np.float32)
    k = max(3, int(min(h, w) * 0.08) | 1)
    inner = np.zeros((h, w), np.float32)
    pad = max(1, int(min(h, w) * 0.06))
    inner[pad:h - pad, pad:w - pad] = 1.0
    return cv2.GaussianBlur(inner, (k, k), 0) * 0.98 + m * 0.02


def paste(bg, crop, mask, x, y):
    """把 crop 按 mask 贴到 bg 的 (x,y)，返回实际落点"""
    bh, bw = bg.shape[:2]
    ch, cw = crop.shape[:2]
    x = int(np.clip(x, 0, bw - cw))
    y = int(np.clip(y, 0, bh - ch))
    roi = bg[y:y + ch, x:x + cw].astype(np.float32)
    m3 = mask[:, :, None]
    bg[y:y + ch, x:x + cw] = (roi * (1 - m3) + crop.astype(np.float32) * m3).astype(np.uint8)
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_ball_v15", help="抠图来源数据集（要带 labels）")
    ap.add_argument("--bg", nargs="+", default=["raw_v7", "raw_bb2", "raw_pillar_far"],
                    help="背景来源目录（会筛掉有标注的帧）")
    ap.add_argument("--out", default="dataset_synth")
    ap.add_argument("--n", type=int, default=1200, help="生成多少张")
    ap.add_argument("--per-img", type=int, default=3, help="每张最多贴几个目标")
    ap.add_argument("--small-bias", type=float, default=0.45,
                    help="0~1：越大越多造'远距离小目标'（目标高度占比小）")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    root = Path(args.data)
    balls, pillars = load_crops(root)
    print(f"抠图源: 球 {len(balls)} 个, 定位柱 {len(pillars)} 个")
    if not balls and not pillars:
        raise SystemExit("没抠到东西，检查 --data")

    # 背景：只用没有标注的帧（空场地），避免"背景里已有目标却没标签"
    bg_imgs = []
    for d in args.bg:
        for p in sorted(Path(d).glob("*.jpg")):
            if (root / "labels" / "train" / (p.stem + ".txt")).exists():
                continue
            bg_imgs.append(p)
    # 也收集数据集 val 里没有标注的帧
    for p in sorted((root / "images" / "val").glob("*.jpg")):
        if not (root / "labels" / "val" / (p.stem + ".txt")).exists():
            bg_imgs.append(p)
    if not bg_imgs:
        raise SystemExit("没有可用背景（都是带标注的帧）")
    print(f"空场背景: {len(bg_imgs)} 张（来自 {'、'.join(args.bg)} 等）")

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    for sp in ("train", "val"):
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sp).mkdir(parents=True, exist_ok=True)

    n_val = int(args.n * args.val_ratio)
    made = 0
    for i in range(args.n):
        bgp = rng.choice(bg_imgs)
        bg = cv2.imread(str(bgp))
        if bg is None:
            continue
        H, W = bg.shape[:2]
        labels = []
        for _ in range(rng.randint(1, args.per_img)):
            use_pillar = pillars and rng.random() < 0.18
            if use_pillar:
                crop = rng.choice(pillars).copy()
                mask = rect_mask(crop)
                cid = 2
            else:
                cid, crop0 = rng.choice(balls)
                crop = crop0.copy()
                mask = ball_mask(crop)
            # 目标高度占比：小球偏置（模拟远距离）
            if rng.random() < args.small_bias:
                frac = rng.uniform(0.02, 0.12)
            else:
                frac = rng.uniform(0.12, 0.55)
            target_h = max(12, int(H * frac))
            scale = target_h / crop.shape[0]
            target_w = max(8, int(crop.shape[1] * scale))
            # 必须完整落在画面内，否则贴图会被裁掉而标签仍按完整框算 -> 标签与画面不符
            if target_w > W * 0.85 or target_h > H * 0.85:
                continue
            crop = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            if rng.random() < 0.5:
                crop = cv2.flip(crop, 1)
                mask = cv2.flip(mask, 1)
            # 轻微颜色抖动（光照差异）
            if rng.random() < 0.5:
                a = rng.uniform(0.85, 1.15)
                crop = np.clip(crop.astype(np.float32) * a, 0, 255).astype(np.uint8)
            x = rng.randint(0, max(0, W - target_w))
            y = rng.randint(0, max(0, H - target_h))
            px, py = paste(bg, crop, mask, x, y)
            labels.append(f"{cid} {(px + target_w/2)/W:.6f} {(py + target_h/2)/H:.6f} "
                          f"{target_w/W:.6f} {target_h/H:.6f}")
        if not labels:
            continue
        sp = "val" if made < n_val else "train"
        name = f"syn_{made:05d}"
        cv2.imwrite(str(out / "images" / sp / f"{name}.jpg"), bg, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (out / "labels" / sp / f"{name}.txt").write_text("\n".join(labels), encoding="utf-8")
        made += 1

    (out / "data.yaml").write_text(
        f"# 合成数据（粘贴球/柱到自家空场背景），标签由构造精确决定\n"
        f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\n"
        "names:\n  0: 篮球\n  1: 排球\n  2: 定位柱\n", encoding="utf-8")
    print(f"\n[done] 生成 {made} 张 -> {out.resolve()}")
    print(f"  train {len(list((out/'images'/'train').glob('*.jpg')))} / "
          f"val {len(list((out/'images'/'val').glob('*.jpg')))}")
    print("  拼一张预览: python make_sheet.py --src " + str(out / "images" / "train") +
          " --out synth_preview.jpg --n 12")


if __name__ == "__main__":
    main()
