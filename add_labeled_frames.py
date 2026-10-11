#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
把"已经标好的图片+标签"导入数据集（可指定验证集比例、可改类名）。

和 add_web_frames.py 的区别：
  * add_web_frames.py 用模型现场检测生成标签（适合网图）
  * 本脚本直接用你已有的标签文件（适合用颜色/形状等自研方法标好的帧）

用法:
  # 在 dataset_ball_v4 基础上加"定位柱"这一类，并把它做成第 3 类(0=篮球,1=排球,2=定位柱)
  python add_labeled_frames.py --images raw_pillar --labels pillar_labels \
      --base dataset_ball_v4 --dst dataset_ball_v5 --val-ratio 0.2 \
      --names "篮球,排球,定位柱"

  # 加到已有数据集里（不复制）
  python add_labeled_frames.py --images X --labels Y --dst dataset_ball_v5 --no-base
"""
import argparse
import random
import shutil
from pathlib import Path

import cv2

from infer_onnx import IMG_EXTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True, help="与图片同名的 YOLO txt 目录")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--base", help="基础数据集，给了就先整体复制成 --dst")
    ap.add_argument("--names", required=True, help="类别名，逗号分隔，顺序即 id")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--prefix", default="", help="文件名前缀，避免与已有帧重名")
    ap.add_argument("--min-box-size", type=float, default=0.0, help="框最小面积占比，低于此丢弃")
    args = ap.parse_args()

    names = [s.strip() for s in args.names.replace("，", ",").split(",") if s.strip()]
    src_img, src_lab = Path(args.images), Path(args.labels)
    dst = Path(args.dst)

    if args.base:
        if dst.exists():
            print(f"[warn] {dst} 已存在，直接往里加（如需重来请先删掉）")
        else:
            print(f"[copy] {args.base} -> {dst}")
            shutil.copytree(args.base, dst)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (dst / sub).mkdir(parents=True, exist_ok=True)

    # data.yaml：改 path 和 names（保持类别顺序）
    (dst / "data.yaml").write_text(
        "# 由 add_labeled_frames.py 生成\n"
        f"path: {dst.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)),
        encoding="utf-8")

    files = [p for p in sorted(src_img.rglob("*")) if p.suffix.lower() in IMG_EXTS]
    random.seed(0)
    random.shuffle(files)
    n_val = int(len(files) * args.val_ratio)

    added = {"train": 0, "val": 0}
    empty = dropped = 0
    for i, p in enumerate(files):
        lab = src_lab / f"{p.stem}.txt"
        if not lab.exists():
            dropped += 1
            continue
        lines = [l for l in lab.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not lines:
            empty += 1
            continue
        if args.min_box_size > 0:
            keep = []
            for l in lines:
                q = l.split()
                if len(q) == 5 and float(q[3]) * float(q[4]) >= args.min_box_size:
                    keep.append(l)
            lines = keep
            if not lines:
                dropped += 1
                continue
        split = "val" if i < n_val else "train"
        name = f"{args.prefix}{p.stem}"
        shutil.copyfile(p, dst / "images" / split / f"{name}.jpg")
        (dst / "labels" / split / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")
        added[split] += 1

    n_tr = len(list((dst / "images" / "train").glob("*.jpg")))
    n_va = len(list((dst / "images" / "val").glob("*.jpg")))
    print(f"\n导入 {len(files)} 张 -> train {added['train']} / val {added['val']}"
          f"（空标签跳过 {empty}，缺标签/太小跳过 {dropped}）")
    print(f"数据集现在: train {n_tr} / val {n_va} -> {dst.resolve()}")
    print(f"类别: " + ", ".join(f"{i}={n}" for i, n in enumerate(names)))
    print(f"\n下一步: python train.py --data {dst}\\data.yaml --model runs\\ball_v4\\weights\\best.pt "
          f"--epochs 80 --lr0 0.001 --name ball_v5 --single-process")


if __name__ == "__main__":
    main()
