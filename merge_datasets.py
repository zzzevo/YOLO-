#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
合并多个 YOLO 数据集（比如「网上搜的」+「自己实拍的」）成一个训练集。
用于做实验对比：纯网图 vs 网图+少量实拍。

用法:
  python merge_datasets.py --out dataset_mix --src dataset_web --src dataset_real

要求各来源的类别定义一致（类别数、顺序相同）；文件名会自动加来源前缀避免冲突。
验证集按 val-ratio 从合并后的全部样本里重新划分（保持与各来源原有划分无关的随机性）。
"""
import argparse
import random
import shutil
from pathlib import Path

import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_names(ds: Path):
    y = yaml.safe_load((ds / "data.yaml").read_text(encoding="utf-8"))
    names = y.get("names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda x: int(x))]
    return list(names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", action="append", required=True, help="可重复传多个来源数据集")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    args = ap.parse_args()

    out = Path(args.out)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    ref_names = read_names(Path(args.src[0]))
    print(f"[merge] 类别 = {ref_names}")
    for s in args.src[1:]:
        n = read_names(Path(s))
        if n != ref_names:
            raise SystemExit(f"类别不一致: {s} 是 {n}，但 {args.src[0]} 是 {ref_names}")

    # 收集所有 (image, label) 对，记录来源
    items = []
    for s in args.src:
        s = Path(s)
        tag = s.name
        for split in ("train", "val"):
            for img in sorted((s / "images" / split).glob("*")):
                if img.suffix.lower() not in IMG_EXTS:
                    continue
                lab = s / "labels" / split / (img.stem + ".txt")
                items.append((tag, img, lab if lab.exists() else None))
        print(f"  {tag}: 收集 {sum(1 for t,_,_ in items if t == tag)} 张")

    random.seed(0)
    random.shuffle(items)
    n_val = int(len(items) * args.val_ratio)

    for i, (tag, img, lab) in enumerate(items):
        split = "val" if i < n_val else "train"
        name = f"{tag}__{img.stem}"
        shutil.copyfile(img, out / "images" / split / f"{name}{img.suffix.lower()}")
        (out / "labels" / split / f"{name}.txt").write_text(
            lab.read_text(encoding="utf-8") if lab else "", encoding="utf-8")

    (out / "data.yaml").write_text(
        "# 由 merge_datasets.py 合并生成\n"
        f"path: {out.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(ref_names)),
        encoding="utf-8")

    n_tr = len(list((out / "images" / "train").glob("*")))
    n_va = len(list((out / "images" / "val").glob("*")))
    print(f"[done] 合并后 {len(items)} 张 (train {n_tr} / val {n_va}) -> {out.resolve()}")


if __name__ == "__main__":
    main()
