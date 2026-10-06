#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
用 best.onnx 给「未标注的一堆图片」自动打标，直接生成 YOLO 训练集目录骨架。

流程: 拍图 -> 本脚本自动标 -> 人工用 labelImg/X-AnyLabeling 修正 -> train_yolo26n.ps1 训练

用法:
  python build_dataset.py --src D:\\raw_images --dst D:\\datasets\\robocup_balls
  可选: --conf 0.3 --val-ratio 0.2 --neg-ratio 0.1 --limit 0

注意:
  1) 自动标注只是"半成品"，必须逐张人工校对，尤其是漏检和误检；
  2) 负样本(空 txt)也要留一部分，否则模型会满屏误检；
  3) 类别顺序固定为 CLASS_NAMES，见 infer_onnx.py。
"""
import argparse
import random
import shutil
from pathlib import Path

import cv2

from infer_onnx import CLASS_NAMES, Detector, detect_multiscale, to_yolo_lines, IMG_EXTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="原始图片目录（可递归）")
    ap.add_argument("--dst", required=True, help="输出数据集目录")
    ap.add_argument("--model", default=str(Path(__file__).with_name("best.onnx")))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.30, help="自动标注用的置信度阈值，宁低勿高")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--neg-ratio", type=float, default=0.10, help="空标注负样本保留比例")
    ap.add_argument("--limit", type=int, default=0, help="最多处理多少张，0=全部")
    ap.add_argument("--keep", default="", help="只保留这些类别 id（逗号分隔，按给定顺序重映射为 0..n-1）；"
                                              "例: --keep 1 只做单类数据集")
    ap.add_argument("--val-prefix", default="", help="文件名以这些前缀开头的划入验证集（逗号分隔）。"
                                                     "按视频/场次划分验证集比随机划分更诚实，例: --val-prefix v2_")
    ap.add_argument("--scales", default="1.0",
                    help="尺度空间自动标注：逗号分隔的缩放倍率，如 1.0,0.6,0.4。"
                         "模型对'太大'的目标会失明，缩小画面再检测可救回（实测 0.006 -> 0.92）")
    args = ap.parse_args()

    scales = tuple(float(x) for x in args.scales.replace("，", ",").split(",") if x.strip())

    val_prefixes = tuple(x.strip() for x in args.val_prefix.replace("，", ",").split(",") if x.strip())

    keep = [int(x) for x in args.keep.replace("，", ",").split(",") if x.strip() != ""]
    keep_names = [CLASS_NAMES[k] if k < len(CLASS_NAMES) else f"class{k}" for k in keep] or list(CLASS_NAMES)

    src, dst = Path(args.src), Path(args.dst)
    imgs = [p for p in sorted(src.rglob("*")) if p.suffix.lower() in IMG_EXTS]
    if args.limit:
        imgs = imgs[: args.limit]
    if not imgs:
        raise SystemExit(f"{src} 下没有图片")

    # 防止重复运行导致图片被 _1 后缀复制一遍，先清掉旧的 images/ 与 labels/
    for old in ("images", "labels"):
        if (dst / old).exists():
            shutil.rmtree(dst / old)
            print(f"[clean] 已清空旧的 {dst / old}")

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (dst / sub).mkdir(parents=True, exist_ok=True)

    det = Detector(args.model, args.imgsz, args.conf)
    random.seed(0)
    random.shuffle(imgs)

    n_val = int(len(imgs) * args.val_ratio)
    stats = {"pos": 0, "neg_kept": 0, "neg_dropped": 0, "boxes": 0}
    per_cls = [0] * len(keep_names)

    for i, p in enumerate(imgs):
        img = cv2.imread(str(p))
        if img is None:
            print(f"[skip] 读不到 {p}")
            continue
        dets, _ = det(img) if len(scales) <= 1 else (detect_multiscale(det, img, scales), 0)
        if keep:                                   # 只留指定类别，并重映射 id
            dets = [dict(d, cls=keep.index(d["cls"])) for d in dets if d["cls"] in keep]
        split = "val" if i < n_val else "train"
        if val_prefixes and p.name.startswith(val_prefixes):   # 按前缀强制划入验证集
            split = "val"
        h, w = img.shape[:2]
        lines = to_yolo_lines(dets, w, h)

        if not lines and random.random() > args.neg_ratio:
            stats["neg_dropped"] += 1
            continue
        if not lines:
            stats["neg_kept"] += 1
        else:
            stats["pos"] += 1
            stats["boxes"] += len(lines)
            for d in dets:
                per_cls[d["cls"]] += 1

        name = f"{p.stem}"
        k = 1
        while (dst / "images" / split / f"{name}.jpg").exists():
            name = f"{p.stem}_{k}"; k += 1
        shutil.copyfile(p, dst / "images" / split / f"{name}.jpg")
        (dst / "labels" / split / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")

        if (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{len(imgs)}  正样本 {stats['pos']}")

    yaml_txt = (
        f"# 自动生成 by build_dataset.py —— 类别顺序必须与训练时一致，改这里也要改 infer_onnx.py\n"
        # 必须写绝对路径: ultralytics 会把 data.yaml 里的相对 path 解析到全局 datasets_dir, 而不是 yaml 所在目录
        f"path: {Path(args.dst).resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(keep_names))
    )
    (dst / "data.yaml").write_text(yaml_txt, encoding="utf-8")

    print("\n[done]")
    print(f"  图片总数 {len(imgs)}  有目标 {stats['pos']}  空标注保留 {stats['neg_kept']}  丢弃 {stats['neg_dropped']}")
    print(f"  框总数 {stats['boxes']}  各类: {dict(zip(keep_names, per_cls))}")
    print(f"  data.yaml -> {dst / 'data.yaml'}")
    print("  下一步: 人工校对 labels/ 下的 txt，然后跑 train.py --data <data.yaml> --name yolo26n_v1")


if __name__ == "__main__":
    main()
