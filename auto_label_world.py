#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
开放词汇自动标注：用 YOLO-World + 文字提示词给「模型没见过的类别」自动打标。

适合：你的 best.onnx 只认识 2 类，现在想加新类（比如 class0 那种），又不想一上来就手标几百张。

用法:
  python auto_label_world.py --src raw_web --dst dataset_web --prompt "robot, ball" --conf 0.15

输出与 build_dataset.py 一致（images/{train,val} + labels/{train,val} + data.yaml），
可直接拿去 train.py 训练。

依赖: ultralytics 的 CLIP 文本编码器
    pip install openai-clip            # 若失败且能访问 GitHub: pip install git+https://github.com/ultralytics/CLIP.git
首次运行会自动下载 yolov8s-worldv2.pt (~25MB) 和 CLIP 权重。

⚠️ 实测提醒:
  YOLO-World 在"通用物体"上很灵，但对特定比赛道具/特定视角的召回率有限，
  自动标注结果**必须人工过一遍**；用它做 bootstrap，再训练自己的模型迭代，才是正确用法。
"""
import argparse
import random
import shutil
from pathlib import Path

import cv2

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
WORLD_MODEL = "yolov8s-worldv2.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--prompt", required=True, help='类别提示词，逗号分隔，如 "robot, ball"')
    ap.add_argument("--conf", type=float, default=0.15, help="宁低勿高，漏检比误检更难补")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--neg-ratio", type=float, default=0.1)
    ap.add_argument("--model", default=WORLD_MODEL)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("需要训练环境: conda activate yolo26")

    names = [s.strip() for s in args.prompt.replace("，", ",").split(",") if s.strip()]
    if not names:
        raise SystemExit("--prompt 不能为空")

    model = YOLO(args.model)
    try:
        model.set_classes(names)
    except ModuleNotFoundError as e:
        raise SystemExit(
            f"缺少 CLIP 文本编码器（{e}）。请先安装:\n"
            f"    pip install openai-clip\n"
            f"  若网络能访问 GitHub 也可: pip install git+https://github.com/ultralytics/CLIP.git")
    print(f"[world] 类别 = {names}")

    src, dst = Path(args.src), Path(args.dst)
    imgs = [p for p in sorted(src.rglob("*")) if p.suffix.lower() in IMG_EXTS]
    if args.limit:
        imgs = imgs[: args.limit]
    if not imgs:
        raise SystemExit(f"{src} 下没有图片")

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (dst / sub).mkdir(parents=True, exist_ok=True)
    for old in ("images", "labels"):
        pass  # 目录已就绪

    random.seed(0)
    random.shuffle(imgs)
    n_val = int(len(imgs) * args.val_ratio)
    stats = {"pos": 0, "neg": 0, "box": 0}
    per_cls = [0] * len(names)

    for i, p in enumerate(imgs):
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        res = model.predict(str(p), conf=args.conf, verbose=False)[0]
        lines = []
        for b in res.boxes:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            cls = int(b.cls[0])
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            if bw <= 0 or bh <= 0:
                continue
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            per_cls[cls] += 1

        if not lines and random.random() > args.neg_ratio:
            continue
        split = "val" if i < n_val else "train"
        name = p.stem
        k = 1
        while (dst / "images" / split / f"{name}.jpg").exists():
            name = f"{p.stem}_{k}"; k += 1
        shutil.copyfile(p, dst / "images" / split / f"{name}.jpg")
        (dst / "labels" / split / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")
        if lines:
            stats["pos"] += 1
            stats["box"] += len(lines)
        else:
            stats["neg"] += 1
        if (i + 1) % 25 == 0:
            print(f"  ... {i + 1}/{len(imgs)} 有目标 {stats['pos']}")

    yml = (
        "# 自动生成 by auto_label_world.py —— 类别顺序 = --prompt 里的顺序\n"
        f"path: {dst.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names))
    )
    (dst / "data.yaml").write_text(yml, encoding="utf-8")

    print("\n[done]")
    print(f"  有目标 {stats['pos']} 张 / 空标注 {stats['neg']} 张 / 共 {stats['box']} 个框")
    print(f"  各类: {dict(zip(names, per_cls))}")
    print(f"  data.yaml -> {dst / 'data.yaml'}")
    print("  下一步: 人工校对 labels/，然后 python train.py --data ... --name yolo26n_v1")


if __name__ == "__main__":
    main()
