#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
把「模型没把握的帧」整理成 X-AnyLabeling 能直接打开的复核工程。

产出结构（X-AnyLabeling 打开 images 目录即可，labels 是它的同级目录，会被自动识别）:

    review/
    ├─ images/            待复核的帧（从 raw_ball 复制）
    ├─ labels/            与之一一对应的 YOLO txt（把模型现有预测先填进去，你只需要改错的）
    ├─ classes.txt        类别名，导入 YOLO 标注时选它
    └─ 复核说明.txt        本批次的清单

为什么要先填好预测：模型大部分框是对的，白纸上重画 46 张很痛苦；
填好后你只需要「补漏检、删误检、改类别」，快 5~10 倍。

用法:
  python prepare_review.py --src raw_ball --dst dataset_ball --review review --conf 0.60
  # 只挑指定的帧（比如清单里的）: 加 --list to_review/需人工核对.txt
"""
import argparse
import re
import shutil
from pathlib import Path

import cv2

from infer_onnx import CLASS_NAMES, Detector, IMG_EXTS, to_yolo_lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="原始抽帧目录（raw_ball）")
    ap.add_argument("--dst", help="现有数据集目录（dataset_ball）—— 已有的好标签会直接沿用")
    ap.add_argument("--review", default="review", help="输出复核工程目录")
    ap.add_argument("--model", default="best.onnx")
    ap.add_argument("--conf", type=float, default=0.60, help="低于该置信度的帧入选")
    ap.add_argument("--label-conf", type=float, default=0.25, help="生成预测框用的阈值")
    ap.add_argument("--list", help="只处理这个清单文件里提到的帧名")
    ap.add_argument("--max", type=int, default=0, help="最多多少张，0=全部")
    args = ap.parse_args()

    src = Path(args.src)
    rev = Path(args.review)
    (rev / "images").mkdir(parents=True, exist_ok=True)
    (rev / "labels").mkdir(parents=True, exist_ok=True)
    (rev / "classes.txt").write_text("\n".join(CLASS_NAMES), encoding="utf-8")

    # 1) 选帧：优先用清单，否则按"模型没把握"自动筛
    if args.list:
        names = []
        for line in Path(args.list).read_text(encoding="utf-8").splitlines():
            m = re.match(r"^(\S+\.jpg)", line.strip())
            if m:
                names.append(m.group(1))
        files = [src / n for n in names if (src / n).exists()]
        print(f"[list] 清单里 {len(names)} 帧, 实际找到 {len(files)} 帧")
    else:
        det = Detector(args.model, 640, args.label_conf)
        files = []
        for p in sorted(src.rglob("*")):
            if p.suffix.lower() not in IMG_EXTS:
                continue
            img = cv2.imread(str(p))
            if img is None:
                continue
            dets, _ = det(img)
            if not dets or min(d["conf"] for d in dets) < args.conf or len({d["cls"] for d in dets}) > 1:
                files.append(p)
        print(f"[auto] 按 conf<{args.conf} / 无检测 / 同帧两类 筛出 {len(files)} 帧")

    if args.max:
        files = files[: args.max]
    if not files:
        raise SystemExit("没有需要复核的帧")

    # 2) 复制图片 + 生成初始标签
    det = Detector(args.model, 640, args.label_conf)
    reused = fresh = 0
    rows = []
    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        shutil.copyfile(p, rev / "images" / p.name)

        # 数据集里若已有该帧的标签，直接沿用（通常比现场预测更"干净"）
        lab_src = None
        if args.dst:
            for split in ("train", "val"):
                cand = Path(args.dst) / "labels" / split / (p.stem + ".txt")
                if cand.exists():
                    lab_src = cand
                    break
        if lab_src:
            text = lab_src.read_text(encoding="utf-8")
            reused += 1
            why = "沿用数据集标签"
        else:
            dets, _ = det(img)
            text = "\n".join(to_yolo_lines(dets, w, h))
            fresh += 1
            why = f"模型现测 {len(dets)} 框"
        (rev / "labels" / (p.stem + ".txt")).write_text(text, encoding="utf-8")
        rows.append(f"{p.name}\t{len([l for l in text.splitlines() if l.strip()])} 框\t{why}")

    (rev / "复核说明.txt").write_text(
        f"共 {len(rows)} 帧（沿用已有标签 {reused} / 用模型现测 {fresh}）\n"
        f"类别: " + ", ".join(f"{i}={n}" for i, n in enumerate(CLASS_NAMES)) + "\n\n"
        "用 X-AnyLabeling 打开 images 目录，导入 YOLO 标注（选 labels 与 classes.txt），逐个改：\n"
        "  · 漏检 -> 补一个新框\n  · 误检 -> 选中按 Delete\n  · 类别错 -> 在右侧标签面板改成正确类别\n"
        "改完执行:  python apply_review.py --review review --dst dataset_ball\n\n"
        "文件名\t初始框数\t来源\n" + "\n".join(rows), encoding="utf-8")

    print(f"\n[done] 复核工程: {rev.resolve()}")
    print(f"  images/: {len(list((rev / 'images').glob('*.jpg')))} 张")
    print(f"  labels/: {len(list((rev / 'labels').glob('*.txt')))} 个（沿用 {reused} / 现测 {fresh}）")
    print(f"  classes.txt: {CLASS_NAMES}")
    print("  X-AnyLabeling 里打开 review\\images 目录即可")


if __name__ == "__main__":
    main()
