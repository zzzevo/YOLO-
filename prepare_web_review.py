# -*- coding: utf-8 -*-
"""
为"网图"生成 X-AnyLabeling 复核工程：预填模型框，人只需删掉错的/补上漏的。

为什么需要人：实测三轮，网图的**自动标注不可靠**——模型会把橙子、玫瑰、彩虹、
地图、人脸、商标都当成球，而形状/颜色启发式无法区分"橙子"和"橙色篮球"。
所以网图只能走"人工快速过一遍"的路子。

用法: python prepare_web_review.py --src raw_web_all --dst review_web --conf 0.5
"""
import argparse
import shutil
from pathlib import Path

import cv2

from infer_onnx import CLASS_NAMES, Detector, IMG_EXTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="raw_web_all")
    ap.add_argument("--dst", default="review_web")
    ap.add_argument("--model", default="runs/ball_v6/weights/best.onnx")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--cls-ids", default="0,1")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    keep = {int(x) for x in args.cls_ids.split(",")}
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "images").mkdir(parents=True)
    (dst / "labels").mkdir(parents=True)

    det = Detector(args.model, 640, 0.25)
    files = [p for p in sorted(src.rglob("*")) if p.suffix.lower() in IMG_EXTS]
    n = 0
    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        dets = [d for d in det(img)[0] if d["cls"] in keep and d["conf"] >= args.conf][:4]
        if not dets:                       # 没检出的图不进复核（大概率真没球，标它没意义）
            continue
        shutil.copyfile(p, dst / "images" / p.name)
        lines = []
        for d in dets:
            x1, y1, x2, y2 = d["xyxy"]
            lines.append(f"{d['cls']} {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} "
                         f"{(x2-x1)/w:.6f} {(y2-y1)/h:.6f}")
        (dst / "labels" / (p.stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")
        n += 1

    (dst / "classes.txt").write_text("\n".join(CLASS_NAMES), encoding="utf-8")
    (dst / "说明.md").write_text(
        "# 网图复核工程\n\n"
        f"共 {n} 张（只放了模型检出了东西的图）。\n\n"
        "**目的**：这些是爬来的网图，模型的框**有约三成是错的**（实测：橙子、玫瑰、彩虹、\n"
        "地图、人脸、商标都会被当成球）。请你快速过一遍：\n\n"
        "1. 框对的 → 不用动\n"
        "2. 框错的 → 删掉框（X-AnyLabeling 里选中框按 Delete）\n"
        "3. 图里还有球但没框 → 补一个框（篮球=0 排球=1 定位柱=2）\n\n"
        "启动：双击 `start_labeling.bat`（或 conda activate labeling && xanylabeling）\n"
        "然后把本目录的 images 拖进去，标注保存在 labels/ 下。\n\n"
        "标完告诉我，我把这批图并进数据集重训（v8）。\n", encoding="utf-8")
    print(f"[done] 复核工程 -> {dst.resolve()}  共 {n} 张（源自 {len(files)} 张爬图）")


if __name__ == "__main__":
    main()
