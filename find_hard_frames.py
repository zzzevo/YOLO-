#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
找出「模型自己没把握」的帧 —— 这些帧人工标一遍，收益最大。

为什么需要它：自动标注得到的标签本质是"旧模型的答案"。用这些标签训练，
新模型最好也只能追平旧模型；想真正变强，必须在**旧模型不确定的地方**补人工标注。

判定标准（任一命中即入选）：
  * 置信度低于 --conf 的框
  * 整帧一个框都没有（漏检嫌疑）
  * 同一帧里同时出现两类（易混淆）

用法:
  python find_hard_frames.py --src raw_ball --out to_review --conf 0.6
输出:
  to_review/*.jpg         需要人工核对的帧
  to_review/需人工核对.txt  按严重程度排序的清单
"""
import argparse
import shutil
from pathlib import Path

import cv2

from infer_onnx import CLASS_NAMES, Detector, IMG_EXTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="抽帧后的图片目录")
    ap.add_argument("--out", default="to_review", help="输出目录")
    ap.add_argument("--model", default="best.onnx")
    ap.add_argument("--conf", type=float, default=0.60, help="低于该置信度的框算'没把握'")
    ap.add_argument("--label-conf", type=float, default=0.25, help="与自动标注一致的阈值")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    imgs = [p for p in sorted(src.rglob("*")) if p.suffix.lower() in IMG_EXTS]
    if args.limit:
        imgs = imgs[: args.limit]
    if not imgs:
        raise SystemExit(f"{src} 里没有图片")

    out.mkdir(parents=True, exist_ok=True)
    det = Detector(args.model, 640, args.label_conf)

    low, empty, mixed = [], [], []
    for p in imgs:
        img = cv2.imread(str(p))
        if img is None:
            continue
        dets, _ = det(img)
        if not dets:
            empty.append((p, 0.0, "整帧无检测"))
            continue
        confs = [d["conf"] for d in dets]
        cls = {d["cls"] for d in dets}
        worst = min(confs)
        if len(cls) > 1:
            mixed.append((p, worst, f"同帧两类: {[CLASS_NAMES[c] for c in cls]}"))
        elif worst < args.conf:
            low.append((p, worst, f"低置信度 {worst:.2f}"))

    rows = sorted(empty + low + mixed, key=lambda x: x[1])
    for p, _, _ in rows:
        shutil.copyfile(p, out / p.name)

    lines = [f"共 {len(imgs)} 帧, 需人工核对 {len(rows)} 帧 "
             f"(整帧无检测 {len(empty)} / 低置信 {len(low)} / 同帧两类 {len(mixed)})", ""]
    for p, c, why in rows:
        lines.append(f"{p.name}\t{c:.3f}\t{why}")
    (out / "需人工核对.txt").write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines[:6]))
    print(f"\n[done] {len(rows)} 帧 -> {out.resolve()}")
    print("  下一步: 用 X-AnyLabeling 打开这些图补标/改标, 再重训 —— 这才会真正超过现在的模型")


if __name__ == "__main__":
    main()
