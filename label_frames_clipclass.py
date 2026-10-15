#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
给一批帧自动标注，**类别由 CLIP 判定**（而不是模型判定），并把两者不一致的挑出来供人工核对。

为什么：模型会把"没见过的排球"叫成篮球（比如红白黄配色的排球）。CLIP 是语义模型，
按"看起来像篮球还是排球"判断，更稳。两者不一致的样本会单独出图，人工只需看这些。

用法:
  python label_frames_clipclass.py --src raw_v17 --out v17_labels --model runs/ball_v16/weights/best.onnx
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from infer_onnx import CLASS_NAMES, Detector, IMG_EXTS

LABELS = ["a basketball", "a volleyball", "an orange fruit", "a flower",
          "a logo or trademark", "a map", "a person's face", "a screenshot with text",
          "a plant or grass", "an abstract pattern", "a circle or dot", "other object"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="runs/ball_v16/weights/best.onnx")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--cls-ids", default="0,1", help="只标球（默认不含定位柱）")
    ap.add_argument("--clip-thr", type=float, default=0.90, help="CLIP 判'是球'的门槛")
    ap.add_argument("--cls-thr", type=float, default=0.55, help="CLIP 判'哪一类'的门槛，低于它就用模型类别并标记待核")
    ap.add_argument("--max-boxes", type=int, default=4)
    ap.add_argument("--clip-root", default="weights/clip")
    args = ap.parse_args()

    import torch
    import clip
    from PIL import Image
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cmodel, cprep = clip.load("ViT-B/32", device=dev, download_root=args.clip_root)
    with torch.no_grad():
        tf = cmodel.encode_text(clip.tokenize(LABELS).to(dev))
        tf = tf / tf.norm(dim=-1, keepdim=True)

    def clip_eval(crop):
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            f = cmodel.encode_image(cprep(Image.fromarray(rgb)).unsqueeze(0).to(dev))
            f = f / f.norm(dim=-1, keepdim=True)
            p = (100 * f @ tf.T).softmax(dim=-1)[0].cpu().numpy()
        return float(p[0]), float(p[1]), LABELS[int(p.argmax())]

    det = Detector(args.model, 640, 0.15)
    keep = {int(x) for x in args.cls_ids.split(",")}
    src, out = Path(args.src), Path(args.out)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    rows, stat, flag = [], {"总框": 0, "CLIP定类": 0, "用模型类": 0, "不一致": 0, "丢弃非球": 0}, []
    for p in sorted(src.rglob("*")):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        dets = [d for d in det(img)[0] if d["cls"] in keep and d["conf"] >= args.conf][: args.max_boxes]
        lines = []
        for d in dets:
            stat["总框"] += 1
            x1, y1, x2, y2 = [int(v) for v in d["xyxy"]]
            pad = int(max(x2 - x1, y2 - y1) * 0.08)
            crop = img[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)]
            if crop.size == 0:
                continue
            pb, pv, top = clip_eval(crop)
            if pb + pv < args.clip_thr:                       # CLIP 认为不是球 -> 丢
                stat["丢弃非球"] += 1
                continue
            clip_cls = 0 if pb >= pv else 1
            clip_p = max(pb, pv)
            model_cls = int(d["cls"])
            if clip_p >= args.cls_thr:
                final = clip_cls
                stat["CLIP定类"] += 1
            else:
                final = model_cls
                stat["用模型类"] += 1
                flag.append((p.name, model_cls, clip_cls, clip_p, d["conf"], (x1, y1, x2, y2)))
            if model_cls != clip_cls:
                stat["不一致"] += 1
                flag.append((p.name, model_cls, clip_cls, clip_p, d["conf"], (x1, y1, x2, y2)))
            lines.append((final, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h))
            rows.append({"file": p.name, "final": CLASS_NAMES[final], "model": CLASS_NAMES[model_cls],
                         "clip": CLASS_NAMES[clip_cls], "clip_p": round(clip_p, 3),
                         "yolo_conf": round(float(d["conf"]), 3)})
        (out / "labels" / (p.stem + ".txt")).write_text(
            "\n".join(f"{c} {a:.6f} {b:.6f} {cw:.6f} {ch:.6f}" for c, a, b, cw, ch in lines),
            encoding="utf-8")

    with open(out / "labels_report.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=["file", "final", "model", "clip", "clip_p", "yolo_conf"])
        wr.writeheader()
        wr.writerows(rows)

    print(f"处理 {len(list((out / 'labels').glob('*.txt')))} 帧")
    print(f"  总框 {stat['总框']} | CLIP 定类 {stat['CLIP定类']} | 用模型类 {stat['用模型类']} | "
          f"模型与 CLIP 不一致 {stat['不一致']} | CLIP 判不是球丢弃 {stat['丢弃非球']}")
    from collections import Counter
    print("  最终类别分布:", dict(Counter(r["final"] for r in rows)))
    if flag:
        print(f"\n⚠️ 需要人工核对的框 {len(flag)} 个（前 15）:")
        for f in flag[:15]:
            print(f"   {f[0]}: 模型={CLASS_NAMES[f[1]]} CLIP={CLASS_NAMES[f[2]]}({f[3]:.2f}) "
                  f"yolo_conf={f[4]:.2f} box={f[5]}")
    print(f"\n标签写到 {out/'labels'}")


if __name__ == "__main__":
    main()
