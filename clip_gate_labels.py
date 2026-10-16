# -*- coding: utf-8 -*-
"""
给已有标注再加一道 **CLIP "这是不是球"** 检查，剔除像"红色线缆盘"这种骗过形状判据的框。

（教训：形状+颜色判据会把圆形红线圈当成排球 —— 必须再过一次语义检查）
用法: python clip_gate_labels.py --src raw_v21 --labels v21_labels/labels [--thr 0.9]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, ".")
import cv2
import numpy as np

LABELS = ["a basketball", "a volleyball", "an orange fruit", "a flower",
          "a logo or trademark", "a map", "a person's face", "a screenshot with text",
          "a plant or grass", "an abstract pattern", "a circle or dot", "other object",
          "a coil of cable or rope", "a shoe", "a plastic bottle"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--thr", type=float, default=0.90, help="P(是球) 门槛")
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

    def prob(crop):
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            f = cmodel.encode_image(cprep(Image.fromarray(rgb)).unsqueeze(0).to(dev))
            f = f / f.norm(dim=-1, keepdim=True)
            p = (100 * f @ tf.T).softmax(dim=-1)[0].cpu().numpy()
        return float(p[0] + p[1]), LABELS[int(p.argmax())]

    src, lab_dir = Path(args.src), Path(args.labels)
    kept = dropped = 0
    reasons = {}
    for lab in sorted(lab_dir.glob("*.txt")):
        img = cv2.imread(str(src / (lab.stem + ".jpg")))
        if img is None:
            continue
        h, w = img.shape[:2]
        out = []
        for l in lab.read_text(encoding="utf-8").splitlines():
            if not l.strip():
                continue
            c, cx, cy, bw, bh = l.split()
            x1, y1 = int((float(cx) - float(bw) / 2) * w), int((float(cy) - float(bh) / 2) * h)
            x2, y2 = int((float(cx) + float(bw) / 2) * w), int((float(cy) + float(bh) / 2) * h)
            crop = img[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            if crop.size < 100:
                dropped += 1
                continue
            p_ball, top = prob(crop)
            if p_ball < args.thr:
                dropped += 1
                reasons[top] = reasons.get(top, 0) + 1
                continue
            out.append(l)
            kept += 1
        lab.write_text("\n".join(out), encoding="utf-8")
    print(f"{args.labels}: 保留 {kept} / CLIP 判不是球剔除 {dropped}")
    if reasons:
        print("  被剔除的框 CLIP 认为是什么:", dict(sorted(reasons.items(), key=lambda x: -x[1])))


if __name__ == "__main__":
    main()
