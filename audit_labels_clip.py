# -*- coding: utf-8 -*-
"""
用 CLIP 全量体检数据集里"篮球/排球"两类的标签对不对。

背景：伪标签流水线会把模型的类别错误固化进数据集（模型把排球认成篮球 → 标成篮球 →
重训后更倾向把排球叫篮球）。本工具用语义模型 CLIP 逐个框复核，找出错标。

用法:
  # 只体检，出报告 + 可疑样本拼图
  python audit_labels_clip.py --data dataset_ball_v10 --out audit_clip.jpg
  # 体检并直接改标签（只改 CLIP 非常确信、且两类都沾边的框）
  python audit_labels_clip.py --data dataset_ball_v10 --fix --backup
"""
import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")
import cv2
import numpy as np

from infer_onnx import CLASS_NAMES

LABELS = ["a basketball", "a volleyball", "an orange fruit", "a flower",
          "a logo or trademark", "a map", "a person's face", "a screenshot with text",
          "a plant or grass", "an abstract pattern", "a circle or dot", "other object"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_ball_v10")
    ap.add_argument("--out", default="audit_clip.jpg")
    ap.add_argument("--fix", action="store_true", help="把确信的错标改过来（篮球<->排球互换）")
    ap.add_argument("--backup", action="store_true", help="改之前备份到 <data>/labels_bak")
    ap.add_argument("--thr-conf", type=float, default=0.55, help="CLIP 判另一类的概率下限")
    ap.add_argument("--thr-margin", type=float, default=0.25, help="两类概率差的下限")
    ap.add_argument("--clip-root", default="weights/clip")
    args = ap.parse_args()

    import torch
    import clip
    from PIL import Image
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=dev, download_root=args.clip_root)
    with torch.no_grad():
        tf = model.encode_text(clip.tokenize(LABELS).to(dev))
        tf = tf / tf.norm(dim=-1, keepdim=True)

    root = Path(args.data)
    if args.fix and args.backup:
        bak = root / "labels_bak"
        if not bak.exists():
            shutil.copytree(root / "labels", bak)
            print(f"已备份标签 -> {bak}")

    rows = []
    for sp in ("train", "val"):
        for lab in sorted((root / "labels" / sp).glob("*.txt")):
            lines = [l for l in lab.read_text(encoding="utf-8").splitlines() if l.strip()]
            if not lines:
                continue
            imgp = root / "images" / sp / (lab.stem + ".jpg")
            img = cv2.imread(str(imgp))
            if img is None:
                continue
            h, w = img.shape[:2]
            for i, l in enumerate(lines):
                c, cx, cy, bw, bh = l.split()
                c = int(float(c))
                if c not in (0, 1):
                    continue
                cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
                x1, y1 = int(max(0, cx - bw / 2)), int(max(0, cy - bh / 2))
                x2, y2 = int(min(w, cx + bw / 2)), int(min(h, cy + bh / 2))
                pad = int(max(x2 - x1, y2 - y1) * 0.15)
                crop = img[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)]
                if crop.size == 0:
                    continue
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                with torch.no_grad():
                    f = model.encode_image(preprocess(Image.fromarray(rgb)).unsqueeze(0).to(dev))
                    f = f / f.norm(dim=-1, keepdim=True)
                    p = (100 * f @ tf.T).softmax(dim=-1)[0].cpu().numpy()
                rows.append({"file": lab.name, "split": sp, "idx": i, "label": c,
                             "pb": float(p[0]), "pv": float(p[1]),
                             "top": LABELS[int(p.argmax())], "box": (x1, y1, x2, y2),
                             "img": str(imgp)})

    print(f"体检了 {len(rows)} 个（篮球/排球）框")
    tot = Counter(r["label"] for r in rows)
    sus = []          # 可疑：CLIP 更倾向另一类
    for r in rows:
        other = r["pv"] if r["label"] == 0 else r["pb"]
        mine = r["pb"] if r["label"] == 0 else r["pv"]
        if other >= args.thr_conf and (other - mine) >= args.thr_margin:
            sus.append(r)
    print(f"  其中 篮球标签 {tot[0]} / 排球标签 {tot[1]}")
    print(f"  可疑（CLIP 更认为是另一类）: {len(sus)} 个 "
          f"= 占两类 {len(sus)/max(1,len(rows))*100:.1f}%")
    by = Counter((CLASS_NAMES[r['label']], CLASS_NAMES[0 if r['label']==1 else 1]) for r in sus)
    for (a, b), n in by.most_common():
        print(f"     标签是 {a}，CLIP 认为是 {b}: {n} 个")

    # 按可疑程度出图
    sus.sort(key=lambda r: -(max(r['pb'], r['pv']) - min(r['pb'], r['pv'])))
    tiles = []
    for r in sus[:24]:
        img = cv2.imread(r["img"])
        x1, y1, x2, y2 = r["box"]
        pad = int(max(x2 - x1, y2 - y1) * 0.3)
        c = img[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
        if c.size == 0:
            continue
        s = 170 / max(c.shape[:2])
        c = cv2.resize(c, (max(1, int(c.shape[1] * s)), max(1, int(c.shape[0] * s))))
        cv2.rectangle(c, (2, 2), (c.shape[1] - 3, c.shape[0] - 3), (0, 0, 255), 2)
        cv2.putText(c, f"标:{CLASS_NAMES[r['label']][:1]} CLIP:{r['top'][:12]}", (3, c.shape[0] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 255), 1)
        canvas = np.full((170, 170, 3), 25, np.uint8)
        canvas[:c.shape[0], :c.shape[1]] = c
        tiles.append(canvas)
    if tiles:
        while len(tiles) % 6:
            tiles.append(np.full((170, 170, 3), 25, np.uint8))
        cv2.imwrite(args.out, np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"  可疑样本拼图 -> {args.out}")

    if args.fix and sus:
        changed = 0
        per_file = {}
        for r in sus:
            per_file.setdefault((r["split"], r["file"]), []).append(r)
        for (sp, fname), rs in per_file.items():
            lab = root / "labels" / sp / fname
            lines = [l for l in lab.read_text(encoding="utf-8").splitlines() if l.strip()]
            for r in rs:
                c, cx, cy, bw, bh = lines[r["idx"]].split()
                newc = 1 if r["label"] == 0 else 0
                lines[r["idx"]] = f"{newc} {cx} {cy} {bw} {bh}"
                changed += 1
            lab.write_text("\n".join(lines), encoding="utf-8")
        print(f"\n[fix] 已改 {changed} 个框的类别（篮球<->排球互换）")
        (root / "audit_fixed.json").write_text(
            json.dumps([{k: v for k, v in r.items() if k != "img"} for r in sus],
                       ensure_ascii=False, indent=1), encoding="utf-8")
    elif sus:
        print("\n（没加 --fix，标签未改动）")


if __name__ == "__main__":
    main()
