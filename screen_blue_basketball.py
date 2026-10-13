# -*- coding: utf-8 -*-
"""针对性筛查：把"标成篮球、但主色是蓝/白"的框挑出来（排球是蓝黄白，篮球是橙或绿）。

CLIP 对模糊/远距离的球会没把握，漏过这类错标；颜色是个很硬的旁证。
用法: python screen_blue_basketball.py --data dataset_ball_v12 [--fix]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, ".")
import cv2
import numpy as np

from infer_onnx import CLASS_NAMES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_ball_v12")
    ap.add_argument("--out", default="blue_as_basketball.jpg")
    ap.add_argument("--fix", action="store_true", help="把确认的改成排球")
    ap.add_argument("--blue-frac", type=float, default=0.18,
                    help="框内蓝色像素占比超过它就算可疑")
    args = ap.parse_args()

    root = Path(args.data)
    sus = []
    for sp in ("train", "val"):
        for lab in sorted((root / "labels" / sp).glob("*.txt")):
            imgp = root / "images" / sp / (lab.stem + ".jpg")
            img = cv2.imread(str(imgp))
            if img is None:
                continue
            h, w = img.shape[:2]
            lines = [l for l in lab.read_text(encoding="utf-8").splitlines() if l.strip()]
            for i, l in enumerate(lines):
                c, cx, cy, bw, bh = l.split()
                if int(float(c)) != 0:                     # 只看标成篮球的
                    continue
                x1, y1 = int((float(cx) - float(bw) / 2) * w), int((float(cy) - float(bh) / 2) * h)
                x2, y2 = int((float(cx) + float(bw) / 2) * w), int((float(cy) + float(bh) / 2) * h)
                crop = img[max(0, y1):max(1, y2), max(0, x1):max(1, x2)]
                if crop.size < 100:
                    continue
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                hh, ss, vv = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
                m = (ss > 60) & (vv > 40)
                if m.sum() < 50:
                    continue
                blue = ((hh >= 95) & (hh <= 135) & m).sum() / m.sum()
                if blue >= args.blue_frac:
                    sus.append({"split": sp, "label": lab.name, "idx": i,
                                "file": imgp.name, "blue": blue,
                                "box": (x1, y1, x2, y2), "img": str(imgp)})

    print(f"可疑（标成篮球但主色偏蓝）: {len(sus)} 个 / 篮球框总数 "
          f"{sum(1 for sp in ('train','val') for t in (root/'labels'/sp).glob('*.txt') for l in t.read_text(encoding='utf-8').splitlines() if l.strip() and l.split()[0]=='0')}")
    tiles = []
    for s in sorted(sus, key=lambda x: -x["blue"])[:24]:
        img = cv2.imread(s["img"])
        x1, y1, x2, y2 = s["box"]
        pad = int(max(x2 - x1, y2 - y1) * 0.3)
        c = img[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
        if c.size == 0:
            continue
        sc = 170 / max(c.shape[:2])
        c = cv2.resize(c, (max(1, int(c.shape[1] * sc)), max(1, int(c.shape[0] * sc))))
        cv2.rectangle(c, (2, 2), (c.shape[1] - 3, c.shape[0] - 3), (0, 0, 255), 2)
        cv2.putText(c, f"蓝{s['blue']:.2f} {s['file'][:14]}", (3, c.shape[0] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 255), 1)
        canvas = np.full((170, 170, 3), 25, np.uint8)
        canvas[:c.shape[0], :c.shape[1]] = c
        tiles.append(canvas)
    if tiles:
        while len(tiles) % 6:
            tiles.append(np.full((170, 170, 3), 25, np.uint8))
        cv2.imwrite(args.out, np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        print("可疑样本 ->", args.out)

    if args.fix and sus:
        per = {}
        for s in sus:
            per.setdefault((s["split"], s["label"]), []).append(s["idx"])
        n = 0
        for (sp, fname), idxs in per.items():
            lab = root / "labels" / sp / fname
            lines = [l for l in lab.read_text(encoding="utf-8").splitlines() if l.strip()]
            for i in idxs:
                c, cx, cy, bw, bh = lines[i].split()
                lines[i] = f"1 {cx} {cy} {bw} {bh}"      # 篮球 -> 排球
                n += 1
            lab.write_text("\n".join(lines), encoding="utf-8")
        print(f"[fix] 已把 {n} 个框从篮球改为排球")
    elif sus:
        print("（未加 --fix，仅筛查）")


if __name__ == "__main__":
    main()
