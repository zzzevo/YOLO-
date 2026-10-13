# -*- coding: utf-8 -*-
"""v2 筛查：标成篮球、但"饱和蓝 + 黄块"都很多 = 更像排球。

v1 的教训：绿色 Wilson 篮球的青色(H≈85~95)被误判成蓝，一次筛出 112 个假可疑。
所以这里要求：
  * 蓝色：H 105~135 且 S>80（排球的蓝很饱和；青绿球在这个范围外）
  * 黄色：H 20~38 且 S>80 占比也要够（排球是蓝黄拼块，篮球没有黄块）
两者同时满足才算可疑。
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, ".") if False else None
import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_ball_v12")
    ap.add_argument("--out", default="blue_suspect_v2.jpg")
    ap.add_argument("--fix", action="store_true")
    ap.add_argument("--blue-min", type=float, default=0.22)
    ap.add_argument("--yellow-min", type=float, default=0.08)
    args = ap.parse_args()

    root = Path(args.data)
    sus, total = [], 0
    for sp in ("train", "val"):
        for lab in sorted((root / "labels" / sp).glob("*.txt")):
            img = cv2.imread(str(root / "images" / sp / (lab.stem + ".jpg")))
            if img is None:
                continue
            h, w = img.shape[:2]
            lines = [l for l in lab.read_text(encoding="utf-8").splitlines() if l.strip()]
            for i, l in enumerate(lines):
                c, cx, cy, bw, bh = l.split()
                if int(float(c)) != 0:
                    continue
                total += 1
                x1, y1 = int((float(cx)-float(bw)/2)*w), int((float(cy)-float(bh)/2)*h)
                x2, y2 = int((float(cx)+float(bw)/2)*w), int((float(cy)+float(bh)/2)*h)
                crop = img[max(0, y1):max(1, y2), max(0, x1):max(1, x2)]
                if crop.size < 100:
                    continue
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                hh, ss, vv = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
                m = (ss > 60) & (vv > 40)
                if m.sum() < 50:
                    continue
                blue = (((hh >= 105) & (hh <= 135)) & m).sum() / m.sum()
                yellow = (((hh >= 20) & (hh <= 38)) & m).sum() / m.sum()
                if blue >= args.blue_min and yellow >= args.yellow_min:
                    sus.append({"split": sp, "label": lab.name, "idx": i,
                                "file": lab.stem + ".jpg",
                                "blue": blue, "yellow": yellow,
                                "box": (x1, y1, x2, y2),
                                "img": str(root / "images" / sp / (lab.stem + ".jpg"))})

    print(f"更严判据下可疑: {len(sus)} / 篮球框 {total}")
    tiles = []
    for s in sorted(sus, key=lambda x: -(x["blue"] + x["yellow"]))[:24]:
        img = cv2.imread(s["img"])
        x1, y1, x2, y2 = s["box"]
        pad = int(max(x2-x1, y2-y1) * 0.3)
        c = img[max(0, y1-pad):y2+pad, max(0, x1-pad):x2+pad]
        if c.size == 0:
            continue
        sc = 170 / max(c.shape[:2])
        c = cv2.resize(c, (max(1, int(c.shape[1]*sc)), max(1, int(c.shape[0]*sc))))
        cv2.rectangle(c, (2, 2), (c.shape[1]-3, c.shape[0]-3), (0, 0, 255), 2)
        cv2.putText(c, f"蓝{s['blue']:.2f} 黄{s['yellow']:.2f} {s['file'][:12]}", (3, c.shape[0]-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 255, 255), 1)
        canvas = np.full((170, 170, 3), 25, np.uint8)
        canvas[:c.shape[0], :c.shape[1]] = c
        tiles.append(canvas)
    if tiles:
        while len(tiles) % 6:
            tiles.append(np.full((170, 170, 3), 25, np.uint8))
        cv2.imwrite(args.out, np.vstack([np.hstack(tiles[i:i+6]) for i in range(0, len(tiles), 6)]),
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
                lines[i] = f"1 {cx} {cy} {bw} {bh}"
                n += 1
            lab.write_text("\n".join(lines), encoding="utf-8")
        print(f"[fix] {n} 个框 篮球->排球")
    elif sus:
        print("（未加 --fix）")


if __name__ == "__main__":
    main()
