# -*- coding: utf-8 -*-
"""在爬到的网图上试几种"这真的是球吗"的判据，看哪种能把玫瑰/文字/水印挡掉。"""
import sys
from pathlib import Path

sys.path.insert(0, ".")
import cv2
import numpy as np

from infer_onnx import CLASS_NAMES, Detector, IMG_EXTS


def ball_metrics(img, box):
    """返回球的形状指标：extent(填充率) / circularity(圆度) / solidity(凸性) / aspect"""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(max(0, min(v, d))) for v, d in zip(box, (w, h, w, h))]
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None
    crop = img[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hh, ss, vv = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    valid = ss > 50
    if valid.sum() < 0.25 * valid.size:
        return None
    hist = np.bincount(hh[valid].ravel(), minlength=180)
    peak = int(hist.argmax())
    lo, hi = max(0, peak - 20), min(179, peak + 20)
    mask = cv2.inRange(hsv, (lo, 50, 20), (hi, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    a = cv2.contourArea(c)
    if a < 100:
        return None
    per = cv2.arcLength(c, True)
    x, y, bw, bh = cv2.boundingRect(c)
    hull = cv2.convexHull(c)
    ha = cv2.contourArea(hull)
    return {
        "extent": a / (bw * bh + 1e-6),
        "circ": 4 * np.pi * a / (per * per + 1e-6),
        "solid": a / (ha + 1e-6),
        "aspect": bw / (bh + 1e-6),
        "hue": peak,
    }


def main():
    det = Detector("runs/ball_v6/weights/best.onnx", 640, 0.40)
    files = [p for p in sorted(Path("raw_web_all").rglob("*")) if p.suffix.lower() in IMG_EXTS]
    rows = []
    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            continue
        for d in det(img)[0]:
            if d["cls"] not in (0, 1):
                continue
            m = ball_metrics(img, d["xyxy"])
            if m:
                rows.append((p.name, CLASS_NAMES[d["cls"]], d["conf"], m))
    print(f"候选检测: {len(rows)} 个（来自 {len(files)} 张图）")

    def test(name, fn):
        ok = [r for r in rows if fn(r)]
        print(f"  {name:38s} 通过 {len(ok):4d}")

    test("只按置信度>=0.5", lambda r: r[2] >= 0.5)
    test("置信度>=0.7", lambda r: r[2] >= 0.7)
    test("圆度>=0.65", lambda r: r[3]["circ"] >= 0.65)
    test("凸性>=0.90", lambda r: r[3]["solid"] >= 0.90)
    test("圆度>=0.65 + 凸性>=0.90", lambda r: r[3]["circ"] >= 0.65 and r[3]["solid"] >= 0.90)
    test("圆度>=0.60 + 凸性>=0.88 + 长宽0.7~1.4",
         lambda r: r[3]["circ"] >= 0.60 and r[3]["solid"] >= 0.88 and 0.7 <= r[3]["aspect"] <= 1.4)
    test("上面 + 置信度>=0.6",
         lambda r: r[3]["circ"] >= 0.60 and r[3]["solid"] >= 0.88
         and 0.7 <= r[3]["aspect"] <= 1.4 and r[2] >= 0.6)

    # 把"全条件"通过的打一张检查表
    ok = [r for r in rows if r[3]["circ"] >= 0.60 and r[3]["solid"] >= 0.88
          and 0.7 <= r[3]["aspect"] <= 1.4 and r[2] >= 0.6]
    tiles = []
    for name, cls, conf, m in ok[:12]:
        img = cv2.imread(str(Path("raw_web_all") / name))
        s = 300 / max(img.shape[:2])
        vis = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)))
        c = np.full((300, 300, 3), 25, np.uint8)
        c[(300 - vis.shape[0]) // 2:(300 - vis.shape[0]) // 2 + vis.shape[0],
          (300 - vis.shape[1]) // 2:(300 - vis.shape[1]) // 2 + vis.shape[1]] = vis
        cv2.putText(c, f"{cls} {conf:.2f} c{m['circ']:.2f} s{m['solid']:.2f}",
                    (3, 292), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)
        tiles.append(c)
    if tiles:
        while len(tiles) % 4:
            tiles.append(np.full((300, 300, 3), 25, np.uint8))
        cv2.imwrite("web_strict_check.jpg",
                    np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        print("已写出 web_strict_check.jpg")


if __name__ == "__main__":
    main()
