# -*- coding: utf-8 -*-
"""
专查"排球被认成篮球"这类**类间混淆**：自己算"真值类别 -> 预测类别"的对应关系，
并把判错的帧挑出来出图（绿框=真值，红框=预测）。

用法:
  python confusion_check.py --model runs/ball_v10/weights/best.pt --data data_eval_ball11.yaml
  python confusion_check.py --model runs/ball_v11/weights/best.pt --compare runs/ball_v10/weights/best.pt \
      --data data_eval_volley11.yaml --out confusion_wrong.jpg
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")
import cv2
import numpy as np
import yaml

from infer_onnx import CLASS_NAMES


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0


def load_val(data):
    d = yaml.safe_load(Path(data).read_text(encoding="utf-8"))
    root = Path(d["path"])
    names = [str(d["names"][i]) for i in sorted(d["names"])]
    val = d["val"]
    if str(val).endswith(".txt"):
        files = [Path(l.strip()) for l in (root / val).read_text(encoding="utf-8").splitlines() if l.strip()]
    else:
        files = sorted((root / str(val)).glob("*.jpg"))
    return root, names, files


def classify_pairs(model, data, names, imgsz=640, conf=0.25, iou_thr=0.45):
    """返回 {(真值类, 预测类): 计数} 与判错明细列表"""
    root, names, files = load_val(data)
    pairs = Counter()
    wrong = []
    for imgp in files:
        lab = root / "labels" / imgp.parent.name / (imgp.stem + ".txt")
        if not lab.exists():
            continue
        gts = []
        for l in lab.read_text(encoding="utf-8").splitlines():
            if l.strip():
                c, cx, cy, bw, bh = l.split()
                gts.append((int(float(c)),
                            (float(cx) - float(bw) / 2, float(cy) - float(bh) / 2,
                             float(cx) + float(bw) / 2, float(cy) + float(bh) / 2)))
        if not gts:
            continue
        res = model.predict(str(imgp), conf=conf, imgsz=imgsz, verbose=False)[0]
        preds = [(int(b.cls), tuple(float(v) for v in b.xyxyn[0]), float(b.conf)) for b in res.boxes]
        for gc, gbox in gts:
            if not preds:
                pairs[(names[gc], "漏检")] += 1
                continue
            best, bi = 0.0, None
            for i, (pc, pbox, pconf) in enumerate(preds):
                v = iou(gbox, pbox)
                if v > best:
                    best, bi = v, i
            if bi is None or best < iou_thr:
                pairs[(names[gc], "漏检")] += 1
                continue
            pc = preds[bi][0]
            pairs[(names[gc], names[pc])] += 1
            if pc != gc:
                wrong.append({"img": str(imgp), "gt": names[gc], "pred": names[pc],
                              "conf": preds[bi][2], "gt_box": gbox, "pred_box": preds[bi][1]})
    return pairs, wrong


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--compare", help="对比模型")
    ap.add_argument("--data", default="data_eval_ball11.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default="confusion_wrong.jpg")
    args = ap.parse_args()

    from train import install_single_process_pool
    install_single_process_pool()
    from ultralytics import YOLO

    pairs_models = [("A", args.model)] + ([("B", args.compare)] if args.compare else [])
    for tag, mp in pairs_models:
        if not mp:
            continue
        print(f"\n===== {tag}: {mp} =====")
        model = YOLO(mp)
        r = model.val(data=args.data, imgsz=args.imgsz, batch=8, device=0, workers=0,
                      verbose=False, plots=False, conf=args.conf)
        print(f"  总体: P {r.box.mp:.4f}  R {r.box.mr:.4f}  mAP50 {r.box.map50:.4f}  mAP50-95 {r.box.map:.4f}")

        _, names, _ = load_val(args.data)
        pairs, wrong = classify_pairs(model, args.data, names, args.imgsz, args.conf)
        print("  真值 -> 预测（每个真值框的归属）:")
        for (g, p), n in sorted(pairs.items(), key=lambda x: -x[1]):
            mark = "   <-- 类间判错" if (g != p and p != "漏检") else ""
            print(f"     {g:>4s} -> {p:<5s}: {n}{mark}")
        conf_n = sum(n for (g, p), n in pairs.items() if g != p and p != "漏检")
        miss_n = sum(n for (g, p), n in pairs.items() if p == "漏检")
        print(f"  类间判错 {conf_n} / 漏检 {miss_n} / 合计 {sum(pairs.values())}")

        if wrong:
            print("  判错明细（前 8）:")
            for x in wrong[:8]:
                print(f"     {Path(x['img']).name}: 真值 {x['gt']} -> 预测 {x['pred']} ({x['conf']:.2f})")
            tiles = []
            for x in wrong[:12]:
                img = cv2.imread(x["img"])
                h, w = img.shape[:2]
                gx1, gy1, gx2, gy2 = [int(v * d) for v, d in zip(x["gt_box"], (w, h, w, h))]
                px1, py1, px2, py2 = [int(v * d) for v, d in zip(x["pred_box"], (w, h, w, h))]
                cv2.rectangle(img, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
                cv2.rectangle(img, (px1, py1), (px2, py2), (0, 0, 255), 2)
                pad = int(max(gx2 - gx1, gy2 - gy1) * 1.2)
                cx, cy = (gx1 + gx2) // 2, (gy1 + gy2) // 2
                c = img[max(0, cy - pad):cy + pad, max(0, cx - pad):cx + pad]
                if c.size == 0:
                    continue
                s = 200 / max(c.shape[:2])
                c = cv2.resize(c, (int(c.shape[1] * s), int(c.shape[0] * s)))
                canvas = np.full((220, 200, 3), 25, np.uint8)
                canvas[:c.shape[0], :c.shape[1]] = c
                cv2.putText(canvas, f"真{x['gt'][:1]} 预{x['pred'][:1]} {x['conf']:.2f}", (3, 214),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                tiles.append(canvas)
            if tiles:
                while len(tiles) % 4:
                    tiles.append(np.full((220, 200, 3), 25, np.uint8))
                out = args.out if len(pairs_models) == 1 else args.out.replace(".jpg", f"_{tag}.jpg")
                cv2.imwrite(out, np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]),
                            [cv2.IMWRITE_JPEG_QUALITY, 90])
                print(f"  判错样本 -> {out}")
        else:
            print("  ✓ 没有类间判错")


if __name__ == "__main__":
    main()
