#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
一个命令看懂模型效果：指标表 + 混淆矩阵读法 + 新旧模型对比图。

用法:
  python evaluate_model.py                                  # 评最新模型
  python evaluate_model.py --model runs/ball_v3/weights/best.pt
  python evaluate_model.py --model runs/ball_v3/weights/best.pt --compare runs/ball_v2/weights/best.pt
  python evaluate_model.py --model best.onnx               # 也能评原始 onnx

产出:
  屏幕: P/R/mAP50/mAP50-95（总体 + 分类别）+ 一句人话解读
  eval_report/评价报告.md      数字留档
  eval_report/预测检查.jpg     验证集上抽 12 张画框, 人眼确认
  eval_report/新旧对比.jpg     同一批帧上两个模型的并排对比（给了 --compare 才有）
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from train import install_single_process_pool

install_single_process_pool()

from ultralytics import YOLO  # noqa: E402

ROOT = Path(__file__).parent.resolve()


def error_breakdown(model_path, data, conf=0.25, iou_thr=0.5):
    """最直观的口径: 正确检出 / 漏检 / 误检 / 类别判错。
    mAP 变化很小时, 这几个数往往能看出真实差别（例如误检 5 -> 1）。"""
    from infer_onnx import Detector

    data = Path(data)
    val_img, val_lbl = data.parent / "images" / "val", data.parent / "labels" / "val"
    imgs = sorted(val_img.glob("*.jpg"))
    if not imgs:
        return None

    def iou(a, b):
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / ua if ua > 0 else 0

    mp = str(model_path)
    if mp.lower().endswith(".onnx"):
        det = Detector(mp, 640, conf)

        def predict(img, stem):
            return [(d["cls"], d["xyxy"]) for d in det(img)[0]]
    else:
        m = YOLO(mp)

        def predict(img, stem):
            r = m.predict(img, conf=conf, imgsz=640, verbose=False)[0]
            return [(int(b.cls[0]), [float(v) for v in b.xyxy[0]]) for b in r.boxes]

    tp = fp = fn = wrong = 0
    fp_examples = []
    for p in imgs:
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt = []
        lf = val_lbl / f"{p.stem}.txt"
        if lf.exists():
            for line in lf.read_text(encoding="utf-8").splitlines():
                q = line.split()
                if len(q) == 5:
                    c, cx, cy, bw, bh = int(q[0]), *map(float, q[1:])
                    gt.append((c, ((cx - bw / 2) * w, (cy - bh / 2) * h,
                                   (cx + bw / 2) * w, (cy + bh / 2) * h)))
        preds = predict(img, p.stem)
        used = set()
        for c, box in gt:
            best, bi = 0.0, -1
            for i, (_, pb) in enumerate(preds):
                if i in used:
                    continue
                v = iou(box, pb)
                if v > best:
                    best, bi = v, i
            if best >= iou_thr:
                used.add(bi)
                tp += int(preds[bi][0] == c)
                wrong += int(preds[bi][0] != c)
            else:
                fn += 1
        extra = [i for i in range(len(preds)) if i not in used]
        fp += len(extra)
        if extra:
            fp_examples.append(f"{p.name} 多出 {len(extra)} 个框")
    return {"正确检出": tp, "漏检": fn, "误检": fp, "类别判错": wrong, "_误检帧": fp_examples[:6]}


def tag_of(p):
    """用上级目录名区分模型（否则两个都叫 best.pt，报告里分不清）"""
    p = Path(p)
    return f"{p.parent.parent.name}/{p.name}" if p.parent.parent.name else p.name


def evaluate(model_path, data, tag):
    m = YOLO(model_path)
    r = m.val(data=str(data), imgsz=640, batch=16, device=0, workers=0, verbose=False, plots=False)
    b = r.box
    row = {
        "模型": tag,
        "P 精确率": round(float(b.mp), 4),
        "R 召回率": round(float(b.mr), 4),
        "mAP50": round(float(b.map50), 4),
        "mAP50-95": round(float(b.map), 4),
    }
    per = {}
    try:
        for n, i in enumerate(b.ap_class_index):
            per[r.names[int(i)]] = round(float(b.all_ap[n][0]), 4)
    except Exception:
        pass
    return row, per, r


def verdict(row):
    p, rr, m50 = row["P 精确率"], row["R 召回率"], row["mAP50"]
    s = []
    s.append(f"mAP50 = {m50:.3f}：" + ("很好" if m50 > 0.9 else "可用" if m50 > 0.7 else "还不行"))
    s.append(f"P = {p:.3f}：" + ("误检很少" if p > 0.9 else "有误检，可以调高置信度阈值" if p > 0.7 else "误检偏多"))
    s.append(f"R = {rr:.3f}：" + ("基本不漏" if rr > 0.9 else "有漏检，可以调低置信度阈值" if rr > 0.7 else "漏检偏多，需要更多数据"))
    return s


def sheet(files, models, tags, out, conf=0.25, tile=300, cols=4):
    """每张图一行: 左=模型A, 右=模型B"""
    rows_img = []
    for f in files:
        row = []
        for mp, tg in zip(models, tags):
            m = YOLO(mp)
            r = m.predict(str(f), conf=conf, imgsz=640, verbose=False)[0]
            vis = r.plot(line_width=2)
            h, w = vis.shape[:2]
            s = tile / max(h, w)
            vis = cv2.resize(vis, (int(w * s), int(h * s)))
            canvas = np.full((tile, tile, 3), 25, np.uint8)
            y = (tile - vis.shape[0]) // 2
            x = (tile - vis.shape[1]) // 2
            canvas[y:y + vis.shape[0], x:x + vis.shape[1]] = vis
            cv2.putText(canvas, tg, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"{len(r.boxes)} det", (4, tile - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0) if len(r.boxes) else (0, 0, 255), 1, cv2.LINE_AA)
            row.append(canvas)
        rows_img.append(np.hstack(row))
    sheet_img = np.vstack(rows_img)
    cv2.imwrite(str(out), sheet_img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return sheet_img.shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "runs" / "ball_v3" / "weights" / "best.pt"))
    ap.add_argument("--compare", default=str(ROOT / "runs" / "ball_v2" / "weights" / "best.pt"),
                    help="对比模型（留空则不对比）")
    ap.add_argument("--data", default=str(ROOT / "dataset_ball_ms" / "data.yaml"))
    ap.add_argument("--n", type=int, default=12, help="对比图抽多少帧")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    out = ROOT / "eval_report"
    out.mkdir(exist_ok=True)
    data = Path(args.data)
    if not data.exists():
        raise SystemExit(f"找不到 {data}")

    rows = []
    print("=" * 72)
    print(f"验证集: {data}  (模型没见过这些帧)")
    print("=" * 72)
    row, per, _ = evaluate(args.model, data, tag_of(args.model))
    rows.append(row)
    print(f"\n【{row['模型']}】")
    for k, v in row.items():
        if k != "模型":
            print(f"   {k:12s} {v}")
    if per:
        print("   分类别 mAP50:", "  ".join(f"{k}={v}" for k, v in per.items()))
    for s in verdict(row):
        print("   ·", s)

    if args.compare and Path(args.compare).exists():
        row2, per2, _ = evaluate(args.compare, data, tag_of(args.compare))
        rows.append(row2)
        print(f"\n【{row2['模型']}】（对比）")
        for k, v in row2.items():
            if k != "模型":
                print(f"   {k:12s} {v}")
        if per2:
            print("   分类别 mAP50:", "  ".join(f"{k}={v}" for k, v in per2.items()))

    # 预测检查图 + 新旧对比图
    files = sorted((data.parent / "images" / "val").glob("*.jpg"))
    if files:
        idx = np.linspace(0, len(files) - 1, min(args.n, len(files))).astype(int)
        sel = [files[i] for i in idx]
        hw = sheet(sel, [args.model], ["预测"], out / "预测检查.jpg", args.conf)
        print(f"\n已写出 {out / '预测检查.jpg'} {hw}")
        models, tags = [args.compare, args.model], ["旧", "新"]
        if args.compare and Path(args.compare).exists():
            hw = sheet(sel, models, tags, out / "新旧对比.jpg", args.conf)
            print(f"已写出 {out / '新旧对比.jpg'} {hw}")

    # ---- 最直观的口径: 正确 / 漏检 / 误检 / 类别判错 ----
    print("\n" + "=" * 72)
    print("误检漏检明细（比 mAP 更直观，验证集逐帧对比标注）")
    print("=" * 72)
    errs = {}
    for mp in [args.model] + ([args.compare] if args.compare and Path(args.compare).exists() else []):
        e = error_breakdown(mp, data, args.conf)
        if e:
            errs[tag_of(mp)] = e
            print(f"  {tag_of(mp):24s} 正确 {e['正确检出']:3d} | 漏检 {e['漏检']:3d} | "
                  f"误检 {e['误检']:3d} | 类别判错 {e['类别判错']}")
            if e["_误检帧"]:
                print(f"      误检所在帧: {', '.join(e['_误检帧'])}")

    md = ["# 模型评价报告", "", f"- 验证集: `{data}`", f"- 抽取帧数: {args.n}", "",
          "## 一、指标（越高越好）", "",
          "| 模型 | P 精确率 | R 召回率 | mAP50 | mAP50-95 |", "|---|---|---|---|---|"]
    for r_ in rows:
        md.append(f"| {r_['模型']} | {r_['P 精确率']} | {r_['R 召回率']} | {r_['mAP50']} | {r_['mAP50-95']} |")
    if errs:
        md += ["", "## 二、误检漏检明细（最直观）", "",
               "| 模型 | 正确检出 | 漏检 | 误检 | 类别判错 |", "|---|---|---|---|---|"]
        for k, e in errs.items():
            md.append(f"| {k} | {e['正确检出']} | {e['漏检']} | {e['误检']} | {e['类别判错']} |")
    md += ["", "## 三、怎么读", "",
           "| 指标 | 含义 | 低了说明什么 | 怎么办 |", "|---|---|---|---|",
           "| P 精确率 | 报出来的框里有多少是对的 | 误检多（满屏瞎框） | 调高置信度阈值；补负样本 |",
           "| R 召回率 | 真目标里找到多少 | 漏检多 | 调低置信度阈值；补这类场景的数据 |",
           "| mAP50 | IoU>0.5 就算对的综合分 | 看进步主要看这个 | 加数据 / 多训练 |",
           "| mAP50-95 | 更严格，衡量框贴不贴 | 框画得不够准 | 通常和数据标注精度有关 |",
           "",
           "## 四、图", "",
           "- `预测检查.jpg`：当前模型在验证集上的预测（人眼确认最靠谱）",
           "- `新旧对比.jpg`：左旧右新，同一批帧并排",
           "",
           "## 五、训练曲线在哪看", "",
           "双击 `start_workbench.bat`，然后开 <http://127.0.0.1:6006/>；",
           "或直接看 `runs/<本次训练名>/results.png`、`BoxPR_curve.png`、`confusion_matrix.png`。"]
    (out / "评价报告.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n已写出 {out / '评价报告.md'}")
    print("下一步看曲线: 双击 start_workbench.bat, 然后开 http://127.0.0.1:6006/")


if __name__ == "__main__":
    main()
