#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
安全地把「网图 / 额外素材」追加进数据集：只用模型有把握的帧，宁缺毋滥。

为什么需要过滤：网图上模型没检测到，有两种可能——(a) 图里真没球（可以当负样本），
(b) 图里有球但模型瞎了（**当负样本就会教模型更瞎**）。两者肉眼分不清，所以本工具
默认只保留"模型明确检出"的图，其余丢掉，并打印丢弃统计供你抽查。

用法:
  # 1) 先复制一份基础数据集（不动原来的）
  python add_web_frames.py --src raw_web_bb --base dataset_ball_ms --dst dataset_ball_v4

  # 2) 只在一个已有数据集上追加（不复制）
  python add_web_frames.py --src raw_web_bb --dst dataset_ball_v4 --no-copy

参数:
  --scales   尺度空间检测（网图里球常常很大，必须开）
  --conf     保留门槛：框的置信度下限
  --class-ids 只接受这些类别（默认 0=篮球）
  --max-boxes 一张图最多几个框（多于这个数说明模型在图里乱检，丢掉更安全）
  --min-side 图片短边下限（太小的图没训练价值）
"""
import argparse
import shutil
from pathlib import Path

import cv2

from infer_onnx import CLASS_NAMES, Detector, IMG_EXTS, detect_multiscale, to_yolo_lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="要追加的图片目录（如 raw_web_bb）")
    ap.add_argument("--dst", required=True, help="目标数据集（追加到它的 images/train）")
    ap.add_argument("--base", help="基础数据集；给了就先整体复制成 --dst")
    ap.add_argument("--model", default="best.onnx")
    ap.add_argument("--scales", default="1.0,0.6,0.4")
    ap.add_argument("--conf", type=float, default=0.50, help="保留门槛（越高越保守）")
    ap.add_argument("--class-ids", default="0", help="接受的类别 id，逗号分隔")
    ap.add_argument("--force-class", type=int, default=None,
                    help="把保留的框的类别强制改成这个 id（用于模型'定位对了但类别判错'的场景）")
    ap.add_argument("--max-boxes", type=int, default=3)
    ap.add_argument("--min-side", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--prefix", default="web_", help="追加帧的文件名前缀")
    ap.add_argument("--val-ratio", type=float, default=0.0,
                    help="从追加的帧里划多少比例进验证集（0=全进 train）")
    ap.add_argument("--names", default="", help="重写 data.yaml 的类别名（逗号分隔），空=不动")
    ap.add_argument("--strict-ball", action="store_true",
                    help="再加一道'像不像球'检查（单一主色 + 圆形填充率 + 颜色均匀度）。"
                         "网图污染严重时必开：模型会把地图标记、人脸、彩虹也认成球")
    ap.add_argument("--min-box-frac", type=float, default=0.0,
                    help="框最小面积占整图比例（配合 --strict-ball 用，如 0.01）")
    ap.add_argument("--clip-filter", action="store_true",
                    help="用 CLIP 对检测框做二次语义判别（**网图强烈建议开**）。"
                         "模型会把橙子/玫瑰/彩虹/地图/人脸当成球，而形状颜色规则分不清"
                         "'橙子'和'橙色篮球'；CLIP 是语义模型，能分清")
    ap.add_argument("--clip-thr", type=float, default=0.9,
                    help="CLIP 判为'篮球/排球'的概率门槛（0.9 实测能干净切开，见 clip_filter.py）")
    ap.add_argument("--clip-root", default="weights/clip")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if args.base and not dst.exists():
        print(f"[copy] {args.base} -> {dst}")
        shutil.copytree(args.base, dst)
        (dst / "data.yaml").write_text(
            (dst / "data.yaml").read_text(encoding="utf-8").replace(
                str(Path(args.base).resolve().as_posix()), str(dst.resolve().as_posix())),
            encoding="utf-8")
    (dst / "images" / "train").mkdir(parents=True, exist_ok=True)
    (dst / "labels" / "train").mkdir(parents=True, exist_ok=True)
    if args.val_ratio > 0:
        (dst / "images" / "val").mkdir(parents=True, exist_ok=True)
        (dst / "labels" / "val").mkdir(parents=True, exist_ok=True)
    if args.names:
        names = [s.strip() for s in args.names.replace("，", ",").split(",") if s.strip()]
        (dst / "data.yaml").write_text(
            f"# 由 add_web_frames.py 重写\npath: {dst.resolve().as_posix()}\n"
            "train: images/train\nval: images/val\nnames:\n"
            + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)), encoding="utf-8")

    keep_ids = {int(x) for x in args.class_ids.replace("，", ",").split(",") if x.strip()}
    scales = tuple(float(x) for x in args.scales.split(",") if x.strip())
    imgs = [p for p in sorted(src.rglob("*")) if p.suffix.lower() in IMG_EXTS]
    if args.limit:
        imgs = imgs[: args.limit]
    if not imgs:
        raise SystemExit(f"{src} 里没有图片")

    det = Detector(args.model, 640, 0.10)     # 低阈值取原始输出，再用 --conf 过滤
    strict = None
    if args.strict_ball:
        from auto_label_hueshift import looks_like_ball
        strict = looks_like_ball
    clip_fn = None
    if args.clip_filter:
        import torch
        import clip as clipmod
        from PIL import Image
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        cmodel, cprep = clipmod.load("ViT-B/32", device=dev, download_root=args.clip_root)
        labels = ["a basketball", "a volleyball", "an orange fruit", "a flower",
                  "a logo or trademark", "a map", "a person's face", "a screenshot with text",
                  "a plant or grass", "an abstract pattern", "a circle or dot", "other object"]
        with torch.no_grad():
            tf = cmodel.encode_text(clipmod.tokenize(labels).to(dev))
            tf = tf / tf.norm(dim=-1, keepdim=True)

        def clip_fn(img, box):                 # noqa: F811
            h, w = img.shape[:2]
            x1, y1, x2, y2 = [int(max(0, min(v, d))) for v, d in zip(box, (w, h, w, h))]
            mw, mh = int((x2 - x1) * 0.08), int((y2 - y1) * 0.08)
            crop = img[max(0, y1 - mh):min(h, y2 + mh), max(0, x1 - mw):min(w, x2 + mw)]
            if crop.size == 0:
                return 0.0, "空"
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            with torch.no_grad():
                f = cmodel.encode_image(cprep(Image.fromarray(rgb)).unsqueeze(0).to(dev))
                f = f / f.norm(dim=-1, keepdim=True)
                p = (100 * f @ tf.T).softmax(dim=-1)[0].cpu().numpy()
            return float(p[0] + p[1]), labels[int(p.argmax())]

    kept = 0
    drop = {"无检测": 0, "置信度低": 0, "框太多": 0, "尺寸太小": 0, "读不出": 0, "类别不符": 0,
            "不像球": 0, "框太小": 0, "CLIP 否定": 0}
    rows = []
    for p in imgs:
        img = cv2.imread(str(p))
        if img is None:
            drop["读不出"] += 1
            continue
        h, w = img.shape[:2]
        if min(h, w) < args.min_side:
            drop["尺寸太小"] += 1
            continue
        dets = detect_multiscale(det, img, scales) if len(scales) > 1 else det(img)[0]
        dets = [d for d in dets if d["cls"] in keep_ids]
        if not dets:
            drop["无检测"] += 1
            continue
        good = [d for d in dets if d["conf"] >= args.conf]
        if not good:
            drop["置信度低"] += 1
            continue
        if len(good) > args.max_boxes:
            drop["框太多"] += 1
            continue
        if args.force_class is not None:
            good = [dict(d, cls=args.force_class) for d in good]
        if strict is not None:                      # 网图污染严重：再加一道"像不像球"
            passed = []
            for d in good:
                x1, y1, x2, y2 = d["xyxy"]
                if args.min_box_frac > 0 and (x2 - x1) * (y2 - y1) < args.min_box_frac * w * h:
                    drop["框太小"] += 1
                    continue
                ok, _why = strict(img, (x1, y1, x2, y2))
                if ok:
                    passed.append(d)
                else:
                    drop["不像球"] += 1
            good = passed
            if not good:
                continue
        if clip_fn is not None:                     # CLIP 语义复核（网图的最后一道闸）
            passed = []
            for d in good:
                bp, top = clip_fn(img, d["xyxy"])
                if bp >= args.clip_thr:
                    passed.append(d)
                else:
                    drop["CLIP 否定"] += 1
                    if len(rows) < 8:
                        rows.append(f"   [CLIP 否] {p.name}: 模型 {CLASS_NAMES[d['cls']]} "
                                    f"{d['conf']:.2f} 但 CLIP 认为: {top} (球概率 {bp:.2f})")
            good = passed
            if not good:
                continue
        name = f"{args.prefix}{p.stem}"
        split = "val" if (args.val_ratio > 0 and kept < int(len(imgs) * args.val_ratio)) else "train"
        shutil.copyfile(p, dst / "images" / split / f"{name}.jpg")
        (dst / "labels" / split / f"{name}.txt").write_text(
            "\n".join(to_yolo_lines(good, w, h)), encoding="utf-8")
        kept += 1
        if len(rows) < 8:
            desc = ", ".join(f"{CLASS_NAMES[d['cls']]} {d['conf']:.2f}" for d in good)
            rows.append(f"{p.name}: {len(good)} 框 [{desc}]")

    print(f"\n共 {len(imgs)} 张 -> 保留 {kept} 张")
    for k, v in drop.items():
        if v:
            print(f"   丢弃 {k}: {v}")
    for r in rows:
        print("   ", r)
    print(f"\n[done] 已追加到 {dst}/images/train  (现在共 "
          f"{len(list((dst / 'images' / 'train').glob('*.jpg')))} 张训练图)")
    print(f"  下一步: python train.py --data {dst}\\data.yaml --model runs\\ball_v3\\weights\\best.pt "
          f"--epochs 80 --lr0 0.001 --name ball_v4 --single-process")


if __name__ == "__main__":
    main()
