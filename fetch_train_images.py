#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
「爬能直接用来训练的图」——边下边验，不合格的直接不落盘。

为什么之前的网图那么多跟球无关：
  1) Bing 的普通图片搜索会把**海报、logo、文字图、地图、卡通**都算相关（"篮球"这个词
     出现在图片说明里就算命中）；
  2) 旧流程是"先全下下来，事后再筛"，所以你看到 raw_web_* 目录里全是垃圾，留用率只有 ~16%。

本脚本的改进（都是实测有效的）：
  * `qft=+filterui:photo-photo`   —— 只要**照片**，把 clipart/logo/插画/地图挡在搜索层
  * 尺寸过滤 imagesize-large/medium
  * **下载后立刻验证**：YOLO 检出球 → 裁框给 **CLIP 语义复核** → 两个都过才保存（图片+标签）
  * 跨关键词**内容级去重**（平均哈希，按图像内容而不是 URL）
  * 每个关键词的**产出率**都打印，方便你删掉没用的关键词

用法:
  python fetch_train_images.py --dst raw_good \
      --queries "篮球 商品图" "volleyball product photo" "basketball isolated" \
      --per-query 40 --clip-thr 0.9 --conf 0.5
产出: raw_good/images/*.jpg + raw_good/labels/*.txt + raw_good/manifest.csv（可直接进数据集）
"""
import argparse
import csv
import hashlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import cv2
import numpy as np

from fetch_images_bing import UA, ahash, decode, http_get
from infer_onnx import CLASS_NAMES, Detector

LABELS = ["a basketball", "a volleyball", "an orange fruit", "a flower",
          "a logo or trademark", "a map", "a person's face", "a screenshot with text",
          "a plant or grass", "an abstract pattern", "a circle or dot", "other object"]


def search_bing(query, want=35, photo_only=True, size="large"):
    import re
    import urllib.parse
    qft = []
    if photo_only:
        qft.append("+filterui:photo-photo")
    if size:
        qft.append(f"+filterui:imagesize-{size}")
    url = ("https://www.bing.com/images/search?q=" + urllib.parse.quote(query)
           + f"&count={want}&first=1&qft=" + "".join(qft))
    html = http_get(url, timeout=20).decode("utf-8", "replace")
    html = html.replace("&quot;", '"').replace("&amp;", "&")
    murls = re.findall(r'"murl":"(.*?)"', html)
    turls = re.findall(r'"turl":"(.*?)"', html)
    out = []
    for i, m in enumerate(murls):
        t = turls[i] if i < len(turls) else None
        out.append((m.replace("\\/", "/"), t.replace("\\/", "/") if t else None))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", required=True)
    ap.add_argument("--queries", nargs="+", required=True)
    ap.add_argument("--per-query", type=int, default=35)
    ap.add_argument("--model", default="runs/ball_v11/weights/best.onnx")
    ap.add_argument("--conf", type=float, default=0.5, help="YOLO 检出球的置信度门槛")
    ap.add_argument("--clip-thr", type=float, default=0.9, help="CLIP 判为篮球/排球的概率门槛")
    ap.add_argument("--clip-root", default="weights/clip")
    ap.add_argument("--min-side", type=int, default=200)
    ap.add_argument("--max-side", type=int, default=2000)
    ap.add_argument("--photo-only", action="store_true", default=True)
    ap.add_argument("--no-photo-only", dest="photo_only", action="store_false")
    ap.add_argument("--cls-ids", default="0,1")
    ap.add_argument("--sleep", type=float, default=0.05)
    args = ap.parse_args()

    import torch
    import clip
    from PIL import Image
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cmodel, cprep = clip.load("ViT-B/32", device=dev, download_root=args.clip_root)
    with torch.no_grad():
        tf = cmodel.encode_text(clip.tokenize(LABELS).to(dev))
        tf = tf / tf.norm(dim=-1, keepdim=True)

    def clip_ball_prob(crop):
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            f = cmodel.encode_image(cprep(Image.fromarray(rgb)).unsqueeze(0).to(dev))
            f = f / f.norm(dim=-1, keepdim=True)
            p = (100 * f @ tf.T).softmax(dim=-1)[0].cpu().numpy()
        return float(p[0] + p[1]), LABELS[int(p.argmax())]

    det = Detector(args.model, 640, 0.15)
    keep_ids = {int(x) for x in args.cls_ids.split(",")}
    dst = Path(args.dst)
    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "labels").mkdir(parents=True, exist_ok=True)

    hashes, rows, seen = [], [], set()
    grand = {"cand": 0, "down": 0, "noball": 0, "lowconf": 0, "clipno": 0, "dup": 0, "kept": 0}
    per_query = []

    for q in args.queries:
        print(f"\n[搜索] {q}")
        try:
            cand = search_bing(q, args.per_query, args.photo_only)
        except Exception as e:
            print(f"  搜索失败: {type(e).__name__}: {str(e)[:60]}")
            continue
        st = {"cand": len(cand), "down": 0, "kept": 0, "clipno": 0, "noball": 0, "dup": 0}
        for murl, turl in cand:
            if murl in seen:
                continue
            seen.add(murl)
            img = None
            for url in (murl, turl):
                if not url:
                    continue
                try:
                    img = decode(http_get(url, timeout=10, referer="https://www.bing.com/"))
                    if img is not None:
                        break
                except Exception:
                    continue
            if img is None:
                continue
            st["down"] += 1
            h, w = img.shape[:2]
            if min(h, w) < args.min_side:
                continue
            if max(h, w) > args.max_side:
                s = args.max_side / max(h, w)
                img = cv2.resize(img, (int(w * s), int(h * s)))
                h, w = img.shape[:2]
            hv = ahash(img)
            if any(int(np.count_nonzero(hv != x)) < 5 for x in hashes):
                st["dup"] += 1
                continue
            # 边下边验：YOLO 检出 → CLIP 复核
            dets = [d for d in det(img)[0] if d["cls"] in keep_ids and d["conf"] >= args.conf]
            if not dets:
                st["noball"] += 1
                continue
            good = []
            for d in dets[:2]:
                x1, y1, x2, y2 = [int(v) for v in d["xyxy"]]
                pad = int(max(x2 - x1, y2 - y1) * 0.08)
                crop = img[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)]
                if crop.size == 0:
                    continue
                bp, top = clip_ball_prob(crop)
                if bp >= args.clip_thr:
                    good.append(d)
                else:
                    st["clipno"] += 1
            if not good:
                continue
            hashes.append(hv)
            name = hashlib.md5(murl.encode()).hexdigest()[:12]
            cv2.imwrite(str(dst / "images" / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            lines = []
            for d in good:
                x1, y1, x2, y2 = d["xyxy"]
                lines.append(f"{d['cls']} {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} "
                             f"{(x2-x1)/w:.6f} {(y2-y1)/h:.6f}")
            (dst / "labels" / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")
            rows.append({"file": f"{name}.jpg", "query": q, "url": murl,
                         "classes": ",".join(CLASS_NAMES[d["cls"]] for d in good)})
            st["kept"] += 1
            time.sleep(args.sleep)
        rate = st["kept"] / max(1, st["cand"]) * 100
        print(f"  候选 {st['cand']} -> 下载 {st['down']} -> 留用 {st['kept']}"
              f"（无球 {st['noball']} / CLIP否 {st['clipno']} / 重复 {st['dup']}）留用率 {rate:.0f}%")
        per_query.append((q, st))
        for k in st:
            if k in grand:
                grand[k] += st[k]

    with open(dst / "manifest.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=["file", "query", "url", "classes"])
        wr.writeheader()
        wr.writerows(rows)

    print(f"\n{'='*60}")
    print(f"总候选 {grand['cand']} -> 下载 {grand['down']} -> **留用 {grand['kept']}**"
          f"（留用率 {grand['kept']/max(1,grand['cand'])*100:.0f}%，旧流程只有 ~16%）")
    print(f"丢弃原因: 无球 {grand['noball']} / CLIP 否 {grand['clipno']} / 内容重复 {grand['dup']}")
    print("\n各关键词产出率（低于 20% 的建议删掉）:")
    for q, st in sorted(per_query, key=lambda x: -x[1]["kept"]):
        print(f"  {st['kept']:3d}/{st['cand']:3d} = {st['kept']/max(1,st['cand'])*100:3.0f}%   {q}")
    print(f"\n[done] 图片+标签已就绪 -> {dst.resolve()}")
    print(f"  下一步并入数据集: python add_labeled_frames.py --images {dst}/images "
          f"--labels {dst}/labels --base dataset_ball_v11 --dst dataset_ball_v12 "
          f"--val-ratio 0.15 --names \"篮球,排球,定位柱\" --prefix web_")


if __name__ == "__main__":
    main()
