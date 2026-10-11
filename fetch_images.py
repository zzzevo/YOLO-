#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
联网采集图片：按关键词搜索图片 → 下载 → 去重/去糊/尺寸过滤 → 得到可标注的原始图库。

用法:
  python fetch_images.py --dst D:\\raw_web --queries "robocup junior robot" "智能小车 赛道"
  python fetch_images.py --dst D:\\raw_web --queries "足球机器人" --per-query 80 --min-side 320

输出:
  <dst>/*.jpg            图片
  <dst>/manifest.csv     每张图的 来源URL/关键词/尺寸，便于溯源与合规审查

⚠️ 合规与效果提醒（务必读）:
  1) 网上图片有版权，仅用于自己实验/学习；商用或公开数据集请用有授权的来源。
  2) **域差异是致命的**：网上找的图和 MaixCam 实拍在分辨率、光照、反光、背景上完全不同。
     用纯网图训练，模型在你自己的赛道上大概率不work。正确用法是：
       网图 → 扩类/预训练 → **再用自己摄像头实拍微调与验证**（实拍必须占主导）。
  3) 搜索结果质量参差，采完必须人工快速过一遍，把明显不相关的删掉。
"""
import argparse
import csv
import hashlib
import time
from pathlib import Path

import cv2
import numpy as np

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


def ahash(img, size=8):
    g = cv2.cvtColor(cv2.resize(img, (size, size)), cv2.COLOR_BGR2GRAY)
    return (g > g.mean()).flatten()


def hamming(a, b):
    return int(np.count_nonzero(a != b))


def download(url, timeout=15):
    """用 urllib 直连下载（不用 requests，少一个依赖）"""
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Referer": url})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        if r.status != 200:
            return None
        data = r.read()
    if len(data) < 4096:
        return None
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", required=True)
    ap.add_argument("--queries", nargs="+", required=True)
    ap.add_argument("--per-query", type=int, default=60, help="每个关键词最多取多少张")
    ap.add_argument("--min-side", type=int, default=320, help="短边小于该值的丢弃")
    ap.add_argument("--max-side", type=int, default=4000, help="过大的先缩到该尺寸")
    ap.add_argument("--blur-thresh", type=float, default=40.0, help="糊图阈值，0=不过滤")
    ap.add_argument("--dedup-dist", type=int, default=4, help="平均哈希汉明距离小于该值视为重复")
    ap.add_argument("--backend", default="auto",
                    help="搜索引擎后端: auto / duckduckgo / brave / mojeek / yandex。"
                         "某个被墙或限流时换一个（实测 duckduckgo 超时, brave 可用）")
    args = ap.parse_args()

    from ddgs import DDGS

    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    rows, hashes = [], []
    seen_urls = set()
    total_kept = 0

    for q in args.queries:
        print(f"\n[搜索] {q}")
        try:
            if args.backend and args.backend != "auto":
                results = DDGS().images(q, max_results=args.per_query, backend=args.backend)
            else:
                results = DDGS().images(q, max_results=args.per_query)
        except Exception as e:
            print(f"  搜索失败: {type(e).__name__}: {e}")
            continue
        kept = skipped = 0
        for r in results:
            url = r.get("image") or ""
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            try:
                img = download(url)
            except Exception:
                skipped += 1
                continue
            if img is None:
                skipped += 1
                continue
            h, w = img.shape[:2]
            if min(h, w) < args.min_side:
                skipped += 1
                continue
            if max(h, w) > args.max_side:
                s = args.max_side / max(h, w)
                img = cv2.resize(img, (int(w * s), int(h * s)))
            if args.blur_thresh > 0:
                if cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() < args.blur_thresh:
                    skipped += 1
                    continue
            hv = ahash(img)
            if any(hamming(hv, x) < args.dedup_dist for x in hashes):
                skipped += 1
                continue
            hashes.append(hv)
            name = hashlib.md5(url.encode()).hexdigest()[:12] + ".jpg"
            cv2.imwrite(str(dst / name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            rows.append({"file": name, "query": q, "url": url, "w": img.shape[1], "h": img.shape[0]})
            kept += 1
            total_kept += 1
            if kept % 20 == 0:
                print(f"  已下载 {kept} ...")
            time.sleep(0.05)
        print(f"  保留 {kept}，跳过 {skipped}")

    with open(dst / "manifest.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["file", "query", "url", "w", "h"])
        w.writeheader()
        w.writerows(rows)

    print(f"\n[done] 共保留 {total_kept} 张 -> {dst.resolve()}")
    print(f"  清单: {dst / 'manifest.csv'}（含来源 URL，便于合规审查）")
    print(f"  下一步: python auto_label_world.py --src {dst} --prompt \"robot, ball\" --dst {dst.parent / 'labeled'}")
    print("  ⚠️ 网图必须人工筛一遍，且最终要用自己摄像头实拍微调")


if __name__ == "__main__":
    main()
