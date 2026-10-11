#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 Bing 图片搜索抓图（不依赖第三方搜图库；原图下载失败时自动退回 Bing 缩略图）。

为什么需要它：ddgs 的后端时好时坏（duckduckgo 超时、bing async 报 TLS 错），
icrawler 能搜到但原图 CDN 常常超时。本脚本：
  1) 直接抓 https://www.bing.com/images/search?q=... 的 HTML，正则取出 murl(原图) 和 turl(缩略图)
  2) 先试原图，超时/失败就退回缩略图（缩略图在 bing 自己的 CDN 上，通常可达）
  3) 去重(平均哈希) + 尺寸/糊图过滤 + 记录来源清单

用法:
  python fetch_images_bing.py --dst raw_web_bb --queries "pink basketball" "colorful basketball"
  python fetch_images_bing.py --dst x --queries "篮球" --per-query 40 --min-side 200
"""
import argparse
import csv
import hashlib
import io
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


def http_get(url, timeout=12, referer=None):
    h = {"User-Agent": UA}
    if referer:
        h["Referer"] = referer
    return urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=timeout).read()


def search_bing(query, want=35):
    """返回 [(murl, turl)]"""
    url = ("https://www.bing.com/images/search?q=" + urllib.parse.quote(query)
           + f"&count={want}&first=1&qft=+filterui:imagesize-medium")
    html = http_get(url, timeout=20).decode("utf-8", "replace")
    # Bing 的 HTML 里 JSON 是 HTML 转义的（murl&quot;:&quot;...），先反转义再正则
    html = html.replace("&quot;", '"').replace("&amp;", "&")
    murls = re.findall(r'"murl":"(.*?)"', html)
    turls = re.findall(r'"turl":"(.*?)"', html)
    out = []
    for i, m in enumerate(murls):
        t = turls[i] if i < len(turls) else None
        out.append((m.replace("\\/", "/"), t.replace("\\/", "/") if t else None))
    return out


def decode(data):
    if not data or len(data) < 2048:
        return None
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def ahash(img, size=8):
    g = cv2.cvtColor(cv2.resize(img, (size, size)), cv2.COLOR_BGR2GRAY)
    return (g > g.mean()).flatten()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", required=True)
    ap.add_argument("--queries", nargs="+", required=True)
    ap.add_argument("--per-query", type=int, default=35)
    ap.add_argument("--min-side", type=int, default=200)
    ap.add_argument("--max-side", type=int, default=2000)
    ap.add_argument("--blur-thresh", type=float, default=25.0)
    ap.add_argument("--dedup-dist", type=int, default=4)
    ap.add_argument("--sleep", type=float, default=0.05)
    args = ap.parse_args()

    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    rows, hashes, seen = [], [], set()
    total = 0

    for q in args.queries:
        print(f"\n[Bing] {q}")
        try:
            cand = search_bing(q, args.per_query)
        except Exception as e:
            print(f"  搜索失败: {type(e).__name__}: {str(e)[:70]}")
            continue
        print(f"  搜到 {len(cand)} 条候选")
        kept = via_thumb = skipped = 0
        for murl, turl in cand:
            if murl in seen:
                continue
            seen.add(murl)
            img, used = None, "orig"
            for url, tag in ((murl, "orig"), (turl, "thumb")):
                if not url:
                    continue
                try:
                    img = decode(http_get(url, timeout=10, referer="https://www.bing.com/"))
                    if img is not None:
                        used = tag
                        break
                except Exception:
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
            if args.blur_thresh > 0 and cv2.Laplacian(
                    cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() < args.blur_thresh:
                skipped += 1
                continue
            hv = ahash(img)
            if any(int(np.count_nonzero(hv != x)) < args.dedup_dist for x in hashes):
                skipped += 1
                continue
            hashes.append(hv)
            name = hashlib.md5(murl.encode()).hexdigest()[:12] + ".jpg"
            cv2.imwrite(str(dst / name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            rows.append({"file": name, "query": q, "url": murl, "via": used,
                         "w": img.shape[1], "h": img.shape[0]})
            kept += 1
            total += 1
            via_thumb += int(used == "thumb")
            time.sleep(args.sleep)
        print(f"  保留 {kept}（其中缩略图 {via_thumb}）, 跳过 {skipped}")

    with open(dst / "manifest.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=["file", "query", "url", "via", "w", "h"])
        wr.writeheader()
        wr.writerows(rows)

    print(f"\n[done] 共保留 {total} 张 -> {dst.resolve()}")
    print("  下一步: python add_web_frames.py --src " + str(dst) +
          " --dst dataset_ball_v4 --scales 1.0,0.6,0.4 --conf 0.5 --class-ids 0 --max-boxes 2")


if __name__ == "__main__":
    main()
