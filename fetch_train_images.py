#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
大批量抓「能直接训练」的篮球/排球网图，并且**不标错**。

三重过滤（每一层都实测有效）：
  1) 搜索层：`qft=+filterui:photo-photo` —— 只要照片，挡掉海报/logo/文字图/地图/卡通
  2) 落盘前：YOLO 必须检出球（conf 门槛）→ 再把**框裁出来**给 CLIP 判"是不是球"
     （模型会把橙子/玫瑰/彩虹/人脸当球，CLIP 是语义模型，能分开）
  3) **类别以 CLIP 判定为准**：CLIP 判"篮球"就标 0，判"排球"就标 1；
     模型和 CLIP 不一致时记下来（默认信 CLIP，避免把排球标成篮球这种错标）；
     CLIP 自己也没把握（两类概率都低）→ **直接丢掉**，宁缺毋滥

其它：内容级去重(平均哈希)、按关键词统计产出率、产出可直接进数据集的 images/ + labels/ + manifest.csv

用法:
  python fetch_train_images.py --dst raw_web_v2 --per-query 35 --queries "..." "..."
  python fetch_train_images.py --dst raw_web_v2 --queries-file queries.txt
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

from fetch_images_bing import ahash, decode, http_get
from infer_onnx import CLASS_NAMES, Detector

# CLIP 的候选语义（前两个是我们要的球，其余是否定项）
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
    ap.add_argument("--queries", nargs="*", default=[])
    ap.add_argument("--queries-file", help="每行一个关键词")
    ap.add_argument("--per-query", type=int, default=35)
    ap.add_argument("--model", default="runs/ball_v11/weights/best.onnx")
    ap.add_argument("--conf", type=float, default=0.45, help="YOLO 检出球的门槛")
    ap.add_argument("--clip-thr", type=float, default=0.90, help="CLIP 判'是球'的概率门槛")
    ap.add_argument("--cls-thr", type=float, default=0.60,
                    help="CLIP 判'篮球/排球'的概率门槛；低于它说明 CLIP 也没把握 -> 丢掉不标")
    ap.add_argument("--keep-model-class", action="store_true",
                    help="类别仍按模型判定（默认按 CLIP 判定，更不容易标错）")
    ap.add_argument("--clip-root", default="weights/clip")
    ap.add_argument("--min-side", type=int, default=200)
    ap.add_argument("--max-side", type=int, default=2000)
    ap.add_argument("--max-boxes", type=int, default=3)
    ap.add_argument("--no-photo-only", dest="photo_only", action="store_false")
    ap.add_argument("--sleep", type=float, default=0.04)
    ap.add_argument("--timeout", type=float, default=6.0, help="单张图下载超时(秒)")
    ap.add_argument("--query-sleep", type=float, default=3.0, help="每两个关键词之间歇一下，避免被限流")
    ap.add_argument("--query-budget", type=float, default=75.0,
                    help="单个关键词最多花多少秒，超了就跳过（有些词的结果全在连不上的 CDN 上）")
    ap.add_argument("--fail-fast", type=int, default=8,
                    help="连续下载失败多少张就放弃这个关键词")
    ap.add_argument("--prefer-thumb", action="store_true",
                    help="先用 Bing 缩略图(在自己的 CDN 上，通常可达)，失败再试原图")
    args = ap.parse_args()

    queries = list(args.queries)
    if args.queries_file:
        queries += [l.strip() for l in Path(args.queries_file).read_text(encoding="utf-8").splitlines()
                    if l.strip() and not l.strip().startswith("#")]
    if not queries:
        raise SystemExit("没有关键词")

    import torch
    import clip
    from PIL import Image
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cmodel, cprep = clip.load("ViT-B/32", device=dev, download_root=args.clip_root)
    with torch.no_grad():
        tf = cmodel.encode_text(clip.tokenize(LABELS).to(dev))
        tf = tf / tf.norm(dim=-1, keepdim=True)

    def clip_eval(crop):
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            f = cmodel.encode_image(cprep(Image.fromarray(rgb)).unsqueeze(0).to(dev))
            f = f / f.norm(dim=-1, keepdim=True)
            p = (100 * f @ tf.T).softmax(dim=-1)[0].cpu().numpy()
        return float(p[0]), float(p[1]), LABELS[int(p.argmax())]

    det = Detector(args.model, 640, 0.10)          # 低阈值取原始输出，再用 --conf 筛
    dst = Path(args.dst)
    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "labels").mkdir(parents=True, exist_ok=True)

    hashes, rows, seen = [], [], set()
    G = {"cand": 0, "down": 0, "kept": 0, "noball": 0, "lowconf": 0,
         "clipno": 0, "clsunsure": 0, "dup": 0, "disagree": 0}
    per_q = []

    for q in queries:
        print(f"\n[搜索] {q}", flush=True)
        try:
            cand = search_bing(q, args.per_query, args.photo_only)
        except Exception as e:
            print(f"  搜索失败: {type(e).__name__}: {str(e)[:60]}")
            continue
        st = {"cand": len(cand), "kept": 0, "clipno": 0, "clsunsure": 0, "dup": 0,
              "noball": 0, "dlfail": 0, "timeout": 0}
        q_t0 = time.time()
        consec_fail = 0
        for murl, turl in cand:
            if murl in seen:
                continue
            seen.add(murl)
            order = (turl, murl) if args.prefer_thumb else (murl, turl)
            img = None
            for attempt in range(2):                       # 失败重试一次
                for url in order:
                    if not url:
                        continue
                    try:
                        img = decode(http_get(url, timeout=args.timeout,
                                              referer="https://www.bing.com/"))
                        if img is not None:
                            break
                    except Exception:
                        continue
                if img is not None:
                    break
            if img is None:
                consec_fail += 1
                st["dlfail"] = st.get("dlfail", 0) + 1
                if consec_fail >= args.fail_fast:
                    st["timeout"] = 1
                    print(f"  [熔断] 连续 {consec_fail} 张下载失败，跳过本关键词", flush=True)
                    break
                continue
            consec_fail = 0
            if time.time() - q_t0 > args.query_budget:
                st["timeout"] = 1
                print(f"  [超时] 本关键词已花 {time.time()-q_t0:.0f}s，跳到下一个", flush=True)
                break
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
            dets = [d for d in det(img)[0] if d["xyxy"] and d["conf"] >= args.conf]
            if not dets:
                st["noball"] += 1
                continue
            good = []
            for d in dets[: args.max_boxes]:
                x1, y1, x2, y2 = [int(v) for v in d["xyxy"]]
                pad = int(max(x2 - x1, y2 - y1) * 0.08)
                crop = img[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)]
                if crop.size == 0:
                    continue
                pb, pv, top = clip_eval(crop)
                if (pb + pv) < args.clip_thr:            # ② 不是球 -> 丢
                    st["clipno"] += 1
                    continue
                # ③ 类别以 CLIP 为准
                clip_cls = 0 if pb >= pv else 1
                clip_p = max(pb, pv)
                if clip_p < args.cls_thr:                # CLIP 也没把握 -> 丢，宁缺毋滥
                    st["clsunsure"] += 1
                    continue
                model_cls = int(d["cls"])
                final_cls = model_cls if args.keep_model_class else clip_cls
                if model_cls != clip_cls:
                    G["disagree"] += 1
                good.append({"cls": final_cls, "model_cls": model_cls, "clip_cls": clip_cls,
                             "clip_p": clip_p, "conf": float(d["conf"]), "xyxy": d["xyxy"]})
            if not good:
                continue
            hashes.append(hv)
            name = hashlib.md5(murl.encode()).hexdigest()[:12]
            cv2.imwrite(str(dst / "images" / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            lines = []
            for b in good:
                x1, y1, x2, y2 = b["xyxy"]
                lines.append(f"{b['cls']} {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} "
                             f"{(x2-x1)/w:.6f} {(y2-y1)/h:.6f}")
            (dst / "labels" / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")
            for b in good:
                rows.append({"file": f"{name}.jpg", "query": q,
                             "label": CLASS_NAMES[b["cls"]],
                             "model_said": CLASS_NAMES[b["model_cls"]],
                             "clip_said": CLASS_NAMES[b["clip_cls"]],
                             "agree": int(b["model_cls"] == b["clip_cls"]),
                             "clip_p": round(b["clip_p"], 3), "yolo_conf": round(b["conf"], 3),
                             "url": murl})
            st["kept"] += 1
            time.sleep(args.sleep)
        print(f"  候选 {st['cand']} -> 留用 {st['kept']}（无球 {st['noball']} / CLIP否 {st['clipno']}"
              f" / 类别没把握 {st['clsunsure']} / 重复 {st['dup']}）", flush=True)
        per_q.append((q, st))
        time.sleep(args.query_sleep)
        for k in G:
            if k in st:
                G[k] += st[k]

    with open(dst / "manifest.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=["file", "query", "label", "model_said", "clip_said",
                                           "agree", "clip_p", "yolo_conf", "url"])
        wr.writeheader()
        wr.writerows(rows)

    print("\n" + "=" * 66)
    print(f"候选 {G['cand']} -> **留用 {G['kept']} 张**（含 {len(rows)} 个框）")
    print(f"丢弃: 无球 {G['noball']} / CLIP 判不是球 {G['clipno']} / "
          f"CLIP 也分不清哪类 {G['clsunsure']} / 重复 {G['dup']}")
    print(f"模型与 CLIP 类别不一致的框: {G['disagree']} 个（标签已按 CLIP 定）")
    print("\n各关键词产出:")
    for q, st in sorted(per_q, key=lambda x: -x[1]["kept"])[:20]:
        print(f"  {st['kept']:3d}/{st['cand']:3d}  {q}")
    print(f"\n[done] 图片+标签就绪 -> {dst.resolve()}")


if __name__ == "__main__":
    main()
