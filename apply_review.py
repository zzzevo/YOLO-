#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
把标注回写到数据集，然后就可以直接重训。

支持两种来源：
  * X-AnyLabeling 的原生 JSON（`review/labels/*.json`）—— 标注时自动保存的那个
  * YOLO txt（如果你在工具里做了「导出 YOLO-HBB 标注」）

用法:
  python apply_review.py --review review --dst dataset_ball_ms              # 干跑，只看不改
  python apply_review.py --review review --dst dataset_ball_ms --write      # 真回写
  python apply_review.py --review review --dst dataset_ball_ms --fix "d=volleyball"   # 修手滑的类名

类名映射（中英都认）：
  basketball / 篮球 / 0  -> 0
  volleyball / 排球 / 1  -> 1
"""
import argparse
import json
from collections import Counter
from pathlib import Path

# 类名 -> 类别 id。中英混用都能认，避免手滑打错导致整张图作废。
ALIASES = {
    "basketball": 0, "篮球": 0, "0": 0,
    "volleyball": 1, "排球": 1, "1": 1,
}


def load_shapes(review: Path, fixes: dict, aliases: dict):
    """返回 {stem: (boxes, source)}，boxes = [(cls, cx, cy, w, h)]"""
    lab_dir = review / "labels"
    out, unknown = {}, Counter()

    # --- 1) 优先用 JSON（X-AnyLabeling 原生，信息最全：含图片宽高）---
    for j in sorted(lab_dir.glob("*.json")):
        d = json.loads(j.read_text(encoding="utf-8"))
        w, h = d.get("imageWidth"), d.get("imageHeight")
        if not w or not h:
            continue
        boxes = []
        for s in d.get("shapes", []):
            raw = str(s.get("label", "")).strip()
            cls = aliases.get(raw, fixes.get(raw))
            if cls is None:
                unknown[raw] += 1
                continue
            if s.get("shape_type") not in ("rectangle", "polygon", "quadrilateral"):
                unknown[f"shape:{s.get('shape_type')}"] += 1
                continue
            xs = [p[0] for p in s["points"]]
            ys = [p[1] for p in s["points"]]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            if bw <= 0 or bh <= 0:
                continue
            boxes.append((cls, round((x1 + x2) / 2 / w, 6), round((y1 + y2) / 2 / h, 6),
                          round(bw, 6), round(bh, 6)))
        out[j.stem] = (boxes, "json")

    # --- 2) 没有 JSON 的用 YOLO txt ---
    for t in sorted(lab_dir.glob("*.txt")):
        if t.stem in out:
            continue
        boxes, bad = [], 0
        for line in t.read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) != 5:
                continue
            try:
                boxes.append((int(float(p[0])), *[round(float(x), 6) for x in p[1:]]))
            except ValueError:
                bad += 1
        out[t.stem] = (boxes, "txt")
        if bad:
            unknown[f"txt 格式问题 x{bad}"] += 1

    return out, unknown


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", default="review")
    ap.add_argument("--dst", default="dataset_ball_ms")
    ap.add_argument("--write", action="store_true", help="真回写（不加就是干跑）")
    ap.add_argument("--fix", default="", help='修正类名，如 "d=volleyball,robot0=basketball"')
    args = ap.parse_args()

    fixes = {}
    for kv in args.fix.replace("，", ",").split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            fixes[k.strip()] = ALIASES.get(v.strip(), fixes.get(v.strip()))

    rev, dst = Path(args.review), Path(args.dst)
    data, unknown = load_shapes(rev, fixes, ALIASES)

    print(f"复核目录: {rev}")
    print(f"读入 {len(data)} 个标注文件 "
          f"(JSON {sum(1 for v in data.values() if v[1] == 'json')} / "
          f"YOLO txt {sum(1 for v in data.values() if v[1] == 'txt')})")
    if unknown:
        print("⚠️ 无法识别的标签（这些框会被跳过，请用 --fix 指定）:")
        for k, v in unknown.most_common():
            print(f"     {k!r} x{v}")

    updated, added, removed, empty, not_in_ds, appended = 0, 0, 0, [], [], 0
    detail = []
    for stem, (new_boxes, src) in sorted(data.items()):
        target = None
        for split in ("train", "val"):
            cand = dst / "labels" / split / f"{stem}.txt"
            if cand.exists():
                target = cand
                break
        if target is None:
            # 数据集里没有这帧（当初因为"模型看不见"被丢了）——把它补进 train
            img_src = rev / "images" / f"{stem}.jpg"
            if img_src.exists() and new_boxes:
                not_in_ds.append(stem)
                if args.write:
                    import shutil
                    (dst / "images" / "train").mkdir(parents=True, exist_ok=True)
                    (dst / "labels" / "train").mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(img_src, dst / "images" / "train" / img_src.name)
                    target = dst / "labels" / "train" / f"{stem}.txt"
                    target.write_text("\n".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                                                for c, cx, cy, w, h in new_boxes), encoding="utf-8")
                    appended += 1
                    detail.append(f"{stem}\t新增帧 -> train ({len(new_boxes)} 框)")
                else:
                    detail.append(f"{stem}\t数据集里没有, 将新增为 train ({len(new_boxes)} 框)")
            continue

        old_boxes = []
        for line in target.read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) == 5:
                old_boxes.append((int(float(p[0])), *[round(float(x), 6) for x in p[1:]]))

        if not new_boxes:
            empty.append(stem)
        if set(new_boxes) != set(old_boxes):
            updated += 1
            a = len(set(new_boxes) - set(old_boxes))
            r = len(set(old_boxes) - set(new_boxes))
            added += a
            removed += r
            detail.append(f"{stem}\t{len(old_boxes)} -> {len(new_boxes)} 框 (+{a}/-{r})")
            if args.write:
                target.write_text("\n".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                                            for c, cx, cy, w, h in new_boxes), encoding="utf-8")

    print(f"\n有改动 {updated} 个文件（新增框 {added} / 删除框 {removed}）")
    if not_in_ds:
        print(f"{'已新增' if args.write else '将新增'} {appended if args.write else len(not_in_ds)} "
              f"帧到 train（原本因为'模型看不见'被丢掉的那些）: {not_in_ds[:8]}")
    if empty:
        print(f"⚠️ {len(empty)} 个标注是空的（无目标）——确认这是你的本意: {empty[:5]}")
    if detail:
        print("\n改动明细:")
        for d in detail:
            print("  ", d)

    print("\n[dry-run] 没有写入。加 --write 才真回写。" if not args.write
          else f"\n[done] 已回写 {updated} 个标注。下一步重训:\n"
               f"  python train.py --data {dst}\\data.yaml --model runs\\ball_v2\\weights\\best.pt "
               f"--epochs 60 --lr0 0.001 --name ball_v3 --single-process")


if __name__ == "__main__":
    main()
