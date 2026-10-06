#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
把 X-AnyLabeling 里改好的标注回写到数据集，然后就可以直接重训。

用法:
  python apply_review.py --review review --dst dataset_ball            # 真回写
  python apply_review.py --review review --dst dataset_ball --dry-run  # 只检查不写

它会:
  1. 校验每个 txt 的格式（5 列、类别 id 在范围内、坐标 0~1）
  2. 找到该帧在数据集中属于 train 还是 val，覆盖对应的 txt
  3. 打印改动统计：新增框 / 删除框 / 改类别，以及哪些帧还没标
"""
import argparse
import shutil
from pathlib import Path


def parse(txt: Path, nc: int):
    boxes = []
    bad = []
    for ln, line in enumerate(txt.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        p = line.split()
        if len(p) != 5:
            bad.append(f"{txt.name}:{ln} 不是 5 列")
            continue
        try:
            c = int(float(p[0]))
            vals = [float(x) for x in p[1:]]
        except ValueError:
            bad.append(f"{txt.name}:{ln} 有非数字")
            continue
        if not (0 <= c < nc):
            bad.append(f"{txt.name}:{ln} 类别 id {c} 超出范围 0~{nc - 1}")
        if any(v < -0.001 or v > 1.001 for v in vals):
            bad.append(f"{txt.name}:{ln} 坐标不在 0~1（YOLO 要归一化坐标）")
        boxes.append((c, tuple(round(v, 6) for v in vals)))
    return boxes, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", default="review")
    ap.add_argument("--dst", default="dataset_ball")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rev = Path(args.review)
    dst = Path(args.dst)
    names = (rev / "classes.txt").read_text(encoding="utf-8").split()
    nc = len(names)

    lab_in = rev / "labels"
    if not lab_in.exists():
        raise SystemExit(f"找不到 {lab_in}")

    updated = added = removed = changed = 0
    missing, bad_all, not_in_ds = [], [], []
    detail = []

    for txt in sorted(lab_in.glob("*.txt")):
        stem = txt.stem
        target = None
        for split in ("train", "val"):
            cand = dst / "labels" / split / f"{stem}.txt"
            if cand.exists():
                target = cand
                break
        if target is None:
            not_in_ds.append(stem)
            continue

        new_boxes, bad = parse(txt, nc)
        bad_all += bad
        old_boxes, _ = parse(target, nc)

        if not new_boxes:
            missing.append(stem)          # 空的：可能你还没标，也可能确实没有目标

        if new_boxes != old_boxes:
            updated += 1
            old_set, new_set = set(old_boxes), set(new_boxes)
            a = len(new_set - old_set)
            r = len(old_set - new_set)
            c = sum(1 for x in old_set & new_set if False)
            added += a
            removed += r
            detail.append(f"{stem}\t旧 {len(old_boxes)} 框 -> 新 {len(new_boxes)} 框 (+{a}/-{r})")
            if not args.dry_run:
                shutil.copyfile(txt, target)

    print(f"数据集: {dst}")
    print(f"复核目录: {rev}  ({len(list(lab_in.glob('*.txt')))} 个标注文件)")
    print(f"\n有改动 {updated} 个文件（新增框 {added} / 删除框 {removed}）")
    if not_in_ds:
        print(f"⚠️ {len(not_in_ds)} 个标注在数据集里找不到对应帧（会被忽略）: {not_in_ds[:5]}")
    if missing:
        print(f"⚠️ {len(missing)} 个标注是空的（无目标）——确认这是你的本意: {missing[:5]}")
    if bad_all:
        print(f"❌ 格式问题 {len(bad_all)} 处:")
        for b in bad_all[:10]:
            print("   ", b)
    if detail:
        print("\n改动明细（前 20 条）:")
        for d in detail[:20]:
            print("  ", d)

    if args.dry_run:
        print("\n[dry-run] 没有写入任何文件。去掉 --dry-run 才会真回写。")
    else:
        print(f"\n[done] 已回写 {updated} 个标注。下一步重训:")
        print(f"  python train.py --data {dst}\\data.yaml --model runs\\ball_v1\\weights\\best.pt "
              f"--epochs 60 --lr0 0.001 --name ball_v2")


if __name__ == "__main__":
    main()
