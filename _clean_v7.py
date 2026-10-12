# -*- coding: utf-8 -*-
"""把本轮爬的网图/购物图从 dataset_ball_v7 里撤掉（绿色球帧 web_bb__* 要保留）。"""
import re
from collections import Counter
from pathlib import Path

root = Path("dataset_ball_v7")
pat = re.compile(r"^web_[0-9a-f]{6,14}\.jpg$")
removed = Counter()

for sp in ("train", "val"):
    for p in sorted((root / "images" / sp).glob("*.jpg")):
        name = p.name
        is_web = bool(pat.match(name))
        is_shop = name.startswith("shop_")
        if is_web or is_shop:
            lab = root / "labels" / sp / (p.stem + ".txt")
            if lab.exists():
                lab.unlink()
            p.unlink()
            removed["网图" if is_web else "购物图"] += 1

print("撤掉:", dict(removed))
for sp in ("train", "val"):
    c = Counter()
    for p in (root / "images" / sp).glob("*.jpg"):
        n = p.name
        if n.startswith("web_bb_"):
            k = "绿球帧(web_bb_)"
        elif n.startswith("s_"):
            k = "新视频帧(s_)"
        elif n.startswith("pl_"):
            k = "定位柱(pl_)"
        elif n.startswith("vb_"):
            k = "新排球(vb_)"
        else:
            k = "原始球帧"
        c[k] += 1
    print(f"{sp}: 共 {sum(c.values())} 张 -> {dict(c)}")

# 数据集的类别名对不对
print((root / "data.yaml").read_text(encoding="utf-8").strip().splitlines()[-3:])
