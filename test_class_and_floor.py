# -*- coding: utf-8 -*-
"""专门测两件事（用户最关心的）：
   1) 篮球/排球的类间判错率（用新视频 + 已有验证集）
   2) **木地板会不会被误认成球**（在"纯地板帧"上统计误检数）
对比 v14 / v16 / v17。
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")
from train import install_single_process_pool

install_single_process_pool()

import cv2
import yaml
from ultralytics import YOLO

MODELS = [("v14", "runs/ball_v14/weights/best.pt"),
          ("v16", "runs/ball_v16/weights/best.pt"),
          ("v17", "runs/ball_v17/weights/best.pt")]


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0


def load_labelled(frames_spec):
    out = []
    for yml in frames_spec:
        d = yaml.safe_load(Path(yml).read_text(encoding="utf-8"))
        root = Path(d["path"])
        for line in (root / str(d["val"])).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            p = Path(line.strip())
            lab = root / "labels" / p.parent.name / (p.stem + ".txt")
            if not lab.exists():
                continue
            gts = []
            for l in lab.read_text(encoding="utf-8").splitlines():
                if l.strip():
                    c, cx, cy, bw, bh = l.split()
                    c = int(float(c))
                    if c in (0, 1):
                        gts.append((c, (float(cx) - float(bw) / 2, float(cy) - float(bh) / 2,
                                        float(cx) + float(bw) / 2, float(cy) + float(bh) / 2)))
            if gts:
                out.append((p, gts))
    return out


# 1) 分类测试集 = 人工修过的球集 + 网图集 + 新视频 val 帧
frames = load_labelled(["data_eval_ball11.yaml", "data_eval_web.yaml"])
v17 = yaml.safe_load(Path("dataset_ball_v17/data.yaml").read_text(encoding="utf-8"))
v17root = Path(v17["path"])
for line in (v17root / "val_ball.txt").read_text(encoding="utf-8").splitlines() if (v17root / "val_ball.txt").exists() else []:
    pass
# 新视频 val 帧（n1_/n2_/n3_）
for lab in sorted((v17root / "labels" / "val").glob("n?_*.txt")):
    p = v17root / "images" / "val" / (lab.stem + ".jpg")
    if not p.exists():
        continue
    img = cv2.imread(str(p))
    if img is None:
        continue
    h, w = img.shape[:2]
    gts = []
    for l in lab.read_text(encoding="utf-8").splitlines():
        if l.strip():
            c, cx, cy, bw, bh = l.split()
            c = int(float(c))
            if c in (0, 1):
                gts.append((c, (float(cx) - float(bw) / 2, float(cy) - float(bh) / 2,
                                float(cx) + float(bw) / 2, float(cy) + float(bh) / 2)))
    if gts:
        frames.append((p, gts))
print(f"分类测试集: {len(frames)} 帧, "
      f"框 {sum(len(g) for _, g in frames)} 个")

# 2) 纯地板帧（val 里的空标签帧）
floor = []
for lab in sorted((v17root / "labels" / "val").glob("*.txt")):
    if lab.read_text(encoding="utf-8").strip():
        continue
    p = v17root / "images" / "val" / (lab.stem + ".jpg")
    if p.exists():
        floor.append(p)
print(f"纯地板帧（负样本，验证用）: {len(floor)} 张\n")

print(f"{'模型':6s} {'篮球对':>6s} {'篮→排':>6s} {'排球对':>6s} {'排→篮':>6s} "
      f"{'类间判错率':>10s} {'地板误检帧':>10s} {'地板误检框':>10s}")
print("-" * 76)
for tag, mp in MODELS:
    if not Path(mp).exists():
        print(f"{tag:6s} （还没训好）")
        continue
    m = YOLO(mp)
    cm = Counter()
    for imgp, gts in frames:
        r = m.predict(str(imgp), conf=0.25, iou=0.45, imgsz=640, device=0, verbose=False)[0]
        preds = [(tuple(float(v) for v in b.xyxyn[0]), int(b.cls[0]))
                 for b in r.boxes if int(b.cls[0]) in (0, 1)]
        for gc, gbox in gts:
            best, bc = 0.0, None
            for pb, pc in preds:
                v = iou(gbox, pb)
                if v > best:
                    best, bc = v, pc
            cm[(gc, bc)] += 0
            if bc is not None and best >= 0.45:
                cm[(gc, bc)] += 1
            else:
                cm[(gc, "miss")] += 1
    b_ok, b_bad = cm[(0, 0)], cm[(0, 1)]
    v_ok, v_bad = cm[(1, 1)], cm[(1, 0)]
    matched = b_ok + b_bad + v_ok + v_bad
    wrong = b_bad + v_bad
    # 地板误检
    fp_frames = fp_boxes = 0
    for p in floor:
        r = m.predict(str(p), conf=0.25, iou=0.45, imgsz=640, device=0, verbose=False)[0]
        n = sum(1 for b in r.boxes if int(b.cls[0]) in (0, 1))
        if n:
            fp_frames += 1
            fp_boxes += n
    print(f"{tag:6s} {b_ok:6d} {b_bad:6d} {v_ok:6d} {v_bad:6d} "
          f"{wrong/max(1,matched)*100:9.2f}% {fp_frames:6d}/{len(floor):<3d} {fp_boxes:10d}")
