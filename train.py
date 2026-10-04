#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLO26n 微调 + 导出 ONNX（一次搞定）。

用法:
  # 从官方预训练微调（推荐，权重会自动下载）
  python train.py --data D:\\datasets\\robocup_balls\\data.yaml --name yolo26n_v1

  # 从「onnx 回填出来的 .pt」继续（backbone/neck 带你原模型的先验）
  python train.py --data ...\\data.yaml --model runs\\recovered\\best_recovered.pt --name yolo26n_v2 --lr0 0.002

  # 断点续训
  python train.py --resume runs\\robocup\\yolo26n_v1\\weights\\last.pt

  # 只导出
  python train.py --export-only runs\\robocup\\yolo26n_v1\\weights\\best.pt
"""
import argparse
import os
from pathlib import Path


class _SyncPool:
    """同步版假线程池：接口对齐 multiprocessing.pool.ThreadPool，只是不并行。

    为什么需要它：ultralytics 扫描数据集标签(data/dataset.py:cache_labels)和把图片
    缓存进内存(data/base.py)时会创建 ThreadPool，而 ThreadPool 内部靠"命名管道"通信。
    在受限环境(如 DSH 沙箱)里创建命名管道会被拒绝，报 PermissionError: [WinError 5]。
    在普通终端里训练**不需要**这个开关。
    """

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def imap(self, func=None, iterable=(), **_):
        return (func(x) for x in iterable)

    def imap_unordered(self, func=None, iterable=(), **_):
        return (func(x) for x in iterable)

    def map(self, func, iterable, **_):
        return [func(x) for x in iterable]

    def starmap(self, func, iterable, **_):
        return [func(*x) for x in iterable]

    def apply(self, func, args=(), **_):
        return func(*args)

    def close(self):
        pass

    def join(self):
        pass

    def terminate(self):
        pass


def install_single_process_pool():
    import multiprocessing.pool as mp_pool
    mp_pool.ThreadPool = _SyncPool      # 必须在 import ultralytics 之前替换
    print("[single-process] 已用同步假线程池替换 multiprocessing 池（受限环境模式）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="data.yaml 路径")
    ap.add_argument("--model", default="yolo26n.pt", help="yolo26n.pt / yolo26n.yaml / 自定义 .pt")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16, help="8GB 显存建议 16，OOM 就降 8")
    ap.add_argument("--device", default="0", help="0=第一块 GPU, cpu=CPU")
    ap.add_argument("--workers", type=int, default=8, help="Windows 上报错就改 0")
    ap.add_argument("--lr0", type=float, default=0.01, help="微调小数据集建议 0.001~0.01")
    ap.add_argument("--optimizer", default="MuSGD", help="YOLO26 官方配方用的是 MuSGD；也可 auto/AdamW/SGD")
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--close-mosaic", type=int, default=10, help="最后 N 轮关闭 mosaic; 调试时设 0")
    ap.add_argument("--project", default="runs/robocup")
    ap.add_argument("--name", default="yolo26n_v1")
    ap.add_argument("--resume", help="传 last.pt 续训，其余参数忽略")
    ap.add_argument("--export-only", help="只把指定 .pt 导出为 ONNX")
    ap.add_argument("--opset", type=int, default=12)
    ap.add_argument("--cache", default="ram", help="ram/disk/false；数据集小用 ram 最快")
    ap.add_argument("--single-process", action="store_true",
                    help="受限/沙箱环境用：同步假线程池，绕过命名管道限制（普通终端不需要）")
    args = ap.parse_args()

    if args.single_process or os.environ.get("YOLO_SINGLE_PROCESS") == "1":
        install_single_process_pool()

    from ultralytics import YOLO

    if args.export_only:
        m = YOLO(args.export_only)
        p = m.export(format="onnx", imgsz=args.imgsz, opset=args.opset, simplify=True, dynamic=False)
        print(f"[export] -> {p}")
        return

    if args.resume:
        model = YOLO(args.resume)
        model.train(resume=True)
    else:
        if not args.data:
            raise SystemExit("训练必须给 --data（data.yaml 路径）")
        if not Path(args.data).is_file():
            raise SystemExit(f"找不到 {args.data}")
        cache = args.cache if args.cache.lower() not in ("false", "none", "0") else False
        # project 必须是绝对路径: 传相对名时 ultralytics 会拼成 runs/detect/runs/xxx 的嵌套目录
        project = str(Path(args.project).resolve())
        model = YOLO(args.model)
        model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            optimizer=args.optimizer,
            lr0=args.lr0,
            patience=args.patience,
            close_mosaic=args.close_mosaic,   # 最后 N 轮关掉 mosaic，收敛更稳
            cache=cache,             # 数据集不大时放内存，快很多
            plots=True,
            project=project,
            name=args.name,
            exist_ok=False,
        )

    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    print(f"\n[done] best = {best}")
    print(f"  导出 ONNX: python train.py --export-only {best}")
    print(f"  回到 MaixCam: 还需要 ONNX -> .cvimodel（量化需校准图），"
          f"且现有 nn.YOLOv5 后处理代码要改成 4+nc 无 objectness 的解析方式")


if __name__ == "__main__":
    main()
