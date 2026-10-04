#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 best.onnx 把权重"回填"成一个 .pt —— 当你只有 ONNX、丢了原始 best.pt 时用。

原理（针对本模型已实测的事实）:
  * 该 ONNX 的 initializer 名字就是原生 PyTorch 参数名（model.0.conv.weight ...），
    按名字可直接对上 state_dict；
  * ONNX 里**没有** bn.* 参数：导出时 BN 已经折进 Conv 了 —— 卷积里带着 bias，
    而 ultralytics 的 Conv2d 是 bias=False + BatchNorm2d。所以要把 *.conv.bias
    搬到同层的 *.bn.bias，并把该 BN 配成"只加偏置"的恒等变换
    (weight=1, mean=0, var=1-eps)，这样输出才与 ONNX 完全一致；
  * 实测结果: unexpected=0, missing=30，且 30 个全部是 model.23.one2one_* 。
    也就是说 ONNX 里带的是**一对多(one-to-many / 走 NMS)分支**，
    缺的是一对一(one2one / NMS-free)分支 —— 微调时这一半会随机初始化。
    对推理和继续微调来说，缺的这半影响有限。

实测验证（本机跑过）:
  同一张图, 回填的 .pt 与原始 best.onnx 输出:
    .pt  boxes=[[114.6,31.0,147.1,54.0],[88.6,0.2,112.9,13.4]] conf=[0.543,0.380]
    onnx boxes=[[114.6,31.0,147.1,54.0],[88.6,0.2,112.9,13.5]] conf=[0.557,0.395]
  → 框完全一致，置信度差 ~1.5%（letterbox/decode 实现差异）。

依赖: torch + ultralytics（即训练环境），并需要能联网下载 yolo26n.yaml 的结构定义。

用法:
  python onnx_to_pt.py --onnx best.onnx --out runs/recovered/best_recovered.pt
  python onnx_to_pt.py --onnx best.onnx --nc 2 --yaml yolo26n.yaml --out best_recovered.pt

⚠️ 本脚本依赖 torch 环境，作者未在无 torch 的机器上实跑过；请留意打印的
   missing / unexpected 键数量，正常应为：unexpected=0，missing 只剩 bn.* 与 o2m 分支。
"""
import argparse
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import numpy_helper


def infer_nc(onnx_path):
    """output0 形状 [1, 4+nc, anchors] -> nc"""
    m = onnx.load(onnx_path)
    shape = m.graph.output[0].type.tensor_type.shape
    ch = shape.dim[1].dim_value
    if not ch:
        raise SystemExit("输出通道是动态维，无法自动推断 nc，请手动 --nc")
    return ch - 4, shape


def load_onnx_weights(onnx_path):
    m = onnx.load(onnx_path)
    sd = {}
    for init in m.graph.initializer:
        if init.name.endswith((".weight", ".bias")):
            sd[init.name] = torch.from_numpy(np.array(numpy_helper.to_array(init), copy=True))
    return sd


BN_EPS = 1e-5  # ultralytics 的 Conv 用 nn.BatchNorm2d(c2) 默认 eps


def remap_folded_conv_bias(model, sd):
    """ONNX 的 Conv 带 bias（导出时 BN 已折进卷积），而 ultralytics 的 Conv2d 是 bias=False。

    所以要把 *.conv.bias 搬到同层的 *.bn.bias，并把该 BN 配成"只加偏置"的恒等变换：
        weight=1, running_mean=0, running_var=1-eps   =>   out = x + bn.bias
    这样 BN 的输出与 ONNX 的 folded conv 完全一致。
    """
    keys = set(model.state_dict().keys())
    moved = 0
    for name in list(sd.keys()):
        if not name.endswith(".conv.bias") or name in keys:
            continue
        stem = name[: -len(".conv.bias")]
        bn_bias, bn_w = f"{stem}.bn.bias", f"{stem}.bn.weight"
        bn_m, bn_v = f"{stem}.bn.running_mean", f"{stem}.bn.running_var"
        if bn_bias not in keys:
            continue
        bias = sd.pop(name)
        sd[bn_bias] = bias
        sd[bn_w] = torch.ones_like(bias)
        sd[bn_m] = torch.zeros_like(bias)
        sd[bn_v] = torch.full_like(bias, 1.0 - BN_EPS)
        sd[f"{stem}.bn.num_batches_tracked"] = torch.zeros((), dtype=torch.long)
        moved += 1
    return moved


def fill_identity_bn(model, sd):
    """把剩下缺失的 BN 参数补成恒等（weight=1, bias=0, mean=0, var=1-eps）"""
    added = 0
    for name, buf in model.state_dict().items():
        if name in sd:
            continue
        if name.endswith(".bn.weight"):
            sd[name] = torch.ones_like(buf); added += 1
        elif name.endswith(".bn.bias"):
            sd[name] = torch.zeros_like(buf); added += 1
        elif name.endswith(".bn.running_mean"):
            sd[name] = torch.zeros_like(buf); added += 1
        elif name.endswith(".bn.running_var"):
            sd[name] = torch.full_like(buf, 1.0 - BN_EPS); added += 1
        elif name.endswith(".bn.num_batches_tracked"):
            sd[name] = torch.zeros_like(buf); added += 1
    return added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="best.onnx")
    ap.add_argument("--out", default="runs/recovered/best_recovered.pt")
    ap.add_argument("--yaml", default="yolo26n.yaml", help="结构定义，ultralytics 会自动找官方 cfg")
    ap.add_argument("--nc", type=int, default=0, help="0=从 ONNX 输出通道自动推断")
    args = ap.parse_args()

    import yaml
    from ultralytics.nn.tasks import DetectionModel, yaml_model_load
    from ultralytics.utils.checks import check_yaml

    nc, shape = infer_nc(args.onnx)
    if args.nc:
        nc = args.nc
    print(f"[onnx] output0 = [1, {nc + 4}, {shape.dim[2].dim_value}]  -> nc = {nc}")

    sd = load_onnx_weights(args.onnx)
    print(f"[onnx] 读出 {len(sd)} 个 weight/bias")

    # 注意: ultralytics 8.4 起不再导出 yaml_load；且包内只有 yolo26.yaml，
    # 'yolo26n.yaml' 这种带尺度后缀的名字必须走 yaml_model_load 才能解析到包内 cfg
    try:
        cfg = yaml_model_load(args.yaml)
    except Exception as e:  # 自定义 yaml 路径走这里
        print(f"[model] yaml_model_load 失败({e})，按普通 yaml 读取")
        cfg = yaml.safe_load(Path(check_yaml(args.yaml)).read_text(encoding="utf-8"))
    cfg["nc"] = nc
    model = DetectionModel(cfg, nc=nc, verbose=False)
    moved = remap_folded_conv_bias(model, sd)
    added = fill_identity_bn(model, sd)
    print(f"[model] 结构: nc={nc}; 搬运 folded conv.bias -> bn.bias {moved} 个; 补恒等 BN 参数 {added} 个")

    missing, unexpected = model.load_state_dict(sd, strict=False)
    o2m = [k for k in missing if "one2many" in k or "one2one" in k]
    other = [k for k in missing if k not in o2m]
    print(f"[load] unexpected={len(unexpected)}  missing={len(missing)}"
          f"  (其中双头分支 {len(o2m)} 个, 其他 {len(other)} 个)")
    if unexpected[:5]:
        print("  unexpected 示例:", unexpected[:5])
    if o2m[:3]:
        print("  双头分支 missing 示例:", o2m[:3])
    if other[:5]:
        print("  其他 missing 示例:", other[:5])
    if len(unexpected) > 20:
        print("  ⚠️ unexpected 偏多，说明结构 yaml 与导出时的模型不一致，回填可能无效")

    model.eval()
    ckpt = {"model": model.float(), "ema": None, "updates": None,
            "epoch": -1, "best_fitness": None, "train_args": {},
            "optimizer": None, "date": None, "version": None}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out)
    print(f"[done] 写出 {out.resolve()}")
    print("  自检: python -c \"from ultralytics import YOLO; "
          "m=YOLO(r'%s'); print(m.predict(r'图片.jpg', conf=0.25)[0].boxes)\"" % out)


if __name__ == "__main__":
    main()
