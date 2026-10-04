# 域差异对照实验：纯网图训练 vs 网图+少量实拍训练，用真实帧做交叉域测试
# 用法: python _exp_domain_gap.py   (在 conda yolo26 环境下、工作区受限时加 --single-process 的等价补丁)
import json
from pathlib import Path

from train import install_single_process_pool

install_single_process_pool()          # 受限环境：同步假线程池，绕过命名管道

from ultralytics import YOLO           # noqa: E402

ROOT = Path(__file__).parent.resolve()
EPOCHS = 50
results = {}


def train(data, name):
    print(f"\n{'=' * 70}\n[训练] {name}  data={data}\n{'=' * 70}", flush=True)
    m = YOLO("yolo26n.pt")
    m.train(data=str(ROOT / data / "data.yaml"), epochs=EPOCHS, imgsz=640, batch=16, device=0,
            workers=0, close_mosaic=10, optimizer="MuSGD", project=str(ROOT / "runs"), name=name,
            exist_ok=True, plots=False, val=True, verbose=False)
    best = Path(m.trainer.save_dir) / "weights" / "best.pt"
    print(f"[训练完成] {best}", flush=True)
    return str(best)


def evaluate(weights, data, tag):
    m = YOLO(weights)
    r = m.val(data=str(ROOT / data / "data.yaml"), imgsz=640, batch=16, device=0, workers=0,
              verbose=False, plots=False)
    row = {"mAP50": round(float(r.box.map50), 4), "mAP50-95": round(float(r.box.map), 4),
           "precision": round(float(r.box.mp), 4), "recall": round(float(r.box.mr), 4)}
    print(f"[评估] {tag} -> {row}", flush=True)
    return row


# 实验 A：只有网图
w_web = train("dataset_web", "exp_web")
results["A_纯网图_在网图验证集"] = evaluate(w_web, "dataset_web", "A 纯网图 -> 网图域(同域)")
results["A_纯网图_在真实帧"] = evaluate(w_web, "realtest", "A 纯网图 -> 真实帧(跨域)")

# 实验 B：网图 + 少量实拍（39 张真实帧）
w_mix = train("dataset_mix", "exp_mix")
results["B_网图+实拍_在网图验证集"] = evaluate(w_mix, "dataset_web", "B 混合 -> 网图域")
results["B_网图+实拍_在真实帧"] = evaluate(w_mix, "realtest", "B 混合 -> 真实帧(目标域)")

# 实验 C：只用实拍（39 张），看真实帧上限
w_real = train("dataset_real", "exp_real_only")
results["C_纯实拍_在真实帧"] = evaluate(w_real, "realtest", "C 纯实拍 -> 真实帧")

print("\n" + "=" * 70)
print("汇总 (mAP50 / mAP50-95 / P / R)")
print("=" * 70)
for k, v in results.items():
    print(f"{k:32s} {v['mAP50']:.4f}  {v['mAP50-95']:.4f}  {v['precision']:.4f}  {v['recall']:.4f}")
(ROOT / "domain_gap_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n已写入 domain_gap_results.json")
