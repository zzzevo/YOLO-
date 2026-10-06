# 用 sample_raw 里两段视频的数据，从 onnx 回填的权重继续训练篮球/排球模型
# 流程: 基线评估 -> 微调 -> 复评 -> 导出 ONNX，每步结果都打印出来
import json
from pathlib import Path

from train import install_single_process_pool

install_single_process_pool()          # 受限环境: 同步假线程池

from ultralytics import YOLO           # noqa: E402

ROOT = Path(__file__).parent.resolve()
DATA = str(ROOT / "dataset_ball" / "data.yaml")
START = str(ROOT / "runs" / "recovered" / "best_recovered.pt")
EPOCHS = 100
report = {}


def metrics(r, tag):
    box = r.box
    row = {
        "mAP50": round(float(box.map50), 4),
        "mAP50-95": round(float(box.map), 4),
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
    }
    # 每类 mAP50 (all_ap 的第 0 列就是 IoU=0.5)
    try:
        names = r.names
        all_ap = box.all_ap
        idx = list(box.ap_class_index)
        per = {names[int(i)]: round(float(all_ap[n][0]), 4) for n, i in enumerate(idx)}
        row["per_class_mAP50"] = per
    except Exception as e:
        row["per_class_mAP50"] = f"n/a ({e})"
    print(f"[{tag}] {row}", flush=True)
    return row


print("=" * 72)
print("[1/4] 基线: 你上传的 onnx 回填权重，直接在验证集上评估")
print("=" * 72, flush=True)
base = YOLO(START)
report["baseline"] = metrics(base.val(data=DATA, imgsz=640, batch=16, device=0, workers=0,
                                      verbose=False, plots=False), "baseline")

print("\n" + "=" * 72)
print(f"[2/4] 继续训练 (从 {Path(START).name} 起, {EPOCHS} epochs)")
print("=" * 72, flush=True)
m = YOLO(START)
m.train(data=DATA, epochs=EPOCHS, imgsz=640, batch=16, device=0, workers=0,
        lr0=0.002, optimizer="MuSGD", close_mosaic=10, patience=30,
        project=str(ROOT / "runs"), name="ball_v1", exist_ok=True,
        plots=True, val=True, verbose=False, seed=0)

best = Path(m.trainer.save_dir) / "weights" / "best.pt"
print(f"\n训练完成, best = {best}", flush=True)

print("\n" + "=" * 72)
print("[3/4] 训练后复评")
print("=" * 72, flush=True)
m2 = YOLO(str(best))
report["finetuned"] = metrics(m2.val(data=DATA, imgsz=640, batch=16, device=0, workers=0,
                                     verbose=False, plots=False), "finetuned")

print("\n" + "=" * 72)
print("[4/4] 导出 ONNX")
print("=" * 72, flush=True)
onnx_path = m2.export(format="onnx", imgsz=640, opset=12, simplify=True, dynamic=False)
report["onnx"] = str(onnx_path)
print("[onnx]", onnx_path, flush=True)

(ROOT / "ball_train_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n汇总:")
for k, v in report.items():
    print(f"  {k}: {v}")
