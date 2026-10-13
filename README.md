# YOLO26n 训练 / 部署工作台（robocup_balls 模型）

针对你给的 `best.onnx`。**核心结论：ONNX 不能直接训练**，它只有冻结的权重和前向图。
训练必须要有 **①数据集** + **②`.pt` 权重（或官方预训练权重）**。

---

## ☝️ 用哪个模型？（先看这里，别选错）

| 用哪个 | 路径 | 输入 → 输出 | 类别 |
|---|---|---|---|
| ✅ **就用这个** | `runs/ball_v11/weights/best.onnx` | 640 → `[1,7,8400]` | 0=篮球 1=排球 2=定位柱 |
| ⭕ 备选（慢一倍，精度相同）| `runs/ball_v11/weights/best_960.onnx` | 960 → `[1,7,18900]` | 同上 |
| ❌ 别用 | `runs/ball_v1~v4/...` | 640 → `[1,**6**,8400]` | **只有 2 类（无定位柱）** |
| ❌ 别用 | `runs/ball_v5~v9/...` | 640 → `[1,7,8400]` | 3 类历史版本 |
| ❌ 别用 | `runs/yolo26n_v1`、`exp_mix`、`smoke` | — | 实验残留 |

**`best.onnx` vs `best_960.onnx` 怎么快速区分**（不用跑推理）：
看**输出形状**——`8400` 是 640 输入（80²+40²+20²），`18900` 是 960 输入（120²+60²+30²）。
日常用 640 那个：实测远距离定位柱召回已经是 **100%**，960 只在"目标更小更远"的极端情况下才需要。

---

## 一、这个模型到底是什么（已实测，非猜测）

> 🎓 **完全没接触过深度学习？先看 [YOLO零基础入门.md](YOLO零基础入门.md)**：
> 从"训练到底是什么"讲起，用你项目的真实数字解释所有专有名词，并附"我下载了什么"的完整清单。

用 `infer_onnx.py` / ONNX protobuf 直接解析出来的事实：

| 项目 | 值 |
|---|---|
| 来源 | `Ultralytics YOLO26n model trained on /content/robocup_balls/data.yaml` |
| 导出 | PyTorch 2.9.0 → ONNX opset 12, simpler, batch=1, 静态形状 |
| 输入 | `images` = `[1, 3, 640, 640]` float32, RGB, /255 |
| 输出 | `output0` = `[1, 6, 8400]` = **4 box + 2 类**，**没有 objectness**（v8+ 风格） |
| 结构 | 末端 `model.22` 含 `attn/qkv`、`attn/pe`、`ffn` → YOLO26 的注意力块 |
| 权重 | 108 个 conv weight/bias，名字就是原生 PyTorch 名（可回填，见第四节） |
| 无 BN 参数 | BN 已折进 Conv → 回填时 BN 需设成恒等 |

实测推理（CPU，onnxruntime 1.30）：单帧 **~28 ms**。

✅ **类别已确认为「篮球 / 排球」**（2026-10 用裁剪图肉眼核验，见第九节）：

| class id | 类别 | 核验证据 |
|---|---|---|
| 0 | **篮球** | 39 个检测裁剪全是橙色篮球；HSV 色相 7 / 饱和 140 |
| 1 | **排球** | 42 个裁剪全是蓝黄白 Mikasa 排球；中位置信度 0.82 |

ONNX 里不含类名，类名写在 `infer_onnx.py` 顶部的 `CLASS_NAMES`（已订正为 `["篮球", "排球"]`），
**顺序必须与原训练一致**，否则 6 通道解读就错位了。

---

## 二、目录里有什么

| 文件 | 作用 |
|---|---|
| `best.onnx` | 你的模型（工作区内可用副本） |
| `infer_onnx.py` | 纯 CPU 推理 / 视频抽帧验证 / 自动标注，**已跑通** |
| `extract_frames.py` | 视频抽帧（带糊图过滤 + 相似帧去重），给 build_dataset 当输入 |
| `build_dataset.py` | 用 `best.onnx` 给未标注图片自动打标，直接生成 YOLO 数据集骨架 |
| `onnx_to_pt.py` | 只有 ONNX、丢了 `.pt` 时把权重回填成 `.pt`，**已验证与 ONNX 输出一致** |
| `train.py` | YOLO26n 微调 + 导出 ONNX 一条龙 |
| `app.py` | **图形控制台**（Gradio 网页）：采集/标注/训练/推理 |
| `fetch_images.py` | 联网搜图采集（去重/去糊/尺寸过滤 + 来源清单） |
| `auto_label_world.py` | 开放词汇自动标注（YOLO-World + 文字提示词） |
| `merge_datasets.py` | 合并多个数据集（网图 + 实拍），做混合训练 |
| `_exp_domain_gap.py` | 域差异对照实验脚本（纯网图 vs 混合 vs 纯实拍） |
| `domain_gap_compare.jpg` | 同一真实帧上三种模型的直观对比 |
| `domain_gap_results.json` | 上表的原始数字 |
| `setup_env.ps1` | 建训练环境的命令（conda + torch cu128 + ultralytics） |
| `sample_raw/` | 从录像抽出来的 65 帧（192×192）原始图 |
| `dataset_demo/` | 自动标注产出的**演示数据集**：52 图 / 91 框（40 train + 12 val） |
| `runs/smoke/` | 3 轮冒烟训练的产物（best.pt / last.pt / 曲线图） |
| `runs/recovered/best_recovered.pt` | 从你的 `best.onnx` 回填出来的权重 |

### 本机实测记录（全部真跑过，不是纸上步骤）

环境：
```
conda env yolo26 / Python 3.12.15 / torch 2.11.0+cu128 / CUDA True
GPU: NVIDIA GeForce RTX 5070 Laptop GPU (8151 MiB) / ultralytics 8.4.171
```

管线：
```
抽帧      : 65 帧 @192x192
自动标注  : 51 张有目标 + 1 张负样本, 91 个框 (早期把 class1 误记为"机器人", 实为排球)
数据集    : dataset_demo/  images/train 40, images/val 12, labels 一一对应
冒烟训练  : 3 epochs 完成, best.pt 5.4MB, 推理 9.2ms/图 (GPU)
导出      : best.pt -> best.onnx, 结构与你原始 best.onnx 完全一致
            (input [1,3,640,640] / output [1,6,8400] / 219 initializer / 368 node)
权重回填  : best.onnx -> best_recovered.pt, unexpected=0, missing=30(全是 one2one_*)
            同一张图对比: 框完全一致, 置信度差 ~1.5%
```

> ⚠️ **这点数据只够"冒烟测试"**——验证整条链路能跑通，不可能训出可用模型。
> 单一场景 65 帧、12 张验证集，mAP50 只有 0.03，毫无意义。要真出模型，得用 MaixCam 在实际比赛光照/背景下重新拍
> **每类 300~500 张以上**。

---

## 三、图形界面（不用记命令）

> 📖 **具体怎么点、每个框填什么，见 [使用手册.md](使用手册.md)**（照着做就行）

已经给你装好两个本地网页界面，**当前会话里正开着**：

| 界面 | 地址 | 用途 |
|---|---|---|
| **YOLO26 工作台**（Gradio） | <http://127.0.0.1:7860/> | 抽帧 / 联网搜图 / 自动标注 / 训练 / 推理，全在网页里点 |
| **TensorBoard** | <http://127.0.0.1:6006/> | 训练曲线（loss、mAP、PR 曲线）实时刷新 |

**最省事的启动方式：双击 `start_workbench.bat`**（中文名 `启动工作台.bat` 是同款入口）
—— 它会开 TensorBoard 窗口、激活环境、启动工作台，并在就绪后**自动打开浏览器**。
启动器窗口是服务本体，**关掉即停服**；再双击一次会检测到已在运行，只打开浏览器。

命令行方式（等价）：

```powershell
conda activate yolo26
python app.py --browser                                              # 工作台 -> 7860
python -m tensorboard.main --logdir runs --port 6006 --host 127.0.0.1 # 曲线 -> 6006
```

> ⚠️ **bat 内容必须保持纯 ASCII（英文提示）**：cmd 用 GBK 代码页解析 bat 文件，
> 而编辑器默认存 UTF-8；中文的 UTF-8 字节被当 GBK 读会把命令行切碎，表现为
> **"双击一闪、什么都没发生"**。启动器会把每一步写进 `startup_log.txt`，便于定位。

> TensorBoard 有前置开关：ultralytics 只在设置里 `tensorboard=True` 时才写事件文件。
> 已帮你打开（`python -c "from ultralytics import settings; settings.update({'tensorboard': True})"`）。

**人工标注界面**（自动标注之后必须人工过一遍）推荐 **X-AnyLabeling**：中文界面、能加载 YOLO 模型做半自动标注、跨平台。

```powershell
pip install x-anylabeling-cvhub     # 或去 GitHub Releases 下 Windows 便携版（免安装）
```

**完全不想碰命令行**的话，还有云端 GUI 路线（图形化程度更高，但需要联网 + 账号）：

- [Ultralytics Platform 云端训练](https://docs.ultralytics.com/platform/train/cloud-training)：网页传数据集 → 选模型 → 云端 GPU 训 → 导出
- Roboflow：网页标注 + 自动标注 + 一键训练

### 自动化程度实测（重要，别抱不切实际的期待）

我实测了"自己找图 → 自己标注 → 自己训练"这条链，逐项结果：

| 环节 | 实测结果 |
|---|---|
| 联网搜图 | ✅ 可用。`fetch_images.py` 两个关键词拿到 19 张合格图（自动去重/去糊/尺寸过滤），并存 `manifest.csv` 记录来源 |
| 自动标注（用**你自己的** best.onnx） | ✅ 最靠谱。65 帧 → 51 张有目标 / 91 个框，开箱即用 |
| 自动标注（开放词汇 YOLO-World） | ⚠️ **全看提示词**。网图上用 `"toy car"` 出 20 个框(0.51~0.9)，用 `"robot"` 几乎全空；**在你的实拍帧上 8 帧 0 框** → 不能当主力 |
| 自动训练 | ✅ CPU/GPU 全自动，3 轮冒烟 40 秒 |
| 全自动端到端 | ❌ **不现实**。没有任何工具能全自动产出可用模型，缺的是**你场景的实拍数据** |

**为什么不能纯网图训练**：我抓到的第一张图是俯拍的 PCB 电路板特写，而你的模型要认的是 192×192 固定机位下的赛道小车——域差异太大。
网图只能用来**扩类/预训练**，实拍必须占数据集主导（建议 ≥80%）。

### 建议的生产流程

```
自己拍视频(多光照/多角度) ──┐
                          ├─→ extract_frames.py 抽帧
网图搜"扩类"素材 ──────────┘
                                    ↓
              build_dataset.py 用你自己的模型自动打标（最准）
                                    ↓
              X-AnyLabeling 人工快速校对（省不掉，但很快）
                                    ↓
              app.py 里点「开始训练」 → TensorBoard 看曲线
                                    ↓
              train.py --export-only 导出 ONNX → 部署
```

### 域差异对照实验（实测数据，回答"纯网图能不能凑出可用模型"）

我不用猜，直接做了对照实验：3 组训练都在同一台 5070 上跑 50 轮，
测试集统一用**真实帧**（12 张，与三组训练数据零重叠）。

| 实验 | 训练数据 | 测试域 | mAP50 | mAP50-95 | P | R |
|---|---|---|---|---|---|---|
| A | 纯网图 64 张 | 网图验证集（同域） | 0.3353 | 0.2658 | 1.000 | 0.247 |
| A | 纯网图 64 张 | **真实帧（跨域）** | **0.0017** | 0.0005 | 0.003 | 0.480 |
| B | 网图 + 实拍 103 张 | 网图域 | 0.8139 | 0.6565 | 0.628 | 0.833 |
| B | 网图 + 实拍 103 张 | **真实帧（目标域）** | **0.8361** | 0.5849 | 0.912 | 0.826 |
| C | 纯实拍 39 张 | 真实帧 | 0.7810 | 0.5004 | 0.319 | 0.840 |

**结论（三条，都很硬）**：

1. **纯网图训练 = 在你的场景里完全失效**：mAP50 只有 0.0017，比随机还差。同一帧上它一个框都画不出来
   （见 `domain_gap_compare.jpg` 左图），而混合模型能出 2 个框（0.92 / 0.69）。
2. **39 张实拍就能把真实帧 mAP50 从 0.0017 拉到 0.8361** —— 决定成败的是**数据来自哪个域**，不是数量。
3. **网图不是没用，但不能单用**：混合模型的精度 P=0.912，远高于纯实拍的 0.319（纯实拍误检多）；
   网图还能把网图域的 mAP50 从 0.335 抬到 0.814。网图的价值是**背景多样性/降低误检**，不是当主粮。

> 数字的水分也要说清：`realtest` 的标签是你原模型对真实帧的伪标注（循环依赖，绝对值偏乐观），
> 而且全部来自同一个 8 秒小视频（同场景同光照）。**所以看相对差距，不要看绝对值**——
> 但这已经足够回答"纯网图行不行"：不行。

复现命令：`python _exp_domain_gap.py`（结果落在 `domain_gap_results.json`）。

---

## 四、怎么训练（命令行版，三步）

### 第 1 步：建训练环境（本机已建好，可直接跳到第 2 步）

已存在 `yolo26` 环境，用法：

```powershell
D:\Anconda\condabin\conda.bat run -n yolo26 python train.py --help
# 或
conda activate yolo26
```

从零重建（`setup_env.ps1` 已按本机实测校准）：

```powershell
# 本机执行策略禁止直接运行 .ps1，必须显式 Bypass
powershell -ExecutionPolicy Bypass -File setup_env.ps1

# 等价的手工命令：
conda create -n yolo26 python=3.12 -y
conda activate yolo26
# RTX 5070 Laptop 是 Blackwell(sm_120)，必须从 cu128 源装，否则会装成 CPU 版
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install ultralytics onnx onnxslim onnxruntime   # 后三个是导出 ONNX 用的
yolo checks      # 看到 CUDA 可用 才算成功
```

> 本会话的沙箱会禁止 pip 清理临时文件、也禁止创建命名管道，装包与训练都需要一次性提权；
> 在你自己的终端里跑没这些问题（`train.py` 的 `--single-process` 开关也不需要）。

### 第 2 步：准备数据集（你目前缺的就是这个）

没有数据集，训练无从谈起。你已有的素材：`25-DianSai-h/vision/recordings/*.mp4`（只有 1 个 65 帧的小视频）。

```powershell
# a) 视频抽帧（自动丢糊图、去重复帧）
python extract_frames.py --src D:\videos --dst D:\raw_images --stride 5
# b) 用模型自动打标，生成数据集骨架
python build_dataset.py --src D:\raw_images --dst D:\datasets\robocup_balls --conf 0.30
# c) 用 labelImg / X-AnyLabeling 人工逐张校对 D:\datasets\robocup_balls\labels\
```

目录长这样：

```
D:\datasets\robocup_balls\
├─ images\{train,val}\*.jpg
├─ labels\{train,val}\*.txt      # class cx cy w h（归一化）; 空 txt = 负样本，要留 10% 左右
└─ data.yaml                     # build_dataset.py 自动生成
```

数量参考：**每类 ≥ 300~500 张**，train:val ≈ 8:2；正负样本都要有，否则满屏误检。

### 第 3 步：训练

```powershell
# 路线 A（推荐）：从官方预训练微调（yolo26n.pt 自动下载）
python train.py --data D:\datasets\robocup_balls\data.yaml --name yolo26n_v1

# 路线 B：从 onnx 回填的 .pt 继续（用上你原模型的 backbone/neck + 一对多头)
python onnx_to_pt.py --onnx best.onnx --out runs\recovered\best_recovered.pt
python train.py --data D:\datasets\robocup_balls\data.yaml --model runs\recovered\best_recovered.pt --lr0 0.002 --name yolo26n_v2

# 断点续训
python train.py --resume runs\yolo26n_v1\weights\last.pt

# 导出 ONNX
python train.py --export-only runs\yolo26n_v1\weights\best.pt
```

`train.py` 默认值就是给 8GB 显存 + 小数据集调的：`imgsz=640 batch=16 epochs=150
optimizer=MuSGD patience=50 close_mosaic=10 cache=ram`。
OOM 就 `--batch 8` 或加 `--imgsz 512`；Windows 上 dataloader 报错就 `--workers 0`。

**本机实测的完整冒烟命令**（3 轮，几十秒跑完，可拿来验证环境）：

```powershell
$env:MPLCONFIGDIR = "D:\18024\Documents\ARM\yolo26_robocup\.mplconfig"
python train.py --data dataset_demo\data.yaml --epochs 3 --batch 8 --workers 0 `
       --close-mosaic 0 --name smoke --project runs --single-process
```

> YOLO26 是**双头训练**（one-to-many + one-to-one 监督），推理默认走 NMS 头。
> 官方的完整配方见 [YOLO26 Training Recipe](https://docs.ultralytics.com/guides/yolo26-training-recipe/)。

---

## 五、只有 ONNX、丢了 `.pt` 怎么办（已在本机验证可行）

`onnx_to_pt.py` 利用三个实测事实把权重灌回 `yolo26n.yaml` 结构：

1. ONNX 的 initializer 名字就是 PyTorch 参数名，可按名字对号入座；
2. ONNX 里没有 `bn.*`（导出时 BN 折进 Conv，卷积带 bias），而 ultralytics 的 `Conv2d` 是
   `bias=False + BatchNorm2d` → 要把 `*.conv.bias` 搬到 `*.bn.bias`，并把该 BN 配成
   "只加偏置"的恒等变换（`weight=1, mean=0, var=1-1e-5`）；
3. 结果：**unexpected=0，missing=30 且全部是 `model.23.one2one_*`** ——
   也就是说 ONNX 里带的是**一对多（走 NMS）分支**，缺的是一对一（NMS-free）分支。
   继续微调时缺的这一半从随机初始化开始，影响有限。

验证方式（同一张图对比，已跑）：

| | box1 | box2 | conf |
|---|---|---|---|
| `best_recovered.pt` | 114.6, 31.0, 147.1, 54.0 | 88.6, 0.2, 112.9, 13.4 | 0.543 / 0.380 |
| 原始 `best.onnx` | 114.6, 31.0, 147.1, 54.0 | 88.6, 0.2, 112.9, 13.5 | 0.557 / 0.395 |

框完全一致，置信度差 ~1.5%（letterbox/decode 实现差异）。

**唯一救不回的是数据集** —— 训练仍然要你自己标数据。
如果原 `best.pt` 还在 Colab 上，**优先去捞回来**，比任何转换都划算。

---

## 六、如果要回到 MaixCam

你现在 MaixCam 上跑的是 MaixHub 在线训练的 **YOLOv5 单类「钢球」**
（`25-DianSai-h/vision/models/model_296890.mud`）。换成这个 YOLO26 2 类模型要跨三道坎：

1. 需要 ONNX → `.cvimodel`（CVITEK 转换工具 + 量化校准图）；
2. 现有 `main.py` 里 `nn.YOLOv5` 的后处理只认 v5 输出，得改成 `4+nc` 无 objectness 的解析；
3. YOLO26 的 attention 算子能否被 TPU 编译器接受，需要实测。

**如果目标只是让 MaixCam 换成 2 类检测，直接训 YOLOv5/v8 结构出 `.cvimodel` 会省事得多。**

---

## 七、常见坑（都是这次实际踩到的）

- **`data.yaml` 里 `path` 必须是绝对路径**：相对路径会被 ultralytics 解析到全局 `datasets_dir`，不是 yaml 所在目录（`build_dataset.py` 已自动写绝对路径）。
- **`--project` 传相对名会生成 `runs/detect/runs/xxx` 嵌套目录**：`train.py` 已强制转绝对路径。
- **执行策略禁止跑 `.ps1`**：报 "not digitally signed" → 用 `powershell -ExecutionPolicy Bypass -File setup_env.ps1`。
- **受限环境（沙箱）里训练会报 `PermissionError: [WinError 5]`**：ultralytics 扫描标签/缓存图片时用 `multiprocessing.ThreadPool`，它依赖命名管道。加 `--single-process` 用同步假线程池绕过（普通终端不需要）。同理需设 `MPLCONFIGDIR` 到可写目录，否则 matplotlib 报 mkdir 失败。
- **类别顺序 / 数量对不上**：`data.yaml` 必须正好 2 类，顺序与原训练一致；改类名时同步改 `infer_onnx.py` 的 `CLASS_NAMES`。
- **中文输出乱码**：脚本跑之前 `$env:PYTHONIOENCODING="utf-8"`（这个工作区本来就在处理控制台编码问题）。
- **装了 CPU 版 torch**：一定带 `--index-url https://download.pytorch.org/whl/cu128`；`yolo checks` 里 CUDA 必须是 True。
- **导出 ONNX 前要装 `onnx onnxslim onnxruntime`**：ultralytics 不默认带。
- **`cache=ram`**：数据集大（>10GB）时改 `--cache false`。

---

## 八、仓库说明（GitHub: zzzevo/YOLO-）

### 传上去了什么

| 类型 | 内容 |
|---|---|
| 代码 | `infer_onnx.py` `build_dataset.py` `extract_frames.py` `fetch_images.py` `auto_label_world.py` `merge_datasets.py` `train.py` `onnx_to_pt.py` `app.py` |
| 脚本 | `setup_env.ps1` `start_workbench.bat` `start_tensorboard.bat`（`启动工作台.bat` / `启动TensorBoard.bat` 是同名入口） |
| 文档 | `README.md` `使用手册.md` `YOLO零基础入门.md` |
| 配置 | `requirements.txt` `.gitignore` |
| 模型 | `best.onnx`（9.3 MB，本项目的原始模型） |
| 实验证据 | `domain_gap_results.json`、`domain_gap_compare.jpg` |
| 许可 | `LICENSE`（AGPL-3.0） |

### 没传上去什么（以及为什么）

| 被排除 | 体积 | 原因 |
|---|---|---|
| `weights/clip/ViT-B-32.pt` | 337 MB | CLIP 权重，用时自动下载 |
| `runs/` | 100 MB | 训练产物，能重新训练出来 |
| `raw_web2/` `dataset_*` `realtest/` `sample_raw/` | 95 MB | 数据集；且网图有版权、你的实拍涉及个人环境 |
| `yolov8s-worldv2.pt` | 25 MB | 第三方模型，自动下载 |
| `Ultralytics/`（字体） | 23 MB | 绘图字体，自动下载 |
| `yolo26n.pt` `weights/` | 5 MB | 官方预训练权重，自动下载 |

> 想连模型一起备份？`best.onnx` 已经在仓库里了。更大的权重建议用 GitHub Releases 或 Git LFS，别直接提交。

### 换一台机器怎么用

```powershell
git clone https://github.com/zzzevo/YOLO-.git
cd YOLO-
conda create -n yolo26 python=3.12 -y
conda activate yolo26
pip install -r requirements.txt
python app.py            # 工作台 http://127.0.0.1:7860/
```

⚠️ `start_workbench.bat`、`setup_env.ps1`、`onnx_to_pt.py` 里含本机路径（`D:\Anconda`、`D:\Python`），换机器要改。
⚠️ `data.yaml` 里的 `path` 是绝对路径（脚本自动生成的），换机器要重新生成数据集。

### 许可证

本仓库采用 **AGPL-3.0**。原因很直接：`ultralytics` 本身是 AGPL-3.0，你的代码 `import` 了它，
按它的许可条款，衍生作品需要同样开源。**如果将来要闭源商用，需要向 Ultralytics 购买商业授权。**

---

## 九、篮球/排球模型：用你两段视频继续训练（2026-10）

### 数据

`sample_raw/` 里两段微信视频（720×1280 竖屏，30fps，各约 100 秒）：

| 视频 | 内容 | 抽帧 |
|---|---|---|
| `微信视频..._123958_731.mp4` | **全是排球**在地板砖上滚动 | 201 张（v1_ 前缀） |
| `微信视频..._124014_013.mp4` | **全是篮球**在地板砖上滚动 | 218 张（v2_ 前缀） |

```powershell
# 抽帧（注意：这类固定机位视频不要开直方图去重，球只占很小一块，全局直方图几乎一样）
python extract_frames.py --src "sample_raw\微信视频2026-10-06_123958_731.mp4" --dst raw_ball --stride 15 --blur-thresh 0 --dedup 1.0 --prefix v1_
python extract_frames.py --src "sample_raw\微信视频2026-10-06_124014_013.mp4" --dst raw_ball --stride 15 --blur-thresh 0 --dedup 1.0 --prefix v2_
# 自动标注 -> dataset_ball（403 张有目标 / 416 框：篮球 211 + 排球 205）
python build_dataset.py --src raw_ball --dst dataset_ball --conf 0.25 --val-ratio 0.2
# 用你上传的权重继续训练
python train.py --data dataset_ball\data.yaml --model runs\recovered\best_recovered.pt --epochs 100 --lr0 0.002 --name ball_v1
```

### 两个坑（都实测踩到）

1. **糊图阈值不要照搬**：这两段视频的 Laplacian 方差中位数只有 **20.5**（最大 51），
   默认阈值 40 会把 2/3 的帧当"糊图"丢掉（实测 627 帧只留 48 张）。压缩过的手机视频本来就软，
   而这就是模型要面对的域 → `--blur-thresh 0`。
2. **直方图去重对固定机位无效**：球在动，但全局直方图几乎不变 → `--dedup 1.0` 关掉。

### 结果

| | mAP50 | mAP50-95 | P | R |
|---|---|---|---|---|
| 基线（你上传的权重） | 0.9946 | 0.9673 | 0.9608 | 0.9765 |
| 继续训练后（92 轮 / 16 分钟） | 0.9928 | 0.8888 | **0.9970** | 0.9747 |
| 新模型分类别 mAP50 | 篮球 0.9905 / 排球 0.9950 | | | |

**必须说清的"水分"**：这一版的标签是**你原来那个模型自己打的**（伪标注），
所以基线那 0.9946 是"模型和自己比"，天然接近满分。学生模型 0.9928 说明
**它成功复现了教师**（蒸馏成功），但**不等于比教师更强**。mAP50-95 反而略低（0.889 vs 0.967），
说明框的贴合度还不如教师。

### 真正的提升路径

真正能超过当前模型的地方，是**教师自己没把握的帧**。已经帮你筛出来了：

```powershell
python find_hard_frames.py --src raw_ball --out to_review --conf 0.60
# 419 帧 -> 需人工核对 46 帧（整帧无检测 16 / 低置信 24 / 同帧两类 6）
```

用 X-AnyLabeling 把 `to_review/` 里这 46 帧手工标对，再重训，模型才会真的变强。

### 关键发现：模型对"大球"失明（尺度盲区）★

用你这两段视频查出来的、对以后采集数据最有指导意义的一件事：

**同一个模型、同一张图，只要把画面整体缩小（等于让球在输入里变小），置信度就从 0.006 飙到 0.922。**

| 帧 | 缩放 1.0（原始） | 0.70 | 0.50 | 0.35 |
|---|---|---|---|---|
| `v2__001845`（球很大） | **0.006** ❌ | 0.232 | **0.701** ✅ | **0.922** ✅ |
| `v1__000360`（球大） | **0.171** ❌ | **0.868** ✅ | 0.868 ✅ | 0.830 ✅ |
| `v1__000345`（球中等，对照） | 0.521 ✅ | 0.906 ✅ | 0.904 ✅ | 0.839 ✅ |

复现脚本：`python _diag_scale.py`

**原因**：你原来的模型只在"球占画面宽度 ≤20%"的尺度上学过。球一凑近镜头（占到 60~94%），
它就完全认不出来 —— 实测那些帧的类别分只有 0.006~0.17。

**连锁危害**：用这个模型自动标注，**恰好把"大球"这一段漏掉了**（16 帧空标签），
那些帧还会被当成负样本 —— 等于教模型"大球 = 背景"。只靠自动标注重训，永远补不上。

**修法（不用人工）**：`build_dataset.py` 新增 `--scales`，做**尺度空间自动标注**——
把画面缩小后贴灰底再检测，把框按比例映射回原图，跨尺度 NMS 去重：

```powershell
python build_dataset.py --src raw_ball --dst dataset_ball_ms --conf 0.25 --scales "1.0,0.6,0.4"
# 结果: 有目标帧 403 -> 411, 框 416 -> 457 (篮球+7, 排球+34)；16 张盲区帧救回 8 张
```

救回的框尺寸印证了根因：**61%~94% 画面宽**（正常帧只有 10~20%）。

#### 修复验证（重训后，单尺度推理，同一批帧）

| 帧 | 旧模型 | 新模型 `ball_v2` | 检出 |
|---|---|---|---|
| `v1__000360` | **0.000** ❌ | **0.891** ✅ | 排球 63%宽 |
| `v1__001845` | **0.000** ❌ | **0.892** ✅ | 排球 65%宽 |
| `v1__001860` | **0.000** ❌ | **0.771** ✅ | 排球 77%宽 |
| `v2__000300` | **0.000** ❌ | **0.813** ✅ | 篮球 74%宽 |
| `v2__001845` | **0.000** ❌ | **0.802** ✅ | 篮球 78%宽 |
| `v2__001860` | **0.000** ❌ | **0.823** ✅ | 篮球 81%宽 |
| `v2__001890` | **0.000** ❌ | **0.904** ✅ | 篮球 60%宽 |
| `v2__002640` | **0.000** ❌ | **0.667** ✅ | 篮球 85%宽 |
| `v1__000345`（对照） | 0.521 | **0.886** ↑ | 排球 56%宽 |
| `v2__001095`（对照） | 0.931 | 0.935 → | 篮球 24%宽 |

**原先失明的 8 帧全部修复（8/8），正常帧没有退化。** 视觉证据见 `fix_scale_blindness.jpg`。
复现：`python _verify_fix.py`

训练曲线（`ball_v2`，85 轮，第 55 轮早停）：

| 数据集 | mAP50 | mAP50-95 | P | R |
|---|---|---|---|---|
| `ball_v1`（单尺度标注，92 轮） | 0.9928 | 0.8888 | 0.9970 | 0.9747 |
| `ball_v2`（尺度空间标注，85 轮） | **0.988** | **0.876** | 0.982 | 0.931 |

> 注意两者的验证集不同：v2 的验证集**包含那批大球难帧**，所以数字略低不代表变差，
> 真正的判据是上面那张"失明帧是否修复"的表。

> 剩下的 8 帧球又小又暗又远（`v2__002355`~`002460`），属于另一种短板，只能人工标 —— 见
> [标注修正指南.md](标注修正指南.md)。

### 产物

| 文件 | 说明 |
|---|---|
| `runs/ball_v1/weights/best.pt` / `.onnx` | 第一版模型（自动标注数据训练，92 轮） |
| `runs/ball_v2/weights/best.pt` / `.onnx` | 第二版模型（尺度空间数据训练） |
| `dataset_ball/` | 单尺度自动标注数据集（419 帧 / 416 框） |
| `dataset_ball_ms/` | **尺度空间自动标注数据集（419 帧 / 457 框）** ← 用它训练 |
| `review/` | X-AnyLabeling 复核工程（images + labels + classes.txt，已预填预测）|
| `to_review/` | 46 帧待核对清单（`需人工核对.txt`，按严重程度排序）|
| `prepare_review.py` / `apply_review.py` | 生成复核工程 / 回写并校验标注 |
| `start_labeling.bat` | 一键启动 X-AnyLabeling（独立环境 `labeling`，不污染训练环境）|
| `ball_train_report.json` | 训练数字（原始 JSON）|
| `ball_val_predictions.jpg` | 新模型在验证集上的预测拼图（人眼核验用）|
| `_diag_scale.py` | 尺度盲区的证明脚本 |
| `find_hard_frames.py` | 找出模型没把握的帧 |

---

## 十、加第三个类：定位柱（ball_v5，2026-10-11）

需求：识别**定位柱**——20cm 宽 × 100cm 高的标记板，**蓝 30 / 绿 40 / 蓝 30**（三段拼成）。

### 为什么这次不用神经网络标注

定位柱是**强饱和的蓝+绿块**，和灰色地面区分度极高。所以直接用颜色分割标注，
比神经网络更可靠（模型根本不认识新类）：

```powershell
python auto_label_pillar.py --src raw_pillar --out pillar_labels --class-id 2 --sheet _pillar_check.jpg
# 334 帧 -> 检出 303 帧（91%），抽查 12 张框全部紧贴整块板
```

判定逻辑（`auto_label_pillar.py`）：
1. 分别取蓝、绿的 HSV 掩码 → 合并 + 膨胀让蓝绿段连成一整块
2. 每个大连通块必须**同时含有足够的蓝和足够的绿**（这就是"定位柱"的签名，能挡掉纯色物体）
3. 长条形状约束（长短边比 1.3~14）
4. 取外接矩形 → 就是整块板的框

### 数据集与训练

```powershell
python add_labeled_frames.py --images raw_pillar --labels pillar_labels `
    --base dataset_ball_v4 --dst dataset_ball_v5 --val-ratio 0.2 `
    --names "篮球,排球,定位柱" --prefix pl_
# dataset_ball_v5: train 754 / val 143（球 511 + 定位柱 243）

python train.py --data dataset_ball_v5\data.yaml --model runs\ball_v4\weights\best.pt `
    --epochs 80 --lr0 0.001 --name ball_v5 --single-process
```

> **关键点：类别数从 2 变 3**。日志里出现 `Overriding model.yaml nc=2 with nc=3` 和
> **`Transferred 708/708 items from pretrained weights`** —— ultralytics 把旧模型的权重
> **全部迁移**了过来（检测头的类分支按重叠通道对齐），所以球类的能力没有被重置。

### 结果（分开两套验证集，才说得清各自表现）

| 验证集 | 模型 | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| 球（83 帧） | ball_v4（2 类） | 0.985 | 0.9778 | 0.990 | 0.8793 |
| 球（83 帧） | **ball_v5（3 类）** | **0.9979** | 0.9667 | 0.9906 | **0.8861** |
| **定位柱（60 帧）** | **ball_v5** | **0.9989** | **1.000** | **0.995** | — |

**结论**：加第三类**没有让球类退化**（mAP50-95 反而略升），定位柱 mAP50 0.995、召回 100%。
视觉证据 `pillar_predictions.jpg`（12/12 帧框准，远近/旋转都对）。

> 口径说明：定位柱的验证标签来自颜色标注器本身（不是人工），所以这几个数字衡量的是
> "模型与标注器的一致度"；标注器本身已抽查 12/12 准确。要更硬的结论，人工修一批再测。

### 新增工具

| 文件 | 用途 |
|---|---|
| `auto_label_pillar.py` | 颜色+形状自动标注定位柱（蓝绿签名 + 长条约束）|
| `add_labeled_frames.py` | 把**已标好**的图片+标签导入数据集（支持划分验证集、改类名）|
| `data_eval_ball.yaml` / `data_eval_pillar.yaml` | 分开评估用的两套 yaml（val 指向图片清单）|
| `pillar_predictions.jpg` | 定位柱预测肉眼验证图 |

---

## 十一、再加一段排球视频（ball_v6，2026-10-11）

新视频：544×960 / 30fps / 147 秒 → 抽 295 帧，**v5 检出 294 帧**（模型对新场地泛化得不错）。

### 一个关键判断：不是只留排球

类别统计是 `排球 296 / 篮球 35 / 定位柱 3`。乍看像误检，**看帧才发现这段视频里地板上同时有
排球、橙色篮球、绿色球**（多球同框的新场地）。所以标注时**保留模型的所有类别**，而不是只留排球
——这等于顺手给另外两类也补了新场地的样本：

```powershell
python add_web_frames.py --src raw_volley --base dataset_ball_v5 --dst dataset_ball_v6 `
    --model runs/ball_v5/weights/best.onnx --scales 1.0 --conf 0.45 `
    --class-ids 0,1,2 --max-boxes 5 --prefix vb_ --val-ratio 0.2 --names "篮球,排球,定位柱"
# dataset_ball_v6: train 989 / val 202（球 83 + 柱 60 + 新排球 59）
```

### 结果（三个验证集，v5 vs v6）

| 验证集 | 模型 | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| 球（83 帧，旧场景） | v5 | 0.9979 | 0.9667 | 0.9906 | 0.8861 |
| 球（83 帧，旧场景） | **v6** | 0.9876 | 0.9602 | **0.9910** | 0.8851 |
| 定位柱（60 帧） | v5 | 0.9989 | 1.0000 | 0.9950 | 0.9111 |
| 定位柱（60 帧） | **v6** | 0.9990 | 1.0000 | 0.9950 | **0.9427** ↑ |
| **新排球（59 帧，新场地）** | v5 | 0.4924 | 1.0000 | 0.5135 | 0.5066 |
| **新排球（59 帧，新场地）** | **v6** | **0.7316** | 0.9790 | **0.7462** | **0.6451** ↑ |

视觉证据 `volley_v6_predictions.jpg`（12/12 帧框准，0.91~0.95）、`volley_label_check.jpg`（标签抽查 12/12）。

> ⚠️ **口径必须说清**：新排球验证集的标签来自 **v5 自己的检测**（conf≥0.45，且多球场景下必然漏标一部分），
> 而 v6 又在这批帧的**训练部分**上训过。所以 v6 在这套数字上占优是**结构性**的，
> 不能当成"绝对进步"。真正靠得住的结论有两条：
> 1. **旧场景没有退化**（球 0.9906→0.9910，mAP50-95 基本持平）；
> 2. **定位柱框更准了**（mAP50-95 0.9111→0.9427），说明新数据对整体有益。
>
> 要拿到"新场地绝对精度"，得给这批帧做**人工标注**（`prepare_review.py` 那条流程）。

### 本轮新增

| 文件 | 说明 |
|---|---|
| `runs/ball_v6/weights/best.pt` / `.onnx` | 最新模型（3 类，含新排球场地；ONNX 仍是 `[1,7,8400]`）|
| `volley_v6_predictions.jpg` | 新场地预测肉眼验证 |
| `volley_label_check.jpg` | 新帧标签抽查 |
| `data_eval_volley.yaml` | 新排球验证集（val 指向 `dataset_ball_v6/val_volley.txt`）|
| `add_web_frames.py` 新参数 | `--prefix`（文件名前缀）、`--val-ratio`（追加帧也划验证集）、`--names`（重写类名）|

---

## 十二、3 段新视频 + 网图爬取的结论（ball_v7，2026-10-12）

### 3 段视频（有效，已进数据集）

| 视频 | 规格 | 抽帧 | 内容 |
|---|---|---|---|
| 视频1 | 720×1280 / 95 秒 | 115 | 篮球 + 绿球 + 排球（多球同框）|
| 视频2 | 1280×720 / 109 秒 | 131 | 排球（横屏，新视角）|
| 视频3 | 1280×720 / 115 秒 | 138 | 篮球（横屏，新视角）|

用 v6 自动标注（保留全部类别，conf≥0.45）→ `dataset_ball_v7: train 1294 / val 278`。
抽帧标签抽查 12/12 准（`v7_label_check.jpg`）。

### 网图爬取：**自动标注的正解是"模型检测 + CLIP 语义复核"**（2026-10-12 突破）

爬取本身从来不是问题（`fetch_images_bing.py` 直抓 Bing，**单轮 368 张**）。瓶颈在**标注**：
模型会把橙子、玫瑰、彩虹、地图、人脸、商标都当成球，而**形状/颜色启发式分不清"橙子"和"橙色篮球"**
（三轮实测都失败，见下表）。**CLIP 是语义模型，它知道这个区别** —— 于是：

```powershell
python clip_filter.py --src raw_web_big --conf 0.45        # 先看判别效果（出 clip_check.jpg）
# 实测 139 个候选：CLIP 判"是球">=0.9 的 30 个**全是真球**；
#   被否掉的，CLIP 给的理由正是我肉眼看到的："an orange fruit"/"a flower"/"a plant or grass"
python add_web_frames.py --src raw_web_big --dst dataset_ball_v8 `
    --model runs/ball_v6/weights/best.onnx --scales 1.0,0.6 --conf 0.5 --class-ids 0,1 `
    --prefix web_ --clip-filter --clip-thr 0.9
# 368 张 -> 保留 72 张（无检测 105 / 置信度低 122 / **CLIP 否定 91**）
```

| 尝试 | 做法 | 结果 |
|---|---|---|
| 1 | 模型直接检测（conf≥0.45）| 12 张抽查 3~4 张垃圾：**地图标记、人脸**被框 |
| 2 | +"单一主色 + 圆形填充率" | 仍有 **黄玫瑰、`go green` 草地、ZOL 水印、`blue` 文字**被框 |
| 3 | +圆度≥0.60 + 凸性≥0.88 + 长宽比 | 12 张里只剩 2 张对：**"orange"商标、浏览器截图、橙色漩涡、真的橙子、品红圆点**全中招 |
| **4** | **+CLIP 语义复核（P(球)≥0.9）** | **12/12 全是真球**（`clip_added_check.jpg`、`clip_check.jpg`）|

**结论**：想用网图，必须过 CLIP 这道闸；不过闸的标签会教模型"橙子=篮球"，比没数据更糟。
本轮 72 张干净网图（篮球 47 框 / 排球 40 框）已进 `dataset_ball_v8`。

---

## 十三、ball_v8：第 4 段视频 + 173 张 CLIP 复核网图

### 数据

| 来源 | 数量 | 说明 |
|---|---|---|
| 视频4（篮球，544×960/156 秒）| 188 帧抽 → 149 进 train / 37 进 val | 橙色篮球，`s4_` 前缀 |
| 网图第 1 批（32 个关键词）| 368 张爬 → **72 张**过 CLIP | `web_*` |
| 网图第 2 批（32 个新关键词）| 362 张爬 → **101 张**过 CLIP（阈值放宽到 0.85）| `web_*` |

合计 **爬 730 + 295 + 94 张，最终 173 张干净网图进训练集**。
`dataset_ball_v8: train 1585 / val 315`（3 类）。

训练：从 `ball_v7` 继续 80 轮（`Transferred 708/708 items`，标签 0 corrupt）。

### 结果（四个验证集，v6 → v7 → v8 → v9）

| 验证集 | v6 | v7 | v8 | v9 |
|---|---|---|---|---|
| 球（83 帧，旧场景，人工修正标签）mAP50 / mAP50-95 | **0.9910 / 0.8851** | 0.9914 / 0.8829 | 0.9882 / 0.8807 | 0.9851 / 0.8657 |
| 定位柱（60 帧）mAP50 / mAP50-95 | 0.9950 / 0.9427 | 0.9950 / 0.9647 | 0.9950 / 0.9713 | 0.9950 / **0.9771** |
| 新排球（59 帧，伪标签）mAP50 | 0.7462 | 0.5268 | 0.5597 | 0.5804 |
| 新视频4（37 帧）mAP50 | 0.9944 | 0.9939 | 0.9941 | 0.9934 |

> **两个方向相反的效应，必须同时说**：
> 1. **定位柱持续变好**（mAP50-95 0.9427 → 0.9771），新数据对这个类明确有益；
> 2. **旧场景球类轻微下滑**（0.8851 → 0.8657，约 -2%），这是加大量新场景数据的典型代价（数据分布稀释）。
>
> 差距只有 1~2 个百分点、在训练噪声量级内；但如果你**只关心老场地**，`ball_v6/v7` 在那套验证集上
> 数字更好；如果关心**新场地 + 定位柱 + 覆盖面**，用 `ball_v9`。
>
> "新排球"那一行所有版本都别当基准：标签来自 v5 的检测（不完整），v7+ 把 v5 漏标的球也检出来了，
> 按不完整标签算反而是"误检"——**伪标签上限**的典型表现。

### 爬图总量

| 轮次 | 关键词数 | 爬到 | 过 CLIP | 去向 |
|---|---|---|---|---|
| 1 | 24 | 295 | （早期，未过 CLIP，已撤）| — |
| 2 | 10（购物向）| 94 | 19（人工式严格过滤）| 已撤 |
| 3 | 32 | 368 | 72 | v8 |
| 4 | 32（新词）| 362 | 101 | v8 |
| 5 | 30 | 229 | 45 | v9 |
| **合计** | | **1348** | **218 张干净图** | |

### 本轮新增工具

| 文件 | 用途 |
|---|---|
| `clip_filter.py` | **用 CLIP 对检测框做语义复核**（解决网图标注污染，本项目的关键突破）|
| `add_web_frames.py --clip-filter --clip-thr` | 把 CLIP 复核接进标注流水线 |
| `prepare_web_review.py` | 网图的 X-AnyLabeling 复核工程（CLIP 之后仍存疑时的兜底）|
| `data_eval_v8new.yaml` | 视频4 的验证集 |

### 历史清理（本轮完成）

`.git` 从 **120 MB → 11 MB**，远端文件数 **~2500 → 65**：

```bash
git filter-branch -f --index-filter "git rm -r --cached --ignore-unmatch \
    dataset_ball_v4 raw_web_bb hue_labels web_plain_labels hsv_labels raw_bb2 raw_pillar raw_volley review" \
    --prune-empty -- --all
rm -rf .git/refs/original && git reflog expire --expire=now --all && git gc --prune=now
git push --force origin main
```

> 注意：GitHub 的 API 仍会显示 93 MB 一段时间——强推后旧对象变成"不可达"，
> 要等 GitHub 自己 gc 才会真正释放。**远端内容已经是干净的**（`api.github.com/.../git/trees/main`
> 只有 65 个文件，数据目录 0 个、最大文件是故意保留的 `best.onnx` 9.5MB）。

---

## 十四、让定位柱在远处也能认出来（ball_v10，2026-10-13）

需求：**远距离也要识别定位柱**。定位柱是 20×100cm 的**细长条**（长宽比 5:1），
远处只有几十像素宽——典型的"小目标"问题。

### 先量化短板：按目标像素高度分档看召回（`eval_by_size.py`）

视频5（长走廊，720×1280）抽 177 帧，柱高 126~623px：

| 目标高度 | 样本 | v9 整图 640 |
|---|---|---|
| 64~128px | 1 | 100% |
| **128~256px** | **107** | **77% ← 短板** |
| 256~512px | 51 | 100% |
| >512px | 15 | 100% |
| **合计** | 174 | **86%** |

**原因**：720×1280 的图缩到 640 输入时纵向压缩一半，柱高 200px 只剩 ~100px、宽只剩 ~20px，
细长物体在这个尺寸下特征几乎消失。

### 三条解法，实测对比

**必须用 held-out 帧才算数**（我一开始在全 177 帧上测，那里面 80% 是训练帧，得出的 100% 是假的）。
下表用的是 **v10 从未训练过的 34 帧**（柱高 148~578px）：

| 方式 | 远距离召回 | 平均置信度 | 耗时 |
|---|---|---|---|
| v9 @640 整图（原状）| 85% | 0.81 | 77 ms/帧 |
| v9 @960 整图 | 100% | 0.82 | 154 ms/帧 |
| **v10 @640 整图** | **100%** | **0.92** | 77 ms/帧 |
| v10 @960 整图 | 100% | 0.92 | 154 ms/帧 |
| v10 @640 分块放大 | 100% | 0.92 | 268 ms/帧 |

**结论：补远距离数据重训（v10）之后，普通 640 整图就 100% 了** —— 不用加分辨率、不用分块放大，
置信度还从 0.81 提到 0.92。960 导出和分块放大留作"算力富裕时再压榨"的备选。

视觉证据 `pillar_far_v9_vs_v10.jpg`：最远的 6 帧，v9 要么漏检要么只有 0.27~0.70，v10 全部 0.89~0.93。

### 补数据重训（ball_v10）

远距离帧 177 张（颜色标注器检出 98%，抽查 12/12 框准）→ 并进数据集：

| 数据集 | train | val | 定位柱框数 |
|---|---|---|---|
| dataset_ball_v9 | 1630 | 315 | 363 |
| **dataset_ball_v10** | **1770** | **349** | **478** |

训练：从 `ball_v9` 继续 80 轮。

### 五套验证集总表（v9 → v10）

| 验证集 | v9 | v10 |
|---|---|---|
| **远距离定位柱（34 帧，held-out）** | 0.9918 / 0.9412 / 0.9822 / **0.8484** | 0.9984 / **1.0000** / 0.9950 / **0.9585** ↑ |
| 定位柱近距离（60 帧） | 0.9989 / 1.0000 / 0.9950 / 0.9771 | 0.9990 / 1.0000 / 0.9950 / **0.9812** |
| 球（83 帧） | 0.9512 / 0.9760 / 0.9851 / 0.8657 | 0.9410 / **1.0000** / 0.9830 / 0.8658 |
| 新视频4（37 帧） | 0.9551 / 0.9868 / 0.9934 / 0.9293 | 0.9702 / 0.9868 / 0.9934 / **0.9306** |
| 新排球（59 帧，伪标签） | 0.9995 / 0.5000 / 0.5804 / 0.5398 | 0.4900 / 1.0000 / **0.6219** / 0.5687 |
| （P / R / mAP50 / mAP50-95） | | |

**没有退化，远距离定位柱大幅改善**（mAP50-95 0.8484 → 0.9585，召回 0.94 → 1.00）。

### 部署建议

| 场合 | 方案 | 远距离召回 | 代价 |
|---|---|---|---|
| **默认（推荐）** | **v10 整图 640** | **100%** | 最低 |
| 算力富裕 | v10 整图 960 | 100%（置信度相同）| 2× |
| 离线可慢算 | v10 分块 2×2 放大 | 100% | 3.5× |

```powershell
# 960 输入版本（会覆盖 best.onnx，先备份再导出）
python train.py --export-only runs\ball_v10\weights\best.pt --imgsz 960
# 按目标尺寸分档评估召回（一眼看出"多远开始失效"）
python eval_by_size.py --src raw_pillar_far --model runs\ball_v10\weights\best.onnx `
    --model960 runs\ball_v10\weights\best_960.onnx --cls 定位柱 --compare
```

> **踩坑记录（值得记）**：我写的公平评估脚本里 IoU 分母把宽度写成了 `a[2]-a[1]`（本该 `a[2]-a[0]`），
> 结果把几乎完全重合的框算成 IoU 0.3、还出现 IoU>1 的荒谬值，一度得出"v10 只有 71%"的错误结论。
> **教训：出现 >1 的 IoU 就是实现 bug 的铁证；评估工具本身也要先用"明显应该命中"的样本自检。**

### 工具

| 文件 | 用途 |
|---|---|
| `eval_by_size.py` | **按目标像素尺寸分档评估召回**，可对比整图/高分辨率/分块三种推理 |
| `infer_tiled.py` | 分块放大推理（小目标用）|
| `_compare_far.py` / `_fair_far_eval.py` | 三种推理方式对比 / held-out 公平对比 |
| `auto_label_pillar.py` | 颜色+形状自动标注定位柱（远处也能标，98% 覆盖）|

---

## 十五、数据集标签体检：查出并修掉 89 个错标（ball_v11，2026-10-13）

**起因**：用户反馈"排球老被识别成篮球"，怀疑数据集标错了。

### 查证过程（三步，都可复现）

**① 抽样看每一类的框**（`check_dataset_labels.py`）——把每个类别的标注框裁出来放大拼图：

- 排球类 24/24 全是真排球 ✓
- **篮球类 24 张里有 3 张其实是排球**（沙滩排球场景、蓝黄白排球、Mikasa 排球）→ 约 12% 错标

**② CLIP 全量复核**（`audit_labels_clip.py`）——逐个框问 CLIP"这是篮球还是排球"：

```
体检了 1805 个（篮球/排球）框
  可疑（CLIP 更认为是另一类）: 89 个 = 4.9%
     标签是 篮球，CLIP 认为是 排球: 50 个
     标签是 排球，CLIP 认为是 篮球: 39 个
```

可疑样本拼图（`audit_clip.jpg`）肉眼复核：**基本都是真错标**——沙滩排球场景标成篮球、
橙色"JER"篮球标成排球、儿童打篮球标成排球……

**③ 修因**：这些错标来自**我自己的伪标签流水线**——模型把某类认错 → 标签按模型的判定写进数据集 →
重训后模型更坚信这个错误（自我强化）。典型受害者就是那颗橙色"JER"篮球：
它被标成排球后，模型见到它就喊排球。

### 修正与结果

```powershell
# 复制成 v11，用 CLIP 的判定把确信的错标互换（会先备份到 labels_bak）
Copy-Item -Recurse dataset_ball_v10 dataset_ball_v11
python audit_labels_clip.py --data dataset_ball_v11 --fix --backup
# 复核：可疑 89 -> 0
# 注意！改完标签必须删掉 ultralytics 的标签缓存，否则用的还是旧缓存
Get-ChildItem dataset_ball_v11 -Recurse -Filter *.cache | Remove-Item -Force
```

**类间混淆专项检查**（`confusion_check.py`，自算"真值→预测"）：

| 模型 | 篮球→篮球 | 排球→排球 | **类间判错** | P | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| v10（旧标签训练）| 43 | 40 | **5**（都是"篮球→排球"）| 0.9749 | 0.9368 | 0.8267 |
| **v11（修正标签训练）** | **48** | **40** | **0** ✓ | **0.9979** | **0.9450** | **0.8342** |

五套验证集：球类 P 0.9748→**0.9973**、新视频 R 0.9630→**0.9970** / mAP50-95 0.9348→**0.9578**、
定位柱近/远 0.9812→**0.9830** / 0.9585→**0.9721**，**全面不劣**。

### 顺手确认：类别顺序没错

用同时含两类球的真实帧复核（`class_order_recheck.jpg`）：
蓝黄 Mikasa → **排球 0.95**，绿色球 → **篮球 0.91**。**0=篮球 / 1=排球 是对的**。

> 所以如果部署端还看到"排球→篮球"，那**不是数据集问题**，去查部署代码：
> 类别索引映射、是否还在按旧的 2 类（`[1,6,8400]`）解析输出、颜色/名字表是否写反。

### 新增体检工具

| 文件 | 用途 |
|---|---|
| `check_dataset_labels.py` | 按类别抽样出标注框拼图（人眼快速查错标）|
| `audit_labels_clip.py` | **CLIP 全量复核标签**，可 `--fix` 直接改错标（自动备份）|
| `confusion_check.py` | **类间混淆专项**：真值→预测计数 + 判错帧出图 |
| `class_order_recheck.jpg` / `audit_clip.jpg` / `confusion_v11_ball_B.jpg` | 本轮证据图 |


---

## 十六、手机实时测试 + 爬图工具升级（2026-10-13）

### 手机摄像头实时看效果

手机浏览器**只允许 HTTPS 页面开摄像头**（http:// 下摄像头全黑），所以：

```powershell
# 一条命令：局域网监听 + HTTPS（首次自动生成自签证书到 certs/）
python app.py --lan              # 等价于 --host 0.0.0.0 --ssl
```

启动后打印 `https://<你的局域网IP>:7860/`，手机连**同一个 Wi-Fi**打开，
证书警告点「高级→继续访问」，然后在「④ 推理验证 / 手机实时测试」页点摄像头框即可。

- **推理在电脑上跑**，手机只当摄像头+显示器 → 适合快速验收
- `.pt` 走 **PyTorch/GPU**（约 5~10ms/帧，推荐实时用）；`.onnx` 走 CPU（30~60ms/帧）
- 页面有"实时刷新间隔"滑条；详见 [实时识别.md](实时识别.md)

> 自签证书在 `certs/`（已 gitignore，**私钥绝不能入库**）。
> 冒烟验证：`https://127.0.0.1:7868/` 与 `https://10.21.153.134:7868/` 均返回 200、
> 页面含摄像头组件、TLS 1.3 握手成功。

### 训练页说明也写进界面了

「③ 训练」页顶部现在直接写明：**这一页默认就是"在已有模型基础上继续训练"，不是从零开始**，
并给出"继续训练（lr0=0.001）vs 从零开始（yolo26n.pt + lr0=0.01）"的区别。

### 爬图工具升级：留用率 16% → 39%

原来的问题：Bing 普通图片搜索会把**海报、logo、文字图、地图、卡通**都算相关，
而且旧流程是"先全下下来、事后再筛"，所以 `raw_web_*` 里一堆垃圾。

`fetch_train_images.py` 的改进：
1. `qft=+filterui:photo-photo` —— 搜索层就只要**照片**，挡掉 clipart/logo/插画/地图
2. **边下边验**：YOLO 检出球 → 裁框给 CLIP 语义复核 → 两个都过才落盘（直接产出图片+标签）
3. 内容级去重（平均哈希）、**按关键词统计产出率**，方便删掉没用的词

实测（4 个关键词、129 个候选）：

| 关键词 | 候选 | 留用 | 产出率 |
|---|---|---|---|
| 篮球 商品图 | 35 | 24 | **69%** |
| volleyball product photo | 34 | 14 | 41% |
| basketball isolated white background | 35 | 12 | 34% |
| 排球 商品图 | 25 | 0 | **0%（该删）** |
| **合计** | **129** | **50** | **39%**（旧流程 ~16%）|

```powershell
python fetch_train_images.py --dst raw_good --per-query 40 --clip-thr 0.9 --conf 0.5 `
    --queries "篮球 商品图" "basketball isolated white background" "volleyball product photo"
# 产出 raw_good/images + raw_good/labels，可直接并入数据集：
python add_labeled_frames.py --images raw_good/images --labels raw_good/labels `
    --base dataset_ball_v11 --dst dataset_ball_v12 --val-ratio 0.15 `
    --names "篮球,排球,定位柱" --prefix web_
```


---

## 十七、实时识别页（框直接画在实时画面上）

需求：**不要另一个窗口显示结果，要框直接压在手机实时画面上、并跟着球走。**

Gradio 的摄像头组件做不到（它的预览是浏览器本地的，服务端画的框只能显示在另一个组件里），
所以单独写了一个轻量服务 live_overlay.py + live_overlay.html：

`
手机浏览器                          你的电脑
┌──────────────┐   JPEG(640宽)    ┌──────────────────┐
│ 摄像头 → canvas │ ──── POST ────→ │ YOLO 推理(CPU/GPU)│
│ 画面直接显示    │ ←── JSON 框 ─── │                  │
│ 框画在同一画面上│                  └──────────────────┘
└──────────────┘
`

- **HTTP POST 传帧**（不用 WebSocket）：语义一样，但不依赖 websockets 库，也没有断线重连问题；
  前端是上一帧回来才发下一帧，天然自我保护不积压
- **框平滑跟随**：指数平滑逼近目标（SMOOTH=0.45）+ 短暂保留渐隐（COAST_MS=600）+ IoU 匹配轨迹
- **帧率自适应**：往返快多发（最多 25fps），慢自动降到 4fps
- 实测：**服务器 44~67 ms/帧，往返 62~97 ms**（720×1280 帧，ONNX/CPU）→ 约 12~15fps 检测更新，
  画面 60fps 渲染，观感是连续跟踪

`powershell
实时识别.bat                                   # 默认 ONNX/CPU，端口 7871，HTTPS
实时识别.bat 7871 runs\ball_v11\weights\best.pt   # 换 GPU 更快
`

> 因为手机浏览器只允许 HTTPS 页面开摄像头，服务默认自带 HTTPS（复用 certs/），
> 并在电脑端页面显示二维码供手机扫码。详见 [实时识别.md](实时识别.md)。
> **镜像功能已按需求全部移除。**
