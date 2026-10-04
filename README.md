# YOLO26n 训练 / 部署工作台（robocup_balls 模型）

针对你给的 `best.onnx`。**核心结论：ONNX 不能直接训练**，它只有冻结的权重和前向图。
训练必须要有 **①数据集** + **②`.pt` 权重（或官方预训练权重）**。

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

实测推理（CPU，onnxruntime 1.30）：单帧 **~28 ms**，在你的
`25-DianSai-h/vision/recordings/test_01_20260804_115304.mp4` 上能出框（class1, conf 0.26~0.52）。

⚠️ **类别顺序未知**：模型是 2 类，但 `data.yaml` 里的 `names` 没导出到 ONNX 里。
已确认 **class1 = 机器人**；class0 名称未知（现有录像 91 个框全是 class1）。
改了类名要同步改 `infer_onnx.py` 顶部的 `CLASS_NAMES`，**顺序必须和原训练时一致**，否则 6 通道解读就错位了。

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
自动标注  : 51 张有目标 + 1 张负样本, 91 个框, 全部 class1(机器人)
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
