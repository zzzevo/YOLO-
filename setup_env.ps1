# 建立 YOLO26 训练环境（Windows / RTX 5070 Laptop 8GB）
# 用法: powershell -ExecutionPolicy Bypass -File setup_env.ps1
#   注意: 本机执行策略禁止直接运行未签名脚本，所以必须显式带 -ExecutionPolicy Bypass，
#   否则报 "not digitally signed"。也可以直接把里面的命令逐条粘进终端。
# 说明: RTX 50 系是 Blackwell(sm_120)，torch 必须从 cu128 源装，否则会装成 CPU 版。

$ErrorActionPreference = "Stop"
$envName = "yolo26"
$conda   = "D:\Anconda\condabin\conda.bat"

Write-Host "==> 1/4 创建 conda 环境 ($envName, python 3.12)" -ForegroundColor Cyan
& $conda create -n $envName python=3.12 -y
if ($LASTEXITCODE -ne 0) { throw "conda create 失败" }

Write-Host "==> 2/4 安装 PyTorch (CUDA 12.8)" -ForegroundColor Cyan
& $conda run -n $envName python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw "torch 安装失败" }

Write-Host "==> 3/4 安装 ultralytics" -ForegroundColor Cyan
& $conda run -n $envName python -m pip install ultralytics
if ($LASTEXITCODE -ne 0) { throw "ultralytics 安装失败" }

Write-Host "==> 4/4 自检" -ForegroundColor Cyan
& $conda run -n $envName python -c "import torch, ultralytics; print('torch', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'); print('ultralytics', ultralytics.__version__)"

Write-Host ""
Write-Host "完成。之后每次用:" -ForegroundColor Green
Write-Host "  conda activate $envName"
Write-Host "  python train.py --data D:\datasets\robocup_balls\data.yaml --name yolo26n_v1"
