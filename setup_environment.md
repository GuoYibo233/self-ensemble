# Self-Ensemble 调试环境配置指南

## 快速启动（推荐）

### 1. 直接运行脚本
创建 `start_debug.bat` 文件，双击即可启动：

```batch
@echo off
echo === Self-Ensemble 调试环境 ===
echo.

REM 设置 Anaconda 路径
set ANACONDA_PATH=C:\ProgramData\anaconda3
set PATH=%ANACONDA_PATH%;%ANACONDA_PATH%\Scripts;%ANACONDA_PATH%\Library\bin;%PATH%

REM 检查环境是否存在
if not exist "%USERPROFILE%\.conda\envs\self-ensemble-debug" (
    echo 环境不存在，正在创建...
    conda env create -f environment_simple.yml
)

echo 激活调试环境并运行...
"%USERPROFILE%\.conda\envs\self-ensemble-debug\python.exe" debug_generate.py --method per_prompt

pause
```

### 2. PowerShell 一键脚本
创建 `start_debug.ps1` 文件：

```powershell
# Self-Ensemble 调试环境启动脚本
Write-Host "=== Self-Ensemble 调试环境 ===" -ForegroundColor Green

# 设置 Anaconda 路径
$env:PATH = "C:\ProgramData\anaconda3;C:\ProgramData\anaconda3\Scripts;C:\ProgramData\anaconda3\Library\bin;" + $env:PATH

# 检查环境
$envPath = "$env:USERPROFILE\.conda\envs\self-ensemble-debug"
if (!(Test-Path $envPath)) {
    Write-Host "环境不存在，正在创建..." -ForegroundColor Yellow
    conda env create -f environment_simple.yml
}

# 运行调试脚本
Write-Host "启动调试脚本..." -ForegroundColor Cyan
& "$envPath\python.exe" debug_generate.py --method per_prompt
```

## 手动配置步骤

### 第一次设置

1. **配置 PATH 环境变量**
```powershell
$env:PATH = "C:\ProgramData\anaconda3;C:\ProgramData\anaconda3\Scripts;C:\ProgramData\anaconda3\Library\bin;" + $env:PATH
```

2. **创建 conda 环境**
```powershell
conda env create -f environment_fixed.yml
```

3. **验证环境**
```powershell
& "C:\Users\Kryos Seira\.conda\envs\self-ensemble-debug\python.exe" test_environment.py
```

### 日常使用

**选项 1：直接使用环境中的 Python**
```powershell
& "C:\Users\Kryos Seira\.conda\envs\self-ensemble-debug\python.exe" debug_generate.py --method avg
```

**选项 2：设置 PATH 后使用 conda**
```powershell
# 设置路径
$env:PATH = "C:\ProgramData\anaconda3;C:\ProgramData\anaconda3\Scripts;C:\ProgramData\anaconda3\Library\bin;" + $env:PATH

# 激活环境（如果需要）
conda activate self-ensemble-debug

# 运行脚本
python debug_generate.py --method weighted_avg
```

## VSCode 调试配置

VSCode 的调试配置已经创建在 `.vscode/launch.json`：

1. 打开 VSCode
2. 按 `F5` 或点击 Run → Start Debugging
3. 选择你想要的调试配置：
   - `Debug - per_prompt`
   - `Debug - avg`
   - `Debug - weighted_avg`

## 可用的调试命令

### 不同的集成方法
```bash
# 逐提示生成
python debug_generate.py --method per_prompt

# 平均集成
python debug_generate.py --method avg

# 加权平均
python debug_generate.py --method weighted_avg

# 最大值集成
python debug_generate.py --method max

# 加权最大值
python debug_generate.py --method weighted_max
```

### 设备选择
```bash
# 自动选择设备
python debug_generate.py --method avg --device auto

# 强制使用 CPU
python debug_generate.py --method avg --device cpu

# 强制使用 GPU（如果可用）
python debug_generate.py --method avg --device cuda
```

## 环境管理

### 查看环境信息
```powershell
& "C:\Users\Kryos Seira\.conda\envs\self-ensemble-debug\python.exe" test_environment.py
```

### 重新创建环境
```powershell
# 删除旧环境
conda env remove -n self-ensemble-debug

# 重新创建
conda env create -f environment_simple.yml
```

### GPU 性能测试
```powershell
& "C:\Users\Kryos Seira\.conda\envs\self-ensemble-debug\python.exe" gpu_benchmark.py
```

## 故障排除

### 问题 1：conda 命令找不到
**解决方案：**
```powershell
$env:PATH = "C:\ProgramData\anaconda3;C:\ProgramData\anaconda3\Scripts;C:\ProgramData\anaconda3\Library\bin;" + $env:PATH
```

### 问题 2：环境激活失败
**解决方案：** 直接使用环境中的 Python
```powershell
& "C:\Users\Kryos Seira\.conda\envs\self-ensemble-debug\python.exe" your_script.py
```

### 问题 3：依赖缺失
**解决方案：** 重新创建环境
```powershell
conda env remove -n self-ensemble-debug
conda env create -f environment_simple.yml
```

## 文件说明

- `environment_fixed.yml`: 修复的依赖配置（解决NumPy/Pandas兼容性）
- `environment_simple.yml`: 简化的依赖配置（已废弃）
- `debug_generate.py`: 主要调试脚本
- `test_environment.py`: 环境验证脚本
- `gpu_benchmark.py`: GPU 性能测试
- `.vscode/launch.json`: VSCode 调试配置

## 快速参考

| 文件                     | 用途                         |
| ------------------------ | ---------------------------- |
| `debug_generate.py`      | 主调试脚本，查看每步执行过程 |
| `test_environment.py`    | 验证环境是否正确配置         |
| `gpu_benchmark.py`       | 测试 GPU vs CPU 性能         |
| `environment_simple.yml` | Conda 环境配置               |

现在你可以轻松地"在本地运行少量，来看每一步都干了什么"了！🎯