# 快速环境配置脚本 - 仅需运行一次
Write-Host "===== Self-Ensemble 环境一次性配置 =====" -ForegroundColor Green
Write-Host ""

# 设置 Anaconda 路径
$anacondaPath = "C:\ProgramData\anaconda3"
$env:PATH = "$anacondaPath;$anacondaPath\Scripts;$anacondaPath\Library\bin;" + $env:PATH

Write-Host "[1/3] 检查 Anaconda 安装..." -ForegroundColor Cyan
if (!(Test-Path "$anacondaPath\Scripts\conda.exe")) {
    Write-Host "[ERROR] 未找到 Anaconda，请检查安装路径" -ForegroundColor Red
    Read-Host "按Enter键退出"
    exit 1
}
Write-Host "[SUCCESS] Anaconda 检查通过" -ForegroundColor Green

Write-Host "[2/3] 创建调试环境..." -ForegroundColor Cyan
$envPath = "$env:USERPROFILE\.conda\envs\self-ensemble-debug"
if (Test-Path $envPath) {
    $response = Read-Host "环境已存在，是否重新创建？(y/N)"
    if ($response -eq "y" -or $response -eq "Y") {
        Write-Host "删除旧环境..." -ForegroundColor Yellow
        conda env remove -n self-ensemble-debug -y
    } else {
        Write-Host "[SKIP] 使用现有环境" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "[3/3] 验证环境..." -ForegroundColor Cyan
        & "$envPath\python.exe" test_environment.py
        Write-Host ""
        Write-Host "[SUCCESS] 环境配置完成！" -ForegroundColor Green
        Write-Host "使用方法：双击 start_debug.bat 或运行 start_debug.ps1" -ForegroundColor Cyan
        Read-Host "按Enter键退出"
        exit 0
    }
}

conda env create -f environment_fixed.yml
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] 环境创建失败！" -ForegroundColor Red
    Read-Host "按Enter键退出"
    exit 1
}
Write-Host "[SUCCESS] 环境创建成功" -ForegroundColor Green

Write-Host "[3/3] 验证环境..." -ForegroundColor Cyan
& "$envPath\python.exe" test_environment.py

Write-Host ""
Write-Host "========================" -ForegroundColor Green
Write-Host "🎉 环境配置完成！" -ForegroundColor Green
Write-Host "========================" -ForegroundColor Green
Write-Host ""
Write-Host "使用方法：" -ForegroundColor Cyan
Write-Host "  方法1: 双击 start_debug.bat" -ForegroundColor White
Write-Host "  方法2: 运行 .\start_debug.ps1" -ForegroundColor White
Write-Host "  方法3: F5 在 VSCode 中调试" -ForegroundColor White
Write-Host ""

Read-Host "按Enter键退出"