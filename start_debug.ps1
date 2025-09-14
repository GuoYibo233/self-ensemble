# Self-Ensemble 调试环境启动脚本
Write-Host "===== Self-Ensemble 调试环境 =====" -ForegroundColor Green
Write-Host ""

# 设置 Anaconda 路径
$anacondaPath = "C:\ProgramData\anaconda3"
$env:PATH = "$anacondaPath;$anacondaPath\Scripts;$anacondaPath\Library\bin;" + $env:PATH

# 检查环境是否存在
$envPath = "$env:USERPROFILE\.conda\envs\self-ensemble-debug"
if (!(Test-Path $envPath)) {
    Write-Host "[INFO] 环境不存在，正在创建..." -ForegroundColor Yellow
    conda env create -f environment_fixed.yml
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] 环境创建失败！" -ForegroundColor Red
        Read-Host "按Enter键退出"
        exit 1
    }
    Write-Host "[SUCCESS] 环境创建成功！" -ForegroundColor Green
    Write-Host ""
}

Write-Host "[INFO] 环境已准备就绪" -ForegroundColor Cyan
Write-Host "[INFO] 可用的调试方法:" -ForegroundColor Cyan
Write-Host "  1. per_prompt   - 逐提示生成" -ForegroundColor White
Write-Host "  2. avg          - 平均集成" -ForegroundColor White  
Write-Host "  3. weighted_avg - 加权平均集成" -ForegroundColor White
Write-Host "  4. max          - 最大值集成" -ForegroundColor White
Write-Host "  5. weighted_max - 加权最大值集成" -ForegroundColor White
Write-Host ""

$method = Read-Host "请选择方法 (1-5, 默认=1)"
if ([string]::IsNullOrEmpty($method)) { $method = "1" }

$methodName = switch ($method) {
    "1" { "per_prompt" }
    "2" { "avg" }
    "3" { "weighted_avg" }
    "4" { "max" }
    "5" { "weighted_max" }
    default { "per_prompt" }
}

Write-Host ""
Write-Host "[INFO] 运行方法: $methodName" -ForegroundColor Cyan
Write-Host "[INFO] 启动调试脚本..." -ForegroundColor Cyan
Write-Host ""

& "$envPath\python.exe" debug_generate.py --method $methodName

Write-Host ""
Write-Host "[INFO] 调试完成！" -ForegroundColor Green
Read-Host "按Enter键退出"