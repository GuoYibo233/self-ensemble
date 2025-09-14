@echo off
echo ===== Self-Ensemble 调试环境 =====
echo.

REM 设置 Anaconda 路径
set ANACONDA_PATH=C:\ProgramData\anaconda3
set PATH=%ANACONDA_PATH%;%ANACONDA_PATH%\Scripts;%ANACONDA_PATH%\Library\bin;%PATH%

REM 检查环境是否存在
if not exist "%USERPROFILE%\.conda\envs\self-ensemble-debug" (
    echo [INFO] 环境不存在，正在创建...
    conda env create -f environment_fixed.yml
    if errorlevel 1 (
        echo [ERROR] 环境创建失败！
        pause
        exit /b 1
    )
    echo [SUCCESS] 环境创建成功！
    echo.
)

echo [INFO] 环境已准备就绪
echo [INFO] 可用的调试方法:
echo   1. per_prompt  - 逐提示生成
echo   2. avg         - 平均集成  
echo   3. weighted_avg - 加权平均集成
echo   4. max         - 最大值集成
echo   5. weighted_max - 加权最大值集成
echo.

set /p method="请选择方法 (1-5, 默认=1): "
if "%method%"=="" set method=1
if "%method%"=="1" set method_name=per_prompt
if "%method%"=="2" set method_name=avg
if "%method%"=="3" set method_name=weighted_avg
if "%method%"=="4" set method_name=max
if "%method%"=="5" set method_name=weighted_max

echo.
echo [INFO] 运行方法: %method_name%
echo [INFO] 启动调试脚本...
echo.

"%USERPROFILE%\.conda\envs\self-ensemble-debug\python.exe" debug_generate.py --method %method_name%

echo.
echo [INFO] 调试完成！
pause