@echo off
title VITA - Digital Life Awakening
chcp 65001 >nul
cd /d "%~dp0"

REM 检测 Python
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python 未安装或不在 PATH 中，请先安装 Python 3.10+。
    pause
    exit /b 1
)

python main.py

if errorlevel 1 (
    echo.
    echo [ERROR] VITA 启动失败，请检查上方报错信息。
    pause
)
