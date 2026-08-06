@echo off
chcp 65001 >nul
title VITA - Artificial Affective Core
cd /d "%~dp0"

REM ============================================================
REM  VITA launcher
REM  1) verify engine / checkpoint / model files
REM  2) launch engine (init -> awakening animation -> mode select)
REM  NOTE: keep this file pure ASCII (cmd/GBK parse bug - project rule)
REM ============================================================

REM ---- config: edit paths here ----
set "ENGINE=build\root\bin\vita_engine.exe"
set "CHECKPOINT=checkpoints\middle_1a_longarc_all\ckpt_step110000.snn2e"
set "MODEL=F:\hb_models\Qwen3-4B-Q4_K_M.gguf"
REM must match the corpus the checkpoint was trained with (fingerprint check)
set "TEXT=data\scripts\story_text_all.txt"

REM ---- extra args (optional, e.g. --freeze-weights) ----
set "EXTRA="

REM ---- verify files ----
if not exist "%ENGINE%" (
    echo [ERROR] engine not found: %ENGINE%
    echo          build first: cmake --build build/root --target vita_engine
    pause
    exit /b 1
)
if not exist "%CHECKPOINT%" (
    echo [ERROR] checkpoint not found: %CHECKPOINT%
    echo          edit CHECKPOINT variable at top of this script
    pause
    exit /b 1
)
if not exist "%MODEL%" (
    echo [ERROR] model not found: %MODEL%
    echo          edit MODEL variable at top of this script
    pause
    exit /b 1
)

echo [OK] launching VITA...
echo.

REM ---- launch ----
"%ENGINE%" --resume "%CHECKPOINT%" --llm "%MODEL%" --text "%TEXT%" %EXTRA%

REM ---- pause on failure so the window stays ----
if errorlevel 1 (
    echo.
    echo [ERROR] VITA failed to start, see log above.
    pause
)