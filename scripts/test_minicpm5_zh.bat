@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul
cd /d F:\thetrueai\build\bin
echo [INFO] Running llama-cli with MiniCPM5-1B + Jinja template...
echo [INFO] Model: F:\hb_models\MiniCPM5-1B-Q4_K_M.gguf
echo [INFO] Template: F:\hb_models\minicpm5-chat.jinja
echo [INFO] Prompt file: F:\thetrueai\logs\test_prompt.txt
echo.
llama-cli.exe ^
    -m F:\hb_models\MiniCPM5-1B-Q4_K_M.gguf ^
    --chat-template-file F:\hb_models\minicpm5-chat.jinja ^
    --jinja ^
    -f F:\thetrueai\logs\test_prompt.txt ^
    -st ^
    -n 400 ^
    -ngl 99 ^
    -c 2048 ^
    --temp 0.7 ^
    --top-p 0.9 ^
    --top-k 40 ^
    --repeat-penalty 1.1 ^
    --seed 42 ^
    --no-warmup
echo.
echo [INFO] Exit code: %ERRORLEVEL%
