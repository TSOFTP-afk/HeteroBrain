@echo off
REM Switch console to UTF-8 to avoid Chinese input garbled
chcp 65001 >nul
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul
cd /d F:\thetrueai\build\bin
echo [INFO] MiniCPM5-1B interactive chat (Ctrl+C to exit, empty line to quit)
echo [INFO] Model : F:\hb_models\MiniCPM5-1B-Q4_K_M.gguf
echo [INFO] Tmpl  : F:\hb_models\minicpm5-chat.jinja
echo.
llama-cli.exe ^
    -m F:\hb_models\MiniCPM5-1B-Q4_K_M.gguf ^
    --chat-template-file F:\hb_models\minicpm5-chat.jinja ^
    --jinja ^
    -ngl 99 ^
    -c 2048 ^
    --temp 0.7 ^
    --top-p 0.9 ^
    --top-k 40 ^
    --repeat-penalty 1.1 ^
    --color on ^
    -cnv
echo.
echo [INFO] Exit code: %ERRORLEVEL%
