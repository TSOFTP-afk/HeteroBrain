@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul
cd /d F:\thetrueai\build\bin
echo [INFO] llama-cli version check...
llama-cli.exe --version
echo.
echo [INFO] llama-cli help (chat-template-file section)...
llama-cli.exe --help 2>&1 | findstr /C:"chat-template" /C:"--jinja"
