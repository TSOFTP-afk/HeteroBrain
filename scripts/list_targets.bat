@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul
cd /d F:\hb_build
ninja -t targets all | findstr /R "llama-"
