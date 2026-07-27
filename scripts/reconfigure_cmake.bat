@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul
if errorlevel 1 (
    echo [ERROR] vcvarsall failed
    exit /b 1
)
cd /d F:\hb_build
echo [INFO] Reconfiguring CMake with LLAMA_BUILD_SERVER=ON...
cmake -G Ninja -S F:\hb_llama -B F:\hb_build ^
    -DGGML_CUDA=ON ^
    -DGGML_CUDA_ARCH=86 ^
    -DLLAMA_BUILD_TOOLS=ON ^
    -DLLAMA_BUILD_SERVER=ON
if errorlevel 1 (
    echo [ERROR] cmake reconfigure failed
    exit /b 1
)
echo [OK] CMake reconfigured
