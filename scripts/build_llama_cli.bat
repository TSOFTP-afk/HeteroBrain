@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul
if errorlevel 1 (
    echo [ERROR] vcvarsall failed
    exit /b 1
)
if not exist F:\thetrueai\build mkdir F:\thetrueai\build
cd /d F:\thetrueai\build
echo [INFO] CMake configure with Release build type...
cmake -G Ninja -S F:\hb_llama -B F:\thetrueai\build ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DGGML_CUDA=ON ^
    -DGGML_CUDA_ARCH=86 ^
    -DLLAMA_BUILD_TOOLS=ON ^
    -DLLAMA_BUILD_SERVER=ON ^
    -DGGML_NATIVE_ARCH=OFF
if errorlevel 1 (
    echo [ERROR] cmake configure failed
    exit /b 1
)
echo [OK] CMake configured
echo [INFO] Building llama-cli (Release)...
ninja llama-cli
if errorlevel 1 (
    echo [ERROR] ninja llama-cli failed
    exit /b 1
)
echo [OK] llama-cli built
dir F:\thetrueai\build\bin\llama-cli.exe
