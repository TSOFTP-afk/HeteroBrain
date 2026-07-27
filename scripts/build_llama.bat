@echo off
REM ============================================================
REM HeteroBrain - llama.cpp 编译脚本 (CUDA + RTX 3060 sm_86)
REM ============================================================
REM 在 VS 2022 x64 环境下 cmake + ninja 编译 llama.cpp
REM 输出: third_party/llama.cpp/build/bin/Release/llama-cli.exe
REM ============================================================

setlocal

set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
set "CMAKE_DIR=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin"
set "PATH=%CMAKE_DIR%;%PATH%"
set "SRC=f:\项目\THE TRUE AI\third_party\llama.cpp"
set "BUILD=%SRC%\build"

if not exist "%VCVARS%" (
    echo [ERROR] vcvarsall.bat 未找到: %VCVARS%
    exit /b 1
)

echo === 加载 VS x64 编译环境 ===
call "%VCVARS%" amd64
if errorlevel 1 (
    echo [ERROR] vcvarsall 加载失败
    exit /b 1
)

echo.
echo === 检查工具 ===
cmake --version | findstr "cmake version"
nvcc --version 2>nul | findstr "release"
ninja --version 2>nul

echo.
echo === CMake 配置 (CUDA + sm_86) ===
if exist "%BUILD%" rmdir /s /q "%BUILD%"
mkdir "%BUILD%"
cd /d "%BUILD%"

cmake "%SRC%" ^
    -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DGGML_CUDA=ON ^
    -DGGML_CUDA_ARCH=86 ^
    -DLLAMA_CURL=OFF ^
    -DGGML_RPC=OFF ^
    -DBUILD_SHARED_LIBS=OFF ^
    -DLLAMA_BUILD_EXAMPLES=ON ^
    -DLLAMA_BUILD_TESTS=OFF ^
    -DLLAMA_BUILD_SERVER=OFF

if errorlevel 1 (
    echo [ERROR] CMake 配置失败
    exit /b 1
)

echo.
echo === 编译 llama-cli (这是大头, 预计 5-10 分钟) ===
cmake --build . --target llama-cli --config Release -j

if errorlevel 1 (
    echo [ERROR] 编译失败
    exit /b 1
)

echo.
echo === 验证产物 ===
if exist "%BUILD%\bin\llama-cli.exe" (
    echo [OK] %BUILD%\bin\llama-cli.exe
    "%BUILD%\bin\llama-cli.exe" --version
) else (
    echo [WARN] llama-cli.exe 未找到, 列出 bin 目录:
    dir "%BUILD%\bin\*.exe" 2>nul
)

endlocal
