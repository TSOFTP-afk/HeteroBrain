@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul
if errorlevel 1 (
    echo [ERROR] vcvarsall failed
    exit /b 1
)
if not exist F:\thetrueai\build\heterobrain_v2 mkdir F:\thetrueai\build\heterobrain_v2
if not exist F:\thetrueai\build\heterobrain_v2\build.ninja (
    echo [INFO] CMake configure with Release build type...
    cmake -G Ninja -S F:\thetrueai -B F:\thetrueai\build\heterobrain_v2 ^
        -DCMAKE_BUILD_TYPE=Release ^
        -DLLAMA_BUILD_TESTS=OFF ^
        -DGGML_CUDA=ON ^
        -DGGML_CUDA_ARCH=86
    if errorlevel 1 (
        echo [ERROR] cmake configure failed
        exit /b 1
    )
    echo [OK] CMake configured
)
echo [INFO] Building vita_engine (Release)...
ninja -C F:\thetrueai\build\heterobrain_v2 vita_engine
if errorlevel 1 (
    echo [ERROR] ninja vita_engine failed
    exit /b 1
)
echo [OK] vita_engine built
dir F:\thetrueai\build\heterobrain_v2\bin\vita_engine.exe