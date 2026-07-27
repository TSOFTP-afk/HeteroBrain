@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul
if errorlevel 1 (
    echo [ERROR] vcvarsall failed
    exit /b 1
)
if not exist F:\thetrueai\build\snn mkdir F:\thetrueai\build\snn
echo [INFO] CMake configure with Release build type...
cmake -G Ninja -S F:\thetrueai\src\snn -B F:\thetrueai\build\snn -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 (
    echo [ERROR] cmake configure failed
    exit /b 1
)
echo [OK] CMake configured
echo [INFO] Building snn_train (Release)...
ninja -C F:\thetrueai\build\snn snn_train
if errorlevel 1 (
    echo [ERROR] ninja snn_train failed
    exit /b 1
)
echo [OK] snn_train built
dir F:\thetrueai\build\snn\bin\snn_train.exe
