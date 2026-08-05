#!/usr/bin/env python3
"""
vita 构建脚本 (Linux / WSL)

用法:
    ./scripts/build.sh           # Release 构建
    ./scripts/build.sh debug     # Debug 构建
    ./scripts/build.sh clean     # 清理构建
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
BUILD = ROOT / "build"


def run(cmd, cwd=ROOT):
    print(f"$ {cmd}")
    subprocess.run(cmd, shell=True, cwd=cwd, check=True)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "release"

    if mode == "clean":
        if BUILD.exists():
            run(f"rm -rf {BUILD}")
        print("清理完成")
        return

    build_type = "Debug" if mode == "debug" else "Release"
    BUILD.mkdir(exist_ok=True)

    # CMake 配置
    generator = "Ninja" if subprocess.run("which ninja", shell=True, capture_output=True).returncode == 0 else "Unix Makefiles"
    run(f"cmake -G {generator} -DCMAKE_BUILD_TYPE={build_type} ..", cwd=BUILD)

    # 编译
    if generator == "Ninja":
        run("ninja vita_engine", cwd=BUILD)
    else:
        run("make vita_engine -j$(nproc)", cwd=BUILD)

    print(f"\n构建完成: {BUILD}/vita_engine")
    print(f"运行: ./{BUILD.relative_to(ROOT)}/vita_engine --interactive")


if __name__ == "__main__":
    main()
