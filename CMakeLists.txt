# =============================================================================
# vita — 顶层 CMake 构建文件
# =============================================================================
# 异构中文对话 AI 引擎: SNN (CUDA) + LLM (llama.cpp/MiniCPM5-1B) + Bridge
#
# 构建目标:
#   vita_engine  - 主引擎可执行文件
#   snn_subsystem       - SNN 子系统静态库
#   llm_subsystem       - LLM 子系统静态库
#   bridge_subsystem    - Bridge 转换层静态库
# =============================================================================

cmake_minimum_required(VERSION 3.18)
project(vita
    VERSION 0.1.0
    DESCRIPTION "异构中文对话 AI 引擎 (SNN × LLM)"
    LANGUAGES CXX CUDA
)

# =============================================================================
# 全局选项
# =============================================================================
option(VITA_BUILD_TESTS    "Build test suite"           ON)
option(VITA_ENABLE_CUDA    "Enable CUDA SNN subsystem"  ON)
option(VITA_ENABLE_LLM     "Enable LLM subsystem"       ON)
option(VITA_BUILD_SHARED   "Build shared libs"          OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# =============================================================================
# 编译器要求
# =============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# CUDA 架构: sm_86 (RTX 30/40/A) + sm_120 (DGX Spark GB10)
if(VITA_ENABLE_CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "86;120")
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    if(MSVC)
        # 注意: -Xcompiler 参数必须逗号分隔 (/wd4819,/wd4505),
        # 空格分隔会让第二个 /wd 选项变成游离 token 被 nvcc 当作输入文件
        # (nvcc fatal: single input file required; 2026-08-05 踩坑)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/wd4819,/wd4505")
    else()
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wno-deprecated-declarations")
    endif()
endif()

# =============================================================================
# 全局包含路径
# =============================================================================
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/third_party)

# =============================================================================
# 子模块
# =============================================================================
# llama.cpp (后续作为子模块引入 third_party/llama.cpp)
# if(EXISTS "${CMAKE_SOURCE_DIR}/third_party/llama.cpp/CMakeLists.txt")
#     set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
#     set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
#     add_subdirectory(third_party/llama.cpp)
# endif()

# =============================================================================
# 子系统
# =============================================================================
add_subdirectory(src/snn)
add_subdirectory(src/llm)
add_subdirectory(src/bridge)
add_subdirectory(src/vita)

# =============================================================================
# 测试
# =============================================================================
if(VITA_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# =============================================================================
# 安装
# =============================================================================
# vita_engine 目标由 src/vita 的 if(ENGINE_SRCS) 保护, 未实现时优雅跳过
if(TARGET vita_engine)
    install(TARGETS vita_engine
        RUNTIME DESTINATION bin
    )
endif()

install(DIRECTORY configs/
    DESTINATION etc/vita
    FILES_MATCHING PATTERN "*.yaml"
)

# =============================================================================
# 信息输出
# =============================================================================
message(STATUS "============================================================")
message(STATUS "vita ${PROJECT_VERSION}")
message(STATUS "  Build type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA enabled:     ${VITA_ENABLE_CUDA}")
message(STATUS "  LLM enabled:      ${VITA_ENABLE_LLM}")
if(VITA_ENABLE_CUDA)
    message(STATUS "  CUDA archs:       ${CMAKE_CUDA_ARCHITECTURES}")
endif()
message(STATUS "  Tests:            ${VITA_BUILD_TESTS}")
message(STATUS "============================================================")
