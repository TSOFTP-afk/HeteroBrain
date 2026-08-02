# =============================================================================
# SNN 训练子系统 — Stage 2e 移植版 (60K 神经元 / 10.7M 突触)
# =============================================================================
# 从 legacy/stage2e/ 整体移植, 作为 Phase 2 SNN 独立训练入口
#
# 目标: snn_train.exe (独立可执行, 不依赖 ggml/llama.cpp)
# CUDA: sm_86 (RTX 3060), C++17 / CUDA 17
# 输出: ${CMAKE_BINARY_DIR}/bin/snn_train.exe
# =============================================================================

cmake_minimum_required(VERSION 3.18)
project(snn_train LANGUAGES CXX CUDA)

# 默认 Release 配置: 避免 MSVC Debug 模式 /RTC1 与 nvcc -O3 传过去的 /O2 冲突
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.0)
    message(FATAL_ERROR "CUDA 11.0+ required")
endif()

find_package(CUDAToolkit REQUIRED)

# MSVC 兼容: UTF-8 源编码 (全局, 所有 target 共享)
if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
endif()

# =============================================================================
# 源文件清单
# =============================================================================
# snn_train: 主训练可执行 (含 main.cpp, 不含 decoder_main.cpp, 避免 main() 冲突)
set(SNN_TRAIN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scheduler.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/scheduler_checkpoint.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/bptt_trainer.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/bptt_curriculum.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/csv_stats.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/curriculum_loader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/synapse_kernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/neuron_kernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/modulatory_kernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/coactivation_kernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/hippocampal_kernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/pca_kernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/thalamic_gate.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/wm_kernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/decode_kernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/network_init.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/memory_allocator.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/input_encoding.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/run_config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/event_scheduler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/embodied_env.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/embodied_motor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/embodied_motor.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/multi_sensory_inject.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/motor_teacher.cu
)

# snn_decoder: 独立解码评估工具 (host-only, decoder.cu 实为 host 代码, 强制 CXX 编译)
set(SNN_DECODER_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/decoder.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/decoder_main.cpp
)

# =============================================================================
# Target: snn_train (主训练入口)
# =============================================================================
add_executable(snn_train ${SNN_TRAIN_SRCS})

target_include_directories(snn_train PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/tools
    ${CUDAToolkit_INCLUDE_DIRS}
)

# 不依赖 ggml/llama.cpp, 仅链接 CUDA runtime
target_link_libraries(snn_train PRIVATE CUDA::cudart)

# MSVC 兼容: 禁用隐式转换警告 + /bigobj
if(MSVC)
    target_compile_options(snn_train PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/wd4244>    # conversion from 'type1' to 'type2', possible loss of data
        $<$<COMPILE_LANGUAGE:CXX>:/wd4267>    # conversion from 'size_t' to 'type', possible loss of data
        $<$<COMPILE_LANGUAGE:CXX>:/wd4305>    # truncation from 'double' to 'float'
        $<$<COMPILE_LANGUAGE:CXX>:/bigobj>    # 增大目标文件节区数 (10.7M 突触相关 TU 节区数较多)
    )
else()
    target_compile_options(snn_train PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wno-unused-parameter>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wno-unused-parameter>
    )
endif()

# =============================================================================
# Phase 3a-C1: EventScheduler 单元测试 (无 CUDA 依赖, 纯 host 逻辑)
# test_event_scheduler.cpp 提供 set_event_signal stub, 不链接 modulatory_kernels.cu
# =============================================================================
add_executable(test_event_scheduler
    ${CMAKE_CURRENT_SOURCE_DIR}/test_event_scheduler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/event_scheduler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/curriculum_loader.cpp
)
target_include_directories(test_event_scheduler PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if(MSVC)
    target_compile_options(test_event_scheduler PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/wd4244>
        $<$<COMPILE_LANGUAGE:CXX>:/wd4267>
        $<$<COMPILE_LANGUAGE:CXX>:/wd4305>
    )
endif()

# =============================================================================
# Phase 3a-D1: Embodied Environment 单元测试 (无 CUDA 依赖)
# test_embodied.cpp 定义 SNN_NO_CUDA, 排除 read_motor_output (GPU版本)
# =============================================================================
add_executable(test_embodied
    ${CMAKE_CURRENT_SOURCE_DIR}/test_embodied.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/embodied_env.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/embodied_motor.cpp
)
target_include_directories(test_embodied PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(test_embodied PRIVATE SNN_NO_CUDA)
if(MSVC)
    target_compile_options(test_embodied PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CXX>:/wd4244>
        $<$<COMPILE_LANGUAGE:CXX>:/wd4267>
        $<$<COMPILE_LANGUAGE:CXX>:/wd4305>
    )
endif()
set_target_properties(test_embodied PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# CUDA 编译选项: -O3 (Release) + -lineinfo (profiling 用 debug 信息)
target_compile_options(snn_train PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
)

# CUDA 架构: sm_86 (RTX 3060 Ampere)
set_target_properties(snn_train PROPERTIES
    CUDA_ARCHITECTURES "86"
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# =============================================================================
# Target: snn_decoder (解码评估工具, host-only)
# =============================================================================
add_executable(snn_decoder ${SNN_DECODER_SRCS})

target_include_directories(snn_decoder PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

# decoder.cu 实为 host 代码 (legacy 设计: 用 .cu 扩展名但无 __device__/__global__ kernel)
# 强制用 CXX 编译以加速构建, 并避免 nvcc 不必要的启动
set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/decoder.cu
    PROPERTIES LANGUAGE CXX
)

if(MSVC)
    target_compile_options(snn_decoder PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/wd4244>
        $<$<COMPILE_LANGUAGE:CXX>:/wd4267>
        $<$<COMPILE_LANGUAGE:CXX>:/wd4305>
    )
else()
    target_compile_options(snn_decoder PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wno-unused-parameter>
    )
endif()

set_target_properties(snn_decoder PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
