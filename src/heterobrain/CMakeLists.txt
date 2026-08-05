# =============================================================================
# HeteroBrain 顶层引擎 — 异构主循环 (SNN 情感核心 + EmotionBridge + LLM 进程内接线)
# =============================================================================
# 模块:
#   engine  - EmotionEngine: 对话主循环 (事件注入 → 训练推进 → 情感调制 → LLM 生成)
#   main    - CLI 入口 (--resume <snn.ckpt> --llm <model.gguf> ...)
#
# 依赖: src/snn 核心源 (直接编译进引擎, 规避静态库 PDB nvcc bug)
#       + llm_subsystem (llama.cpp) + bridge_subsystem
# =============================================================================

find_package(CUDAToolkit REQUIRED)

set(ENGINE_SRCS
    engine.cpp
    main.cpp
)

set(ENGINE_HEADERS
    engine.h
    http_server.h
    mini_json.h
)

# =============================================================================
# SNN 核心源: 直接编译进引擎 (可执行目标, 复用目录形式 PDB, 规避静态库 nvcc bug)
# 排除: main.cpp/decoder_main.cpp/test_*.cpp (各自含 main())
# =============================================================================
file(GLOB SNN_CORE_SRCS CONFIGURE_DEPENDS
    ${CMAKE_SOURCE_DIR}/src/snn/*.cu
    ${CMAKE_SOURCE_DIR}/src/snn/*.cpp)
list(REMOVE_ITEM SNN_CORE_SRCS
    ${CMAKE_SOURCE_DIR}/src/snn/main.cpp
    ${CMAKE_SOURCE_DIR}/src/snn/decoder_main.cpp
    ${CMAKE_SOURCE_DIR}/src/snn/test_embodied.cpp
    ${CMAKE_SOURCE_DIR}/src/snn/test_event_scheduler.cpp)

add_executable(heterobrain_engine ${ENGINE_SRCS} ${ENGINE_HEADERS} ${SNN_CORE_SRCS})

target_link_libraries(heterobrain_engine
    PRIVATE
        llm_subsystem
        bridge_subsystem
        CUDA::cudart
        ws2_32
)

target_include_directories(heterobrain_engine
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src/snn
        ${CUDAToolkit_INCLUDE_DIRS}
)

# decoder.cu 实为 host 代码 (与 src/snn 的 snn_decoder 一致), 强制 CXX 编译加速
set_source_files_properties(
    ${CMAKE_SOURCE_DIR}/src/snn/decoder.cu
    PROPERTIES LANGUAGE CXX
)

# CUDA 编译属性 (与 snn_train 一致): sm_86 + 可分离编译 + -O3
target_compile_options(heterobrain_engine PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
)

set_target_properties(heterobrain_engine PROPERTIES
    CUDA_ARCHITECTURES "86"
    CUDA_SEPARABLE_COMPILATION ON
)

# 同一进程内两个 CUDA 使用者 (SNN kernels + ggml-cuda) 共享 cudart
# 引擎链接 snn_subsystem (CUDA::cudart 传播) + llama_prebuilt (CUDA libs 传播)

if(MSVC)
    target_compile_options(heterobrain_engine PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
        $<$<COMPILE_LANGUAGE:CXX>:/wd4244>
        $<$<COMPILE_LANGUAGE:CXX>:/wd4267>
        $<$<COMPILE_LANGUAGE:CXX>:/wd4305>
        $<$<COMPILE_LANGUAGE:CXX>:/bigobj>
    )
else()
    target_compile_options(heterobrain_engine PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wno-unused-parameter>
    )
endif()

set_target_properties(heterobrain_engine PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# 安装
install(TARGETS heterobrain_engine
    RUNTIME DESTINATION bin
)
