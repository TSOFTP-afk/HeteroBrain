# =============================================================================
# LLM 子系统 — 进程内 llama.cpp (MiniCPM5-1B INT4 GGUF)
# =============================================================================
# 模块:
#   llama_backend  - LlmBackend SPI 适配器 (模型加载 / 采样器链 / chat 生成)
#
# 链接方式 (默认): 预编译静态库 F:\hb_build (llama.lib + ggml*.lib + CUDA libs),
#   链接清单与 llama-simple-chat.exe 一致 (F:\hb_build\build.ninja L5688 验证)。
# 备选: VITA_USE_LLAMA_SUBDIR=ON 时从 third_party/llama.cpp 源码构建。
#
# 演进槽位 (后续实现):
#   tokenizer_bridge / prompt_builder / logit_bias 情感偏置
# =============================================================================

cmake_minimum_required(VERSION 3.18)
project(llm_subsystem LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# =============================================================================
# llama.cpp 预编译产物路径 (可缓存覆盖)
# =============================================================================
set(HB_LLAMA_INCLUDE "F:/hb_llama/include" CACHE PATH "llama.cpp include dir")
set(HB_LLAMA_BUILD   "F:/hb_build"         CACHE PATH "llama.cpp prebuilt build dir")
set(HB_CUDA_LIB      "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/lib/x64"
    CACHE PATH "CUDA toolkit lib dir")

option(VITA_USE_LLAMA_SUBDIR "Build llama.cpp from source (third_party/llama.cpp)" OFF)

if(VITA_USE_LLAMA_SUBDIR)
    add_subdirectory(third_party/llama.cpp)
    set(LLAMA_TARGET llama)
else()
    add_library(llama_prebuilt INTERFACE)
    target_include_directories(llama_prebuilt INTERFACE
        ${HB_LLAMA_INCLUDE}            # llama.h
        ${HB_LLAMA_INCLUDE}/../ggml/include  # ggml.h / gguf.h
    )
    target_link_libraries(llama_prebuilt INTERFACE
        ${HB_LLAMA_BUILD}/src/llama.lib
        ${HB_LLAMA_BUILD}/ggml/src/ggml.lib
        ${HB_LLAMA_BUILD}/ggml/src/ggml-cpu.lib
        ${HB_LLAMA_BUILD}/ggml/src/ggml-cuda/ggml-cuda.lib
        ${HB_LLAMA_BUILD}/ggml/src/ggml-base.lib
        ${HB_CUDA_LIB}/cudart.lib
        ${HB_CUDA_LIB}/cublas.lib
        ${HB_CUDA_LIB}/cublasLt.lib
        ${HB_CUDA_LIB}/cuda.lib
    )
    set(LLAMA_TARGET llama_prebuilt)
endif()

# =============================================================================
# Target: llm_subsystem
# =============================================================================
set(LLM_SRCS
    llama_backend.cpp
    # 未来模块在此追加:
    # tokenizer_bridge.cpp
    # prompt_builder.cpp
)

set(LLM_HEADERS
    llama_backend.h
)

add_library(llm_subsystem STATIC ${LLM_SRCS} ${LLM_HEADERS})

target_include_directories(llm_subsystem
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
    PRIVATE
        ${HB_LLAMA_INCLUDE}
)

target_link_libraries(llm_subsystem PUBLIC ${LLAMA_TARGET})

if(MSVC)
    target_compile_options(llm_subsystem PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CXX>:/wd4244>
        $<$<COMPILE_LANGUAGE:CXX>:/wd4267>
        $<$<COMPILE_LANGUAGE:CXX>:/wd4305>
    )
else()
    target_compile_options(llm_subsystem PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wno-unused-parameter>
    )
endif()

# =============================================================================
# Target: bridge_llm_demo (全链路验证: EmotionBridge → LlamaBackend → MiniCPM5)
# =============================================================================
add_executable(bridge_llm_demo
    ${CMAKE_CURRENT_SOURCE_DIR}/bridge_llm_demo.cpp
)

target_link_libraries(bridge_llm_demo PRIVATE
    llm_subsystem
    bridge_subsystem
)

if(MSVC)
    target_compile_options(bridge_llm_demo PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
    )
endif()

set_target_properties(bridge_llm_demo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
