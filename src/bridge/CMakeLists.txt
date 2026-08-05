# =============================================================================
# Bridge 链接层 — SNN 情感核心 ↔ LLM (llama.cpp) 之间的可复用桥接
# =============================================================================
# 模块:
#   emotion_types.h    - 数据契约 (EmotionState / SamplerParams / MappingConfig)
#   affective_mapping  - 情感状态 → 采样参数映射 (纯函数, 配置驱动)
#   emotion_prompt     - 情感 system prompt / 心情描述生成
#   llm_backend.h      - LLM 后端 SPI (进程内 llama.cpp 适配器后续实现)
#   snn_feedback.h     - SNN 反馈接收端 SPI (LLM→SNN 回流)
#   emotion_bridge     - 编排器: SNN→LLM 应用 + LLM→SNN 回流
#
# 设计约束 (可复用性 / 扩展性):
#   1. 桥接层核心为纯 C++17, 不依赖 CUDA / llama.cpp 头文件, 可独立构建与测试
#      (本 CMakeLists 支持 standalone 构建, 与 src/snn 一致)
#   2. 后端与反馈端均通过 SPI 注入, 运行期可替换, 桥接层代码零改动
#   3. 未来 T2H 模块 (spike_embedding / truth_filter / pca_projection) 以独立
#      源文件加入 BRIDGE_SRCS, 复用 emotion_types 契约, 不侵入现有接口
# =============================================================================

cmake_minimum_required(VERSION 3.18)
project(bridge_subsystem LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# =============================================================================
# 源文件清单
# =============================================================================
set(BRIDGE_SRCS
    affective_mapping.cpp
    emotion_prompt.cpp
    emotion_bridge.cpp
    emotion_event.cpp
    # 未来 T2H 模块在此追加:
    # spike_embedding.cpp
    # truth_filter.cpp
    # pca_projection.cpp
)

set(BRIDGE_HEADERS
    emotion_types.h
    affective_mapping.h
    emotion_prompt.h
    llm_backend.h
    snn_feedback.h
    emotion_bridge.h
    emotion_event.h
)

# =============================================================================
# Target: bridge_subsystem
# =============================================================================
add_library(bridge_subsystem STATIC ${BRIDGE_SRCS} ${BRIDGE_HEADERS})

target_include_directories(bridge_subsystem
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

if(MSVC)
    # /utf-8: 情感 prompt 源文件含中文字符; 关闭隐式转换告警
    target_compile_options(bridge_subsystem PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CXX>:/wd4244>
        $<$<COMPILE_LANGUAGE:CXX>:/wd4267>
        $<$<COMPILE_LANGUAGE:CXX>:/wd4305>
    )
else()
    target_compile_options(bridge_subsystem PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wno-unused-parameter>
    )
endif()

# =============================================================================
# Target: test_emotion_bridge (纯 C++ 单元测试, 无 CUDA / llama.cpp 依赖)
# =============================================================================
add_executable(test_emotion_bridge
    ${CMAKE_CURRENT_SOURCE_DIR}/test_emotion_bridge.cpp
)

target_link_libraries(test_emotion_bridge PRIVATE bridge_subsystem)
target_include_directories(test_emotion_bridge PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if(MSVC)
    target_compile_options(test_emotion_bridge PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CXX>:/wd4244>
        $<$<COMPILE_LANGUAGE:CXX>:/wd4267>
        $<$<COMPILE_LANGUAGE:CXX>:/wd4305>
    )
endif()

set_target_properties(test_emotion_bridge PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
