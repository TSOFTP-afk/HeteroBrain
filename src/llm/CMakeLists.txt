# =============================================================================
# LLM 子系统 — MiniCPM5-1B INT4 GGUF + llama.cpp
# =============================================================================
# 模块:
#   llama_runner     - llama.cpp 推理封装 (单例, 复用模型上下文)
#   tokenizer_bridge - BPE 分词器 (与 SNN BPE 数据流对齐)
#   prompt_builder   - 上下文构造 (用户输入 + SNN 检索 + RAG 知识)
# =============================================================================

set(LLM_SRCS
    llama_runner.cpp
    tokenizer_bridge.cpp
    prompt_builder.cpp
)

set(LLM_HEADERS
    llama_runner.h
    tokenizer_bridge.h
    prompt_builder.h
)

# Phase 1 占位: 暂无源文件, 用空 target 保持 CMake 结构
if(LLM_SRCS)
    add_library(llm_subsystem STATIC ${LLM_SRCS} ${LLM_HEADERS})
else()
    add_library(llm_subsystem STATIC INTERFACE)
endif()

target_include_directories(llm_subsystem
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/third_party
)

# TODO: Phase 1 实现 —
#   1. 引入 llama.cpp 作为 third_party/llama.cpp 子模块
#   2. 实现 llama_runner.cpp: 加载 GGUF, 单批推理, 流式输出
#   3. 实现 tokenizer_bridge.cpp: BPE 编码/解码 (与 SNN BPE stream 对齐)
#   4. 实现 prompt_builder.cpp: 拼接 system + history + snn_context + user
