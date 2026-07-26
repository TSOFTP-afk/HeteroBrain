# =============================================================================
# Bridge 转换层 — spike pattern ↔ LLM embedding 投影
# =============================================================================
# 模块:
#   spike_embedding  - spike 序列 → embedding 向量 ([2048,1024] Linear)
#   truth_filter     - token 真实性筛选 (spike 一致性校验)
#   pca_projection   - PCA 签名空间投影 (LLM embedding ↔ SNN PCA)
# =============================================================================

set(BRIDGE_SRCS
    spike_embedding.cpp
    truth_filter.cpp
    pca_projection.cpp
)

set(BRIDGE_HEADERS
    spike_embedding.h
    truth_filter.h
    pca_projection.h
)

if(BRIDGE_SRCS)
    add_library(bridge_subsystem STATIC ${BRIDGE_SRCS} ${BRIDGE_HEADERS})
else()
    add_library(bridge_subsystem STATIC INTERFACE)
endif()

target_include_directories(bridge_subsystem
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src/snn
        ${CMAKE_SOURCE_DIR}/src/llm
)

target_link_libraries(bridge_subsystem
    PUBLIC
        snn_subsystem
        llm_subsystem
)

# TODO: Phase 3 实现 —
#   1. spike_embedding.cpp: 训练 [2048, 1024] 投影矩阵 (PyTorch 离线训练后导出权重)
#   2. pca_projection.cpp: 加载 SNN PCA W 矩阵 (legacy/stage2e/pca_kernels.cu)
#   3. truth_filter.cpp: 实现 spike 一致性校验 (Top-K PCA 签名匹配)
