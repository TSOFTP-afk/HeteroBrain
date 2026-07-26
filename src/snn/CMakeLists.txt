# =============================================================================
# SNN 子系统 — CUDA 脉冲神经网络 (60K 神经元 / 10.7M 突触)
# =============================================================================
# 模块:
#   bptt_trainer     - BPTT 代理梯度训练器 (从 legacy/stage2e 移植)
#   pca_signatures   - PCA 签名提取 (长程记忆索引用)
#   memory_index     - 长程记忆库 Top-K 检索
#   online_stdp      - 在线 STDP 微调 (用户反馈闭环)
# =============================================================================

set(SNN_SRCS
    bptt_trainer.cu
    pca_signatures.cu
    memory_index.cu
    online_stdp.cu
)

set(SNN_HEADERS
    bptt_trainer.cuh
    pca_signatures.cuh
    memory_index.cuh
    online_stdp.cuh
)

add_library(snn_subsystem STATIC ${SNN_SRCS} ${SNN_HEADERS})

target_include_directories(snn_subsystem
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/legacy/stage2e
)

target_compile_features(snn_subsystem PUBLIC cxx_std_17)

# CUDA 架构继承自顶层
if(HETEROBRAIN_ENABLE_CUDA)
    set_target_properties(snn_subsystem PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )
endif()

# TODO: Phase 2 实现 — 从 legacy/stage2e 移植以下文件:
#   - bptt_trainer.cu/.cuh  (legacy/stage2e/bptt_trainer.cu 35KB)
#   - pca_kernels.cu/.cuh   (legacy/stage2e/pca_kernels.cu 10KB)
#   - synapse_kernels.cu    (legacy/stage2e/synapse_kernels.cu 37KB, STDP 双trace)
#   - neuron_kernels.cu     (legacy/stage2e/neuron_kernels.cu 17KB, AdEx)
#   - network_init.cu       (legacy/stage2e/network_init.cu 39KB, 50柱拓扑)
