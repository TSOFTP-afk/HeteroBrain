# =============================================================================
# HeteroBrain 顶层引擎 — 异构主循环
# =============================================================================
# 模块:
#   engine  - 异构引擎主循环 (输入 → SNN 检索 → LLM 生成 → 真实性筛选 → 输出)
#   config  - YAML 配置加载
#   main    - CLI 入口 (--interactive / --batch / --eval)
# =============================================================================

# Phase 3b-3f 实现后恢复源文件列表 (当前模块未实现, ENGINE_SRCS 为空):
set(ENGINE_SRCS
    # engine.cpp
    # config.cpp
    # main.cpp
)

set(ENGINE_HEADERS
    # engine.h
    # config.h
)

# 仅在源文件就绪时才定义可执行目标 (Phase 3b-3f 前优雅跳过)
if(ENGINE_SRCS)
    add_executable(heterobrain_engine ${ENGINE_SRCS} ${ENGINE_HEADERS})

    target_link_libraries(heterobrain_engine
        PRIVATE
            snn_subsystem
            llm_subsystem
            bridge_subsystem
    )

    target_include_directories(heterobrain_engine
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
    )

    # 安装
    install(TARGETS heterobrain_engine
        RUNTIME DESTINATION bin
    )
endif()

# TODO: Phase 1 实现 main.cpp 的最小骨架:
#   1. 解析命令行: --llm <gguf> --config <yaml> --interactive
#   2. 加载 LLM (llama_runner)
#   3. 进入交互循环: getline → prompt_build → llama_infer → print
#   4. Phase 2 后增加: --snn <ckpt> 加载 SNN 子系统
