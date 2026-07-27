# =============================================================================
# HeteroBrain 顶层引擎 — 异构主循环
# =============================================================================
# 模块:
#   engine  - 异构引擎主循环 (输入 → SNN 检索 → LLM 生成 → 真实性筛选 → 输出)
#   config  - YAML 配置加载
#   main    - CLI 入口 (--interactive / --batch / --eval)
# =============================================================================

add_library(heterobrain_router STATIC
    dynamic_router.cpp
    dynamic_router.h
)
target_include_directories(heterobrain_router
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

set(ENGINE_SRCS
    engine.cpp
    config.cpp
    main.cpp
)

set(ENGINE_HEADERS
    engine.h
    config.h
)

set(HETEROBRAIN_ENGINE_READY TRUE)
foreach(source_file IN LISTS ENGINE_SRCS ENGINE_HEADERS)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${source_file}")
        set(HETEROBRAIN_ENGINE_READY FALSE)
    endif()
endforeach()

if(HETEROBRAIN_ENGINE_READY)
    add_executable(heterobrain_engine ${ENGINE_SRCS} ${ENGINE_HEADERS})

    target_link_libraries(heterobrain_engine
        PRIVATE
            heterobrain_router
            snn_subsystem
            llm_subsystem
            bridge_subsystem
    )

    target_include_directories(heterobrain_engine
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
    )

    install(TARGETS heterobrain_engine
        RUNTIME DESTINATION bin
    )
else()
    message(STATUS
        "HeteroBrain engine sources are incomplete; building router core only")
endif()

# TODO: Phase 1 实现 main.cpp 的最小骨架:
#   1. 解析命令行: --llm <gguf> --config <yaml> --interactive
#   2. 加载 LLM (llama_runner)
#   3. 进入交互循环: getline → prompt_build → llama_infer → print
#   4. Phase 2 后增加: --snn <ckpt> 加载 SNN 子系统
