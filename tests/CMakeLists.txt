# =============================================================================
# vita — 测试占位
# =============================================================================
# 说明: 本目录不含测试源文件。
# 实际测试目标 test_event_scheduler / test_embodied 定义于 src/snn/CMakeLists.txt
# (add_test COMMAND 中直接引用 target 名, 构建时自动解析为可执行文件路径)。
# 此处保留占位以兼容顶层 add_subdirectory(tests)。

add_test(NAME test_event_scheduler COMMAND test_event_scheduler)
add_test(NAME test_embodied COMMAND test_embodied)
add_test(NAME test_emotion_bridge COMMAND test_emotion_bridge)
