// =============================================================================
// EventScheduler 单元测试 (无 CUDA 依赖, 纯 host 逻辑)
// 测试 load_jsonl 解析正确性 + apply_modifiers 计算正确性
//
// 注意: dispatch_pending 调用 set_event_signal (定义在 modulatory_kernels.cu),
//       本测试不链接 CUDA 库, 因此用 stub 替代 set_event_signal 实现,
//       仅验证调度逻辑 (查表+apply_modifiers+派发顺序) 正确性。
// =============================================================================
#include "event_scheduler.h"
#include "gene_event_map.h"

#include <cassert>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <string>

using namespace stage2e;

// set_event_signal stub: 记录最后一次调用的参数, 供测试断言
static float g_last_delta[6] = {0};
static int   g_last_duration = -1;
static int   g_call_count = 0;

// 覆盖 modulatory_kernels.cu 中的实现 (链接时本 weak 实现优先)
// 注意: 由于 test_event_scheduler 不链接 modulatory_kernels.cu, 此处直接定义即可
namespace stage2e {
void set_event_signal(const float modulator_delta[6], int duration_steps) {
    for (int i = 0; i < 6; ++i) g_last_delta[i] = modulator_delta[i];
    g_last_duration = duration_steps;
    g_call_count++;
}
}

static int g_test_pass = 0;
static int g_test_fail = 0;

#define TEST(cond, msg) do { \
    if (cond) { g_test_pass++; } \
    else { g_test_fail++; fprintf(stderr, "FAIL: %s\n", msg); } \
} while(0)

// 测试 1: apply_modifiers intensity 调制
void test_apply_modifiers_intensity() {
    GeneMapEntry base = GENE_MAP_BASE[EVT_FOOD_TASTY];  // DA=0.40
    // intensity=0: 无变化
    GeneMapEntry r0 = apply_modifiers(base, 0, 0);
    TEST(fabsf(r0.da_delta - 0.40f) < 1e-5f, "intensity=0 DA unchanged");
    // intensity=50: scale = 1.0 + 50*0.02 = 2.0 → DA=0.80
    GeneMapEntry r50 = apply_modifiers(base, 0, 50);
    TEST(fabsf(r50.da_delta - 0.80f) < 1e-5f, "intensity=50 DA doubled");
    // intensity=-50: scale = max(0.05, 1.0-1.0) = max(0.05, 0) = 0.05 → DA=0.02
    GeneMapEntry rn50 = apply_modifiers(base, 0, -50);
    TEST(fabsf(rn50.da_delta - 0.02f) < 1e-5f, "intensity=-50 DA floored");
}

// 测试 2: apply_modifiers 修饰符
void test_apply_modifiers_flags() {
    GeneMapEntry base = GENE_MAP_BASE[EVT_PRAISE];  // Oxy=0.20, NE=0.15, DA=0.25, 5HT=-0.05
    // MOD_PUBLIC: Oxy×1.5, NE×1.2
    GeneMapEntry rp = apply_modifiers(base, MOD_PUBLIC, 0);
    TEST(fabsf(rp.oxy_delta - 0.30f) < 1e-5f, "MOD_PUBLIC Oxy×1.5");
    TEST(fabsf(rp.ne_delta - 0.18f) < 1e-5f, "MOD_PUBLIC NE×1.2");
    // MOD_AUTHORITY: DA×1.3, 5HT×1.2
    GeneMapEntry ra = apply_modifiers(base, MOD_AUTHORITY, 0);
    TEST(fabsf(ra.da_delta - 0.325f) < 1e-5f, "MOD_AUTHORITY DA×1.3");
    TEST(fabsf(ra.ht5_delta - (-0.06f)) < 1e-5f, "MOD_AUTHORITY 5HT×1.2");
    // MOD_SUSTAINED: duration×3
    GeneMapEntry rs = apply_modifiers(base, MOD_SUSTAINED, 0);
    TEST(fabsf(rs.duration_s - 3.0f) < 1e-5f, "MOD_SUSTAINED duration×3");
}

// 测试 3: load_jsonl 解析
void test_load_jsonl() {
    // 写临时 JSONL 文件
    std::string tmp_path = "test_events_tmp.jsonl";
    {
        std::ofstream fout(tmp_path);
        fout << "{\"event_id\":1,\"step_target\":500,\"event_type\":\"food_tasty\",\"intensity\":30,\"description\":\"chocolate\"}\n";
        fout << "{\"event_id\":2,\"step_target\":1000,\"event_type\":\"praise\",\"modifiers\":{\"publicity\":\"public\",\"authority\":\"authority\"},\"intensity\":20,\"description\":\"boss praise\"}\n";
        fout << "# comment line\n";
        fout << "\n";
        fout << "{\"event_id\":3,\"time_s\":15.0,\"event_type\":\"threat_physical\",\"intensity\":-30}\n";
    }

    EventScheduler sched;
    bool ok = sched.load_jsonl(tmp_path);
    TEST(ok, "load_jsonl success");
    TEST(sched.total_events() == 3, "3 events loaded");

    const auto& events = sched.events();
    // 事件 1
    TEST(events[0].step_target == 500, "evt1 step_target=500");
    TEST(events[0].event_type == EVT_FOOD_TASTY, "evt1 type=FOOD_TASTY");
    TEST(events[0].intensity == 30, "evt1 intensity=30");
    TEST(events[0].description == "chocolate", "evt1 desc");
    // 事件 2 (modifiers)
    TEST(events[1].step_target == 1000, "evt2 step_target=1000");
    TEST(events[1].event_type == EVT_PRAISE, "evt2 type=PRAISE");
    TEST((events[1].modifier_flags & MOD_PUBLIC) != 0, "evt2 MOD_PUBLIC");
    TEST((events[1].modifier_flags & MOD_AUTHORITY) != 0, "evt2 MOD_AUTHORITY");
    // 事件 3 (time_s → step_target: 15.0 * 100 = 1500)
    TEST(events[2].step_target == 1500, "evt3 step_target from time_s");
    TEST(events[2].event_type == EVT_THREAT_PHYSICAL, "evt3 type=THREAT_PHYSICAL");

    // 清理
    std::remove(tmp_path.c_str());
}

// 测试 4: event_type_from_string 边界
void test_event_type_from_string() {
    TEST(event_type_from_string("food_tasty") == EVT_FOOD_TASTY, "string->FOOD_TASTY");
    TEST(event_type_from_string("novelty") == EVT_NOVELTY, "string->NOVELTY");
    TEST(event_type_from_string("unknown") == EVT_COUNT, "unknown->EVT_COUNT");
    TEST(event_type_from_string(nullptr) == EVT_COUNT, "null->EVT_COUNT");
}

// 测试 5: dispatch_pending 调度逻辑 (验证 set_event_signal 被正确调用)
void test_dispatch_pending() {
    std::string tmp_path = "test_dispatch_tmp.jsonl";
    {
        std::ofstream fout(tmp_path);
        // food_tasty intensity=30 → scale=1.6 → DA=0.40*1.6=0.64
        fout << "{\"event_id\":1,\"step_target\":500,\"event_type\":\"food_tasty\",\"intensity\":30}\n";
        // threat_physical intensity=0 → 5HT=0.40
        fout << "{\"event_id\":2,\"step_target\":1000,\"event_type\":\"threat_physical\",\"intensity\":0}\n";
    }

    EventScheduler sched;
    bool ok = sched.load_jsonl(tmp_path);
    TEST(ok, "dispatch test load_jsonl success");

    // 在 step=500 之前派发: 无事件
    g_call_count = 0;
    sched.dispatch_pending(499);
    TEST(g_call_count == 0, "no dispatch before step 500");
    TEST(sched.dispatched_count() == 0, "dispatched_count=0 at step 499");

    // 在 step=500 派发: 1 个事件
    sched.dispatch_pending(500);
    TEST(g_call_count == 1, "1 event dispatched at step 500");
    TEST(sched.dispatched_count() == 1, "dispatched_count=1 at step 500");
    // food_tasty intensity=30: scale=1.0+30*0.02=1.6, DA=0.40*1.6=0.64
    TEST(fabsf(g_last_delta[0] - 0.64f) < 1e-5f, "food_tasty DA delta=0.64");
    // 5HT: -0.05*1.6 = -0.08
    TEST(fabsf(g_last_delta[3] - (-0.08f)) < 1e-5f, "food_tasty 5HT delta=-0.08");
    // duration_steps=0 (pulse 型)
    TEST(g_last_duration == 0, "duration_steps=0 (pulse)");

    // 在 step=1500 派发: 第 2 个事件 (step_target=1000 已过期)
    sched.dispatch_pending(1500);
    TEST(g_call_count == 2, "2 events dispatched at step 1500");
    TEST(sched.dispatched_count() == 2, "dispatched_count=2 at step 1500");
    // threat_physical intensity=0: 5HT=0.40 (无缩放)
    TEST(fabsf(g_last_delta[3] - 0.40f) < 1e-5f, "threat_physical 5HT delta=0.40");

    // 再派发: 无更多事件
    sched.dispatch_pending(9999);
    TEST(g_call_count == 2, "no more events at step 9999");

    std::remove(tmp_path.c_str());
}

int main() {
    fprintf(stdout, "[test_event_scheduler] running...\n");
    test_apply_modifiers_intensity();
    test_apply_modifiers_flags();
    test_load_jsonl();
    test_event_type_from_string();
    test_dispatch_pending();
    fprintf(stdout, "[test_event_scheduler] PASS=%d FAIL=%d\n", g_test_pass, g_test_fail);
    return g_test_fail == 0 ? 0 : 1;
}
