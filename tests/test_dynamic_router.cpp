#include "dynamic_router.h"

#include <cassert>
#include <cstddef>
#include <set>
#include <vector>

using heterobrain::AgentDescriptor;
using heterobrain::AgentRole;
using heterobrain::IndecisiveRouter;
using heterobrain::RouterConfig;
using heterobrain::kControllerAgentId;

int main() {
    RouterConfig config;
    config.controller_probability = 0.35F;
    config.seed = 7;

    std::vector<AgentDescriptor> targets = {
        {10, AgentRole::Writer, "writer"},
        {11, AgentRole::Critic, "critic"},
    };
    std::vector<AgentDescriptor> sources = {
        {10, AgentRole::Writer, "writer"},
        {11, AgentRole::Critic, "critic"},
        {12, AgentRole::SnnMemory, "snn-memory", 0.9F, 0.7F},
        {13, AgentRole::VectorMemory, "vector-memory", 0.8F, 0.9F},
    };

    IndecisiveRouter router(config);
    bool saw_controller = false;
    bool saw_peer = false;

    for (std::uint64_t round = 0; round < 256; ++round) {
        const auto graph = router.resample(round, targets, sources);
        assert(graph.round == round);
        assert(graph.edges.size() == targets.size());

        std::set<heterobrain::AgentId> routed_targets;
        for (const auto& edge : graph.edges) {
            assert(edge.source != edge.target);
            assert(edge.selection_probability > 0.0F);
            assert(edge.selection_probability <= 1.0F);
            assert(routed_targets.insert(edge.target).second);

            saw_controller |= edge.source == kControllerAgentId;
            saw_peer |= edge.source != kControllerAgentId;
        }
    }

    assert(saw_controller);
    assert(saw_peer);

    // With no peer, routing must safely fall back to the controller.
    const auto fallback = router.resample(999, {targets.front()}, {});
    assert(fallback.edges.size() == 1);
    assert(fallback.edges.front().source == kControllerAgentId);
    assert(fallback.edges.front().selection_probability == 1.0F);
    return 0;
}
