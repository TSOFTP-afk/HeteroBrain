#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace heterobrain {

using AgentId = std::uint32_t;

// The controller is the Story Director / global task objective. It is not a
// normal peer and is selected in the first stage of the ISA decision.
constexpr AgentId kControllerAgentId = 0;

enum class AgentRole {
    Controller,
    Planner,
    Writer,
    SnnMemory,
    VectorMemory,
    Rag,
    Critic,
    CanonChecker,
    StyleEditor,
    Verifier,
};

struct AgentDescriptor {
    AgentId id = 0;
    AgentRole role = AgentRole::Writer;
    std::string name;
    float relevance = 0.5F;
    float reliability = 0.5F;
    float freshness = 1.0F;
    float normalized_cost = 0.0F;
    bool enabled = true;
};

struct RouterConfig {
    // Strict ISA split: select the controller with alpha, otherwise select
    // exactly one peer with gamma = 1 - alpha.
    float controller_probability = 0.35F;

    // Engineering extension used only to choose one peer after the ISA
    // controller/peer decision has been made.
    float relevance_weight = 1.0F;
    float reliability_weight = 1.0F;
    float freshness_weight = 0.25F;
    float cost_weight = 0.25F;
    float temperature = 0.75F;
    float minimum_peer_weight = 1.0e-6F;

    std::uint64_t seed = 42;
};

struct RouteEdge {
    AgentId source = kControllerAgentId;
    AgentId target = 0;
    float selection_probability = 1.0F;
};

struct TemporalGraph {
    std::uint64_t round = 0;
    std::vector<RouteEdge> edges;
};

// IndecisiveRouter implements the topology part of ISA:
//   1. every target receives exactly one incoming information edge;
//   2. the source is either the controller or one peer;
//   3. the graph is resampled once per state-update round.
//
// Mandatory safety/canon gates deliberately live outside this stochastic
// graph. Irreversible actions must never depend on random routing alone.
class IndecisiveRouter {
public:
    explicit IndecisiveRouter(RouterConfig config);

    TemporalGraph resample(
        std::uint64_t round,
        const std::vector<AgentDescriptor>& targets,
        const std::vector<AgentDescriptor>& candidate_sources);

    const RouterConfig& config() const noexcept { return config_; }

private:
    RouteEdge sample_for_target(
        const AgentDescriptor& target,
        const std::vector<AgentDescriptor>& candidate_sources);

    RouterConfig config_;
    std::mt19937_64 generator_;
};

}  // namespace heterobrain
