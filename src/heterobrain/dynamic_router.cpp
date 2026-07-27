#include "dynamic_router.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace heterobrain {
namespace {

float clamp_probability(float value) {
    return std::clamp(value, 0.0F, 1.0F);
}

float safe_temperature(float value) {
    return std::max(value, 1.0e-4F);
}

}  // namespace

IndecisiveRouter::IndecisiveRouter(RouterConfig config)
    : config_(config), generator_(config.seed) {
    config_.controller_probability =
        clamp_probability(config_.controller_probability);
    config_.temperature = safe_temperature(config_.temperature);
    config_.minimum_peer_weight =
        std::max(config_.minimum_peer_weight, 1.0e-12F);
}

TemporalGraph IndecisiveRouter::resample(
    std::uint64_t round,
    const std::vector<AgentDescriptor>& targets,
    const std::vector<AgentDescriptor>& candidate_sources) {
    TemporalGraph graph;
    graph.round = round;
    graph.edges.reserve(targets.size());

    for (const auto& target : targets) {
        if (!target.enabled || target.id == kControllerAgentId) {
            continue;
        }
        graph.edges.push_back(sample_for_target(target, candidate_sources));
    }
    return graph;
}

RouteEdge IndecisiveRouter::sample_for_target(
    const AgentDescriptor& target,
    const std::vector<AgentDescriptor>& candidate_sources) {
    std::vector<const AgentDescriptor*> peers;
    std::vector<double> peer_weights;

    for (const auto& source : candidate_sources) {
        if (!source.enabled || source.id == kControllerAgentId ||
            source.id == target.id) {
            continue;
        }

        const float score =
            config_.relevance_weight * source.relevance +
            config_.reliability_weight * source.reliability +
            config_.freshness_weight * source.freshness -
            config_.cost_weight * source.normalized_cost;

        const double weight = std::max(
            static_cast<double>(config_.minimum_peer_weight),
            std::exp(static_cast<double>(score / config_.temperature)));
        peers.push_back(&source);
        peer_weights.push_back(weight);
    }

    // A target with no available peer must listen to the controller.
    if (peers.empty()) {
        return {kControllerAgentId, target.id, 1.0F};
    }

    std::bernoulli_distribution choose_controller(
        static_cast<double>(config_.controller_probability));
    if (choose_controller(generator_)) {
        return {
            kControllerAgentId,
            target.id,
            config_.controller_probability,
        };
    }

    std::discrete_distribution<std::size_t> choose_peer(
        peer_weights.begin(), peer_weights.end());
    const std::size_t selected = choose_peer(generator_);

    double total_weight = 0.0;
    for (double weight : peer_weights) {
        total_weight += weight;
    }
    if (!(total_weight > std::numeric_limits<double>::min())) {
        throw std::runtime_error("ISA peer weights are not normalizable");
    }

    const float conditional_probability = static_cast<float>(
        peer_weights[selected] / total_weight);
    const float probability =
        (1.0F - config_.controller_probability) * conditional_probability;

    return {peers[selected]->id, target.id, probability};
}

}  // namespace heterobrain
