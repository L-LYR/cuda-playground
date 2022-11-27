#pragma once

#include "graph.hh"

namespace pp {

// Shortest Path Problem
// Naive & Serialized
auto NaiveSPP(const Graph &g, id_t vid) -> std::vector<uint64_t>;
// Queue-based
auto BetterSPP(const Graph &g, id_t vid) -> std::vector<uint64_t>;

}; // namespace pp