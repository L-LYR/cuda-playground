#pragma once

#include "graph.hh"

namespace pp {

// Shortest Path Problem
// Naive & Serialized
// Queue-based
auto NaiveSPP(const Graph &g, id_t vid) -> std::vector<uint64_t>;

}; // namespace cpu