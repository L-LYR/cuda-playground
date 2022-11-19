#pragma once

#define USE_CUDA
#include "graph.hh"

namespace pp {

auto ParallalSPP(const Graph &g, id_t vid) -> thrust::host_vector<uint64_t>;

}

#undef USE_CUDA