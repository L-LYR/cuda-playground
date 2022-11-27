#define USE_CUDA
#include "graph.hh"
#include <chrono>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <spdlog/spdlog.h>
#include <thrust/functional.h>

namespace pp {

__global__ auto Relax(const OutEdge *adj_list, uint64_t adj_list_len,
                      uint64_t min_dis, uint64_t *cur_dis, bool *visited)
    -> void {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < adj_list_len) {
    auto to = adj_list[tid].To();
    auto dis = adj_list[tid].Distance();
    if (not visited[to] and dis != inf_dis) {
      cur_dis[to] = thrust::min(cur_dis[to], min_dis + dis);
    }
  }
}

auto ParallalSPP(const Graph &g, id_t vid) -> thrust::host_vector<uint64_t> {
  thrust::device_vector<uint64_t> d_dis(g.Vertices().size(), inf_dis);
  auto d_dis_p = thrust::raw_pointer_cast(d_dis.data());

  thrust::device_vector<bool> d_visited(g.Vertices().size(), false);
  auto d_visited_p = thrust::raw_pointer_cast(d_visited.data());

  const int threads_per_block = 128;
  const dim3 block_dim(threads_per_block, 1, 1);
  const dim3 grid_dim(
      (g.Vertices().size() + threads_per_block - 1) / threads_per_block, 1, 1);

  auto min_begin = thrust::make_zip_iterator(d_dis.begin(), d_visited.begin());
  auto min_end = thrust::make_zip_iterator(d_dis.end(), d_visited.end());
  auto less = [] __device__(thrust::tuple<uint64_t, bool> l,
                            thrust::tuple<uint64_t, bool> r) -> bool {
    auto l_dis = l.get<0>();
    auto r_dis = r.get<0>();
    auto l_visited = l.get<1>();
    auto r_visited = r.get<1>();

    return (l_visited == r_visited) ? (l_dis < r_dis) : not l_visited;
  };

  d_dis[0] = 0;

  auto start = std::chrono::steady_clock::now();
  for (uint64_t i = 0; i < g.Vertices().size(); i++) {
    auto it = thrust::min_element(thrust::device, min_begin, min_end, less);
    auto min_dis = *it.get_iterator_tuple().get<0>();
    auto cur_vid = it - min_begin;
    if (d_visited[cur_vid]) {
      break;
    }
    d_visited[cur_vid] = true;
    auto d_adj_list = g.AdjacentList(cur_vid);
    Relax<<<grid_dim, block_dim>>>(d_adj_list.get<0>(), d_adj_list.get<1>(),
                                   min_dis, d_dis_p, d_visited_p);
  }
  auto end = std::chrono::steady_clock::now();
  spdlog::info("{} ms", std::chrono::duration_cast<std::chrono::milliseconds>(
                            end - start)
                            .count());

  return d_dis;
}

} // namespace pp

#undef USE_CUDA