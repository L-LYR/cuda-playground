#include "cpu/spp.hh"
#include <chrono>
#include <queue>
#include <spdlog/spdlog.h>

namespace pp {

auto NaiveSPP(const Graph &g, id_t vid) -> std::vector<uint64_t> {
  std::vector<uint64_t> dis(g.Vertices().size(), inf_dis);
  std::vector<bool> visited(g.Vertices().size(), false);

  auto less = [&dis](const id_t &l, const id_t &r) -> bool {
    return dis[l] > dis[r];
  };
  std::priority_queue<id_t, std::vector<id_t>, decltype(less)> next(less);

  dis[vid] = 0;
  next.push(vid);
  auto start = std::chrono::steady_clock::now();
  while (not next.empty()) {
    auto cur_vid = next.top();
    next.pop();
    if (visited[cur_vid]) {
      continue;
    }
    visited[cur_vid] = true;
    for (auto &oe : g.AdjacentMatrix()[cur_vid]) {
      auto cur_dis = dis[cur_vid] + oe.Distance();
      if (cur_dis < dis[oe.To()]) {
        dis[oe.To()] = cur_dis;
        next.push(oe.To());
      }
    }
  }
  auto end = std::chrono::steady_clock::now();
  spdlog::info("{} ms", std::chrono::duration_cast<std::chrono::milliseconds>(
                            end - start)
                            .count());
  return dis;
}

} // namespace pp