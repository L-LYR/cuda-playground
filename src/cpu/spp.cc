#include "cpu/spp.hh"
#include <boost/iterator/zip_iterator.hpp>
#include <boost/tuple/tuple.hpp>
#include <chrono>
#include <queue>
#include <spdlog/spdlog.h>

namespace pp {

auto NaiveSPP(const Graph &g, id_t vid) -> std::vector<uint64_t> {
  std::vector<uint64_t> dis(g.Vertices().size(), inf_dis);
  std::vector<bool> visited(g.Vertices().size(), false);

  auto min_begin =
      boost::zip_iterator(boost::make_tuple(dis.begin(), visited.begin()));
  auto min_end =
      boost::zip_iterator(boost::make_tuple(dis.end(), visited.end()));
  auto less = [&](const boost::tuple<uint64_t, bool> &l,
                  const boost::tuple<uint64_t, bool> &r) -> bool {
    auto l_dis = l.get<0>();
    auto r_dis = r.get<0>();
    auto l_visited = l.get<1>();
    auto r_visited = r.get<1>();

    return (l_visited == r_visited) ? (l_dis < r_dis) : not l_visited;
  };

  dis[vid] = 0;
  auto start = std::chrono::steady_clock::now();
  for (uint32_t i = 0; i < g.Vertices().size(); i++) {
    auto it = std::min_element(min_begin, min_end, less);
    auto cur_vid = it - min_begin;
    if (visited[cur_vid]) {
      break;
    }
    visited[cur_vid] = true;
    for (auto &oe : g.AdjacentMatrix()[cur_vid]) {
      auto cur_dis = dis[cur_vid] + oe.Distance();
      if (cur_dis < dis[oe.To()]) {
        dis[oe.To()] = cur_dis;
      }
    }
  }
  auto end = std::chrono::steady_clock::now();
  spdlog::info("{} ms", std::chrono::duration_cast<std::chrono::milliseconds>(
                            end - start)
                            .count());
  return dis;
}

auto BetterSPP(const Graph &g, id_t vid) -> std::vector<uint64_t> {
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