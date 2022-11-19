#include "cpu/spp.hh"

#include <fmt/format.h>
#include <gtest/gtest.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/property_map/property_map.hpp>

TEST(BoostSPP, SmallDataSetTest) {
  auto [es, vs] = pp::ReadGraphFile("./data/USA-road-d.NY.gr");

  using VertexProperty = boost::property<boost::vertex_distance_t, int>;
  using EdgeProperty = boost::property<boost::edge_weight_t, int>;
  using Graph =
      boost::adjacency_list<boost::listS, boost::vecS, boost::directedS,
                            VertexProperty, EdgeProperty>;
  Graph g;

  for (auto &e : es) {
    boost::add_edge(e.From(), e.To(), e.Distance(), g);
  }

  std::vector<uint64_t> res(vs.size(), 0);
  boost::dijkstra_shortest_paths(g, boost::vertex(0, g),
                                 boost::distance_map(res.data()));

  spdlog::info("{}\n", fmt::join(res, ", "));
}

auto main(int argc, char *argv[]) -> int {
  auto file_logger = spdlog::basic_logger_mt("boost", "boost.log", true);
  spdlog::set_default_logger(file_logger);
  spdlog::set_level(spdlog::level::info);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}