#include "gpu/spp.cuh"

#include <fmt/format.h>
#include <gtest/gtest.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>

TEST(ParallalSPP, CorrectnessTest) {
  std::vector<pp::Edge> es = {
      {0, 2, 5}, {0, 4, 1}, {1, 3, 1}, {2, 3, 4}, {4, 1, 2}, {4, 2, 3},
  };

  std::vector<pp::Vertex> vs = {0, 1, 2, 3, 4};

  pp::Graph g(es, vs);

  auto res = pp::ParallalSPP(g, 0);

  spdlog::info("{}\n", fmt::join(res, ", "));
}

TEST(ParallalSPP, SmallDataSetTest) {
  auto [es, vs] = pp::ReadGraphFile("./data/USA-road-d.NY.gr");

  pp::Graph g(es, vs);

  auto res = pp::ParallalSPP(g, 0);

  spdlog::info("{}\n", fmt::join(res, ", "));
}

auto main(int argc, char *argv[]) -> int {
  auto file_logger = spdlog::basic_logger_mt("gpu", "gpu.log", true);
  spdlog::set_default_logger(file_logger);
  spdlog::set_level(spdlog::level::info);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
