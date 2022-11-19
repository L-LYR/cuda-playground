#pragma once

#include <cstdint>
#include <fstream>
#include <string>

#if defined(USE_CUDA)
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#endif

#include <vector>

namespace pp {

using id_t = uint32_t;
constexpr const uint64_t inf_dis = 0x3f3f3f3f3f3f3f3f;

class Vertex {
public:
  Vertex(id_t id) : id_(id) {}

public:
  auto ID() const -> id_t { return id_; }

private:
  id_t id_;
};

class Edge {
public:
  Edge(id_t from, id_t to, uint64_t dis) : from_(from), to_(to), dis_(dis) {}

public:
  auto From() const -> id_t { return from_; }
  auto To() const -> id_t { return to_; }
  auto Distance() const -> uint64_t { return dis_; }

private:
  id_t from_;
  id_t to_;
  uint64_t dis_;
};

class OutEdge {
public:
  OutEdge(id_t to, uint64_t dis) : to_(to), dis_(dis) {}

public:
#if defined(USE_CUDA)
  __host__ __device__ auto To() const -> id_t { return to_; }
  __host__ __device__ auto Distance() const -> id_t { return dis_; }
#else
  auto To() const -> id_t { return to_; }
  auto Distance() const -> id_t { return dis_; }
#endif

private:
  id_t to_;
  uint64_t dis_;
};

// direct graph by default
class Graph {
public:
  Graph(const std::vector<Edge> &edges, const std::vector<Vertex> &vertices)
      : edges_(edges), vertices_(vertices) {
#if defined(USE_CUDA)
    std::vector<std::vector<OutEdge>> temp_adj_mat;
    temp_adj_mat.resize(vertices_.size());
    for (auto &e : edges_) {
      temp_adj_mat[e.From()].emplace_back(e.To(), e.Distance());
    }

    adjacent_matrix_.reserve(edges_.size());
    adjacent_list_length_.reserve(vertices_.size());
    adjacent_list_start_.reserve(vertices_.size());
    for (auto &adj_list : temp_adj_mat) {
      adjacent_list_start_.push_back(adjacent_matrix_.size());
      adjacent_list_length_.push_back(adj_list.size());
      adjacent_matrix_.insert(adjacent_matrix_.end(), adj_list.begin(),
                              adj_list.end());
    }
#else
    adjacent_matrix_.resize(vertices_.size());
    for (const auto &e : edges_) {
      adjacent_matrix_[e.From()].emplace_back(e.To(), e.Distance());
    }
#endif
  }

public:
#if defined(USE_CUDA)
  auto AdjacentList(id_t vid) const -> thrust::tuple<const OutEdge *, size_t> {
    auto start = adjacent_list_start_[vid];
    auto length = adjacent_list_length_[vid];
    return {
        thrust::raw_pointer_cast(&adjacent_matrix_[start]),
        length,
    };
  }
#else
  auto AdjacentMatrix() const -> const std::vector<std::vector<OutEdge>> & {
    return adjacent_matrix_;
  }
#endif

  auto Edges() const -> const std::vector<Edge> & { return edges_; }
  auto Vertices() const -> const std::vector<Vertex> & { return vertices_; }

private:
#if defined(USE_CUDA)
  thrust::device_vector<OutEdge> adjacent_matrix_;
  thrust::device_vector<size_t> adjacent_list_start_;
  thrust::device_vector<size_t> adjacent_list_length_;
#else
  std::vector<std::vector<OutEdge>> adjacent_matrix_;
#endif

  std::vector<Edge> edges_;
  std::vector<Vertex> vertices_;
};

auto ReadGraphFile(std::string filename)
    -> std::pair<std::vector<Edge>, std::vector<Vertex>>;

} // namespace pp