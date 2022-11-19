#include "graph.hh"
#include <set>

namespace pp {

auto ReadGraphFile(std::string filename)
    -> std::pair<std::vector<Edge>, std::vector<Vertex>> {
  std::ifstream fin(filename);
  char op;
  std::string dummy;
  id_t from;
  id_t to;
  uint64_t weight;

  std::vector<Edge> es;
  std::vector<Vertex> vs;
  std::set<id_t> v_set;
  uint32_t n_edge;
  uint32_t n_vertex;

  while (not fin.eof()) {
    fin >> op;
    if (op == 'c') {
      getline(fin, dummy);
    } else if (op == 'p') {
      fin >> dummy >> n_vertex >> n_edge;
      es.reserve(n_edge);
      vs.reserve(n_vertex);
    } else if (op == 'a') {
      fin >> from >> to >> weight;
      from--;
      to--;
      es.emplace_back(from, to, weight);
      v_set.insert(from);
      v_set.insert(to);
    }
  }

  vs.assign(v_set.begin(), v_set.end());
  return {es, vs};
}

} // namespace pp