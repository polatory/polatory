#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <polatory/types.hpp>
#include <stack>
#include <stdexcept>
#include <utility>
#include <vector>

namespace polatory::isosurface {

class dense_undirected_graph {
 public:
  explicit dense_undirected_graph(index_t order) : m_(Eigen::MatrixXi::Zero(order, order)) {
    if (order <= 0) {
      throw std::invalid_argument("order must be greater than 0.");
    }
  }

  void add_edge(index_t i, index_t j) {
    if (i > j) {
      std::swap(i, j);
    }
    m_(i, j)++;
  }

  index_t degree(index_t i) const { return m_.col(i).sum() + m_.row(i).sum() - m_(i, i); }

  bool has_edge(index_t i, index_t j) const {
    if (i > j) {
      std::swap(i, j);
    }
    return m_(i, j) != 0;
  }

  // Returns true if the graph is a cycle or a path.
  // The singleton graph is not considered as a cycle or a path.
  // Precondition: the graph is simple.
  bool is_cycle_or_path() const {
    std::vector<bool> visited(order());
    std::stack<index_t> to_visit;

    // DFS
    to_visit.push(0);
    while (!to_visit.empty()) {
      auto i = to_visit.top();
      to_visit.pop();

      if (!(degree(i) == 1 || degree(i) == 2)) {
        return false;
      }

      visited.at(i) = true;

      for (index_t j = 0; j < order(); j++) {
        if (has_edge(i, j) && !visited.at(j)) {
          to_visit.push(j);
        }
      }
    }

    return std::find(visited.begin(), visited.end(), false) == visited.end();
  }

  bool is_simple() const {
    for (index_t i = 0; i < order(); i++) {
      if (m_(i, i) != 0) {
        return false;
      }

      for (index_t j = i + 1; j < order(); j++) {
        if (m_(i, j) > 1) {
          return false;
        }
      }
    }

    return true;
  }

  int order() const { return static_cast<int>(m_.rows()); }

 private:
  Eigen::MatrixXi m_;
};

}  // namespace polatory::isosurface
