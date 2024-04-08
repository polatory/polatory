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
  using Matrix = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

 public:
  explicit dense_undirected_graph(index_t order) : m_(Matrix::Zero(order, order)) {
    if (order <= 0) {
      throw std::invalid_argument("order must be positive");
    }
  }

  void add_edge(index_t i, index_t j) {
    if (i > j) {
      std::swap(i, j);
    }
    m_(i, j)++;
  }

  index_t degree(index_t i) const {
    return m_.col(i).cast<index_t>().sum() + m_.row(i).cast<index_t>().sum() - m_(i, i);
  }

  bool has_edge(index_t i, index_t j) const {
    if (i > j) {
      std::swap(i, j);
    }
    return m_(i, j) != 0;
  }

  bool is_connected() const {
    std::vector<bool> visited(order());
    std::stack<index_t> to_visit;

    // DFS
    to_visit.push(0);
    while (!to_visit.empty()) {
      auto i = to_visit.top();
      to_visit.pop();

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

  index_t max_degree() const {
    index_t max_degree{};
    for (index_t i = 0; i < order(); i++) {
      max_degree = std::max(max_degree, degree(i));
    }
    return max_degree;
  }

  index_t order() const { return m_.rows(); }

 private:
  Matrix m_;
};

}  // namespace polatory::isosurface
