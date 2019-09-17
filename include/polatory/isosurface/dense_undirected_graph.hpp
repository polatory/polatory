// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <algorithm>
#include <stack>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include <polatory/common/exception.hpp>

namespace polatory {
namespace isosurface {

class dense_undirected_graph {
public:
  explicit dense_undirected_graph(int order)
    : m_(Eigen::MatrixXi::Zero(order, order)) {
    if (order <= 0) {
      throw common::invalid_argument("order > 0");
    }
  }

  void add_edge(int i, int j) {
    if (i > j) std::swap(i, j);
    m_(i, j)++;
  }

  int degree(int i) const {
    return m_.col(i).sum() + m_.row(i).sum() - m_(i, i);
  }

  bool has_edge(int i, int j) const {
    if (i > j) std::swap(i, j);
    return m_(i, j) != 0;
  }

  bool is_connected() const {
    if (order() == 0)
      return true;

    std::vector<bool> visited(order(), false);
    std::stack<int> to_visit;

    // DFS
    to_visit.push(0);
    while (!to_visit.empty()) {
      auto i = to_visit.top();
      to_visit.pop();

      visited[i] = true;

      for (auto j = 0; j < order(); j++) {
        if (has_edge(i, j) && !visited[j]) {
          to_visit.push(j);
        }
      }
    }

    return std::find(visited.begin(), visited.end(), false) == visited.end();
  }

  int order() const {
    return static_cast<int>(m_.rows());
  }

  int size() const {
    return m_.sum();
  }

private:
  Eigen::MatrixXi m_;
};

}  // namespace isosurface
}  // namespace polatory
