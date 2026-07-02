#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <polatory/types.hpp>
#include <stack>
#include <stdexcept>
#include <utility>
#include <vector>

namespace polatory::isosurface {

// An undirected multigraph over vertices [0, order).
class DenseUndirectedGraph {
  using Matrix = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

 public:
  explicit DenseUndirectedGraph(Index order) : m_(Matrix::Zero(order, order)) {
    if (order <= 0) {
      throw std::invalid_argument("order must be positive");
    }
  }

  void add_edge(Index i, Index j) {
    if (i > j) {
      std::swap(i, j);
    }
    m_(i, j)++;
  }

  // The vertices of each connected component.
  std::vector<std::vector<Index>> connected_components() const {
    std::vector<Index> component(order(), -1);
    Index count = 0;
    for (Index s = 0; s < order(); s++) {
      if (component.at(s) >= 0) {
        continue;
      }
      std::stack<Index> to_visit;
      to_visit.push(s);
      component.at(s) = count;
      while (!to_visit.empty()) {
        auto i = to_visit.top();
        to_visit.pop();
        for (Index j = 0; j < order(); j++) {
          if (has_edge(i, j) && component.at(j) < 0) {
            component.at(j) = count;
            to_visit.push(j);
          }
        }
      }
      count++;
    }
    std::vector<std::vector<Index>> result(count);
    for (Index i = 0; i < order(); i++) {
      result.at(component.at(i)).push_back(i);
    }
    return result;
  }

  bool is_connected() const {
    std::vector<bool> visited(order());
    std::stack<Index> to_visit;

    // DFS
    to_visit.push(0);
    while (!to_visit.empty()) {
      auto i = to_visit.top();
      to_visit.pop();

      visited.at(i) = true;

      for (Index j = 0; j < order(); j++) {
        if (has_edge(i, j) && !visited.at(j)) {
          to_visit.push(j);
        }
      }
    }

    return std::find(visited.begin(), visited.end(), false) == visited.end();
  }

  bool is_simple() const {
    for (Index i = 0; i < order(); i++) {
      if (m_(i, i) != 0) {
        return false;
      }

      for (Index j = i + 1; j < order(); j++) {
        if (m_(i, j) > 1) {
          return false;
        }
      }
    }

    return true;
  }

  Index max_degree() const {
    Index max_degree{};
    for (Index i = 0; i < order(); i++) {
      max_degree = std::max(max_degree, degree(i));
    }
    return max_degree;
  }

  Index min_degree() const {
    Index min_degree = degree(0);
    for (Index i = 1; i < order(); i++) {
      min_degree = std::min(min_degree, degree(i));
    }
    return min_degree;
  }

  Index order() const { return m_.rows(); }

 private:
  Index degree(Index i) const {
    return m_.col(i).template cast<Index>().sum() + m_.row(i).template cast<Index>().sum() -
           m_(i, i);
  }

  bool has_edge(Index i, Index j) const {
    if (i > j) {
      std::swap(i, j);
    }
    return m_(i, j) != 0;
  }

  Matrix m_;
};

}  // namespace polatory::isosurface
