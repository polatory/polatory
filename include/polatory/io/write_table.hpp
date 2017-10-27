// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <fstream>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "polatory/numeric/to_string.hpp"

namespace polatory {
namespace io {

inline void write_table(const std::string& filename,
                        const std::vector<std::vector<double>>& table) {
  std::ofstream ofs(filename);
  if (!ofs)
    return;

  for (auto& row : table) {
    for (size_t i = 0; i < row.size(); i++) {
      ofs << numeric::to_string(row[i]);
      if (i != row.size() - 1)
        ofs << ' ';
    }
    ofs << '\n';
  }
}

inline void write_points(const std::string& filename,
                         const std::vector<Eigen::Vector3d>& points) {
  std::vector<std::vector<double>> table;

  for (auto& p : points) {
    table.push_back(std::vector<double>{ p(0), p(1), p(2) });
  }

  write_table(filename, table);
}

inline void write_points_and_values(const std::string& filename,
                                    const std::vector<Eigen::Vector3d>& points,
                                    const Eigen::VectorXd& values) {

  std::vector<std::vector<double>> table;

  for (size_t i = 0; i < values.size(); i++) {
    auto& p = points[i];
    table.push_back(std::vector<double>{ p(0), p(1), p(2), values(i) });
  }

  write_table(filename, table);
}

inline void write_values(const std::string& filename,
                         const Eigen::VectorXd& values) {
  std::vector<std::vector<double>> table;

  for (size_t i = 0; i < values.size(); i++) {
    table.push_back(std::vector<double>{ values(i) });
  }

  write_table(filename, table);
}

} // namespace io
} // namespace polatory
