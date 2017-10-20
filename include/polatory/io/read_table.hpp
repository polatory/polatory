// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <boost/algorithm/string.hpp>

#include <Eigen/Core>

namespace polatory {
namespace io {

inline std::vector<std::vector<double>> read_table(const std::string& filename) {
  std::vector<std::vector<double>> table;

  std::ifstream ifs(filename);
  if (!ifs)
    return table;

  std::string line;
  while (std::getline(ifs, line)) {
    if (boost::starts_with(line, "#"))
      continue;

    std::vector<std::string> row;
    boost::split(row, line, boost::is_any_of(" \t,"));

    std::vector<double> row_values;
    for (const auto& cell : row) {
      row_values.push_back(std::stod(cell));
    }
    table.push_back(row_values);
  }

  return table;
}

inline std::vector<Eigen::Vector3d> read_points(const std::string& filename) {
  auto table = read_table(filename);
  auto n_rows = table.size();

  std::vector<Eigen::Vector3d> points(n_rows);

  for (size_t i = 0; i < n_rows; i++) {
    const auto& row = table[i];
    points[i] = Eigen::Vector3d(row[0], row[1], row[2]);
  }

  return points;
}

inline std::pair<std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector3d>> read_points_and_normals(const std::string& filename) {
  auto table = read_table(filename);
  auto n_rows = table.size();

  std::vector<Eigen::Vector3d> points(n_rows);
  std::vector<Eigen::Vector3d> normals(n_rows);
  Eigen::VectorXd values = Eigen::VectorXd(n_rows);

  for (size_t i = 0; i < n_rows; i++) {
    const auto& row = table[i];
    points[i] = Eigen::Vector3d(row[0], row[1], row[2]);
    normals[i] = Eigen::Vector3d(row[3], row[4], row[5]);
  }

  return std::make_pair(std::move(points), std::move(normals));
}

inline std::pair<std::vector<Eigen::Vector3d>, Eigen::VectorXd> read_points_and_values(const std::string& filename) {
  auto table = read_table(filename);
  auto n_rows = table.size();

  std::vector<Eigen::Vector3d> points(n_rows);
  Eigen::VectorXd values = Eigen::VectorXd(n_rows);

  for (size_t i = 0; i < n_rows; i++) {
    const auto& row = table[i];
    points[i] = Eigen::Vector3d(row[0], row[1], row[2]);
    values(i) = row[3];
  }

  return std::make_pair(std::move(points), std::move(values));
}

inline Eigen::VectorXd read_values(const std::string& filename) {
  auto table = read_table(filename);
  auto n_rows = table.size();

  Eigen::VectorXd values = Eigen::VectorXd(n_rows);

  for (size_t i = 0; i < n_rows; i++) {
    const auto& row = table[i];
    values(i) = row[0];
  }

  return values;
}

} // namespace io
} // namespace polatory
