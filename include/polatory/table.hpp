// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <fstream>
#include <string>
#include <vector>

#include <boost/algorithm/string.hpp>
#include <Eigen/Core>

#include <polatory/common/eigen_utility.hpp>
#include <polatory/common/exception.hpp>
#include <polatory/numeric/roundtrip_string.hpp>

namespace polatory {

using tabled = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

inline auto read_table(const std::string& filename) {
  tabled table;

  std::ifstream ifs(filename);
  if (!ifs)
    throw common::io_error("Could not open file '" + filename + "'.");

  std::vector<double> buffer;

  std::string line;
  size_t n_cols = 0;
  while (std::getline(ifs, line)) {
    if (boost::starts_with(line, "#"))
      continue;

    std::vector<std::string> row;
    boost::split(row, line, boost::is_any_of(" \t,"));
    if (n_cols == 0) {
      n_cols = row.size();
    } else if (row.size() != n_cols) {
      continue;
    }

    for (const auto& cell : row) {
      // On Unix platforms, std::getline() keeps the \r if the line ends with \r\n.
      // Use boost::trim_copy() to remove it.
      buffer.push_back(numeric::to_double(boost::trim_copy(cell)));
    }
  }

  table = tabled::Map(buffer.data(), buffer.size() / n_cols, n_cols);

  return table;
}

template <class Derived>
inline void write_table(const std::string& filename,
                        const Eigen::MatrixBase<Derived>& table) {
  std::ofstream ofs(filename);
  if (!ofs)
    throw common::io_error("Could not open file '" + filename + "'.");

  size_t n_cols = table.cols();
  for (auto row : common::row_range(table)) {
    for (size_t i = 0; i < n_cols; i++) {
      ofs << numeric::to_string(row(i));
      if (i != n_cols - 1)
        ofs << ' ';
    }
    ofs << std::endl;
  }
}

}  // namespace polatory
