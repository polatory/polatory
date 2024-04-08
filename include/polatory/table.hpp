#pragma once

#include <Eigen/Core>
#include <boost/algorithm/string.hpp>
#include <format>
#include <fstream>
#include <iostream>
#include <polatory/numeric/conv.hpp>
#include <polatory/types.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace polatory {

inline matrixd read_table(const std::string& filename, const char* delimiters = " \t,") {
  std::ifstream ifs(filename);
  if (!ifs) {
    throw std::runtime_error(std::format("cannot open file '{}'", filename));
  }

  std::vector<double> buffer;

  std::string line;
  auto n_cols = index_t{0};
  auto line_no = 0;
  while (std::getline(ifs, line)) {
    line_no++;

    if (boost::starts_with(line, "#")) {
      continue;
    }

    std::vector<std::string> row;
    boost::split(row, line, boost::is_any_of(delimiters));

    auto row_size = static_cast<index_t>(row.size());
    if (n_cols == 0) {
      n_cols = row_size;
    } else if (row_size != n_cols) {
      std::cerr << std::format("warning: skipping line {} with a different number of columns",
                               line_no)
                << std::endl;
      continue;
    }

    for (const auto& cell : row) {
      // On Unix platforms, std::getline() keeps the \r if the line ends with \r\n.
      // Use boost::trim_copy() to remove it.
      buffer.push_back(numeric::to_double(boost::trim_copy(cell)));
    }
  }

  if (n_cols == 0) {
    throw std::runtime_error(std::format("file '{}' is empty", filename));
  }

  auto n_rows = static_cast<index_t>(buffer.size() / n_cols);
  return matrixd::Map(buffer.data(), n_rows, n_cols);
}

template <class Derived>
void write_table(const std::string& filename, const Eigen::MatrixBase<Derived>& table,
                 char delimiter = ' ') {
  std::ofstream ofs(filename);
  if (!ofs) {
    throw std::runtime_error(std::format("cannot open file '{}'", filename));
  }

  auto n_cols = table.cols();
  for (auto row : table.rowwise()) {
    for (index_t i = 0; i < n_cols; i++) {
      ofs << numeric::to_string(row(i));
      if (i != n_cols - 1) {
        ofs << delimiter;
      }
    }
    ofs << '\n';
  }
}

}  // namespace polatory
