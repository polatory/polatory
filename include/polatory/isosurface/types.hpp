// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <array>
#include <cstdint>

#include <Eigen/Core>

namespace polatory {
namespace isosurface {

typedef uint64_t cell_index;

typedef int64_t cell_index_difference;

typedef Eigen::Vector3i cell_vector;

typedef int vertex_index;

typedef std::array<vertex_index, 3> face;

} // namespace isosurface
} // namespace polatory
