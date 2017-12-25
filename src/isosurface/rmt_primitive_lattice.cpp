// Copyright (c) 2016, GSI and The Polatory Authors.

#include <polatory/isosurface/rmt_primitive_lattice.hpp>

namespace polatory {
namespace isosurface {

const std::array<geometry::vector3d, 3> PrimitiveVectors
  {
    rotation().transform_vector({ +1., +1., -1. }),
    rotation().transform_vector({ +1., -1., +1. }),
    rotation().transform_vector({ -1., +1., +1. })
  };

const std::array<geometry::vector3d, 3> ReciprocalPrimitiveVectors
  {
    rotation().transform_vector({ 1. / 2., 1. / 2., 0. }),
    rotation().transform_vector({ 1. / 2., 0., 1. / 2. }),
    rotation().transform_vector({ 0., 1. / 2., 1. / 2. })
  };

}  // namespace isosurface
}  // namespace polatory
