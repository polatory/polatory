// Copyright (c) 2016, GSI and The Polatory Authors.

#include <polatory/isosurface/rmt_primitive_lattice.hpp>

#include <exception>

namespace polatory {
namespace isosurface {

namespace detail {

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4297)  // 'function' : function assumed not to throw an exception but does
#endif
primitive_vectors::primitive_vectors() noexcept try : base{ {
  rotation().transform_vector({ +1., +1., -1. }),
  rotation().transform_vector({ +1., -1., +1. }),
  rotation().transform_vector({ -1., +1., +1. })
} } {
} catch (const std::exception&) {
  std::terminate();
}

reciprocal_primitive_vectors::reciprocal_primitive_vectors() noexcept try : base{ {
  rotation().transform_vector({ 1. / 2., 1. / 2., 0. }),
  rotation().transform_vector({ 1. / 2., 0., 1. / 2. }),
  rotation().transform_vector({ 0., 1. / 2., 1. / 2. })
} } {
} catch (const std::exception&) {
  std::terminate();
}
#ifdef _MSC_VER
#pragma warning(pop)
#endif

}  // namespace detail

const detail::primitive_vectors PrimitiveVectors;

const detail::reciprocal_primitive_vectors ReciprocalPrimitiveVectors;

}  // namespace isosurface
}  // namespace polatory
