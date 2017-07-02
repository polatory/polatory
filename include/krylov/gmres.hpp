// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <Eigen/Core>

#include "gmres_base.hpp"

namespace polatory {
namespace krylov {

class gmres : public gmres_base {
public:
   gmres(const linear_operator& op, const Eigen::VectorXd& rhs, int max_iter);

   void iterate_process() override;
};

} // namespace krylov
} // namespace polatory
