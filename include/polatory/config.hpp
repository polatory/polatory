// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <polatory/numeric/scoped_ftz.hpp>

namespace polatory {

#ifdef POLATORY_FTZ
const numeric::scoped_ftz static_ftz;
#endif

}
