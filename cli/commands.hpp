#pragma once

#include <memory>

#include "command.hpp"

CommandPtr make_create_model_command();
CommandPtr make_cross_validate_command();
CommandPtr make_estimate_normals_command();
CommandPtr make_evaluate_command();
CommandPtr make_extract_model_command();
CommandPtr make_fit_command();
CommandPtr make_fit_model_to_variogram_command();
CommandPtr make_isosurface_command();
CommandPtr make_normals_to_sdf_command();
CommandPtr make_show_model_command();
CommandPtr make_show_variogram_command();
CommandPtr make_surface_25d_command();
CommandPtr make_unique_command();
CommandPtr make_variogram_command();
