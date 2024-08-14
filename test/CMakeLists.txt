set(TARGET Unittest)

add_executable(${TARGET}
    common/test_eigen_utility.cpp
    common/test_orthonormalize.cpp
    common/test_zip_sort.cpp
    geometry/test_bbox3d.cpp
    interpolation/test_evaluator.cpp
    interpolation/test_fitter.cpp
    interpolation/test_incremental_fitter.cpp
    interpolation/test_inequality_fitter.cpp
    interpolation/test_operator.cpp
    interpolation/test_symmetric_evaluator.cpp
    isosurface/test_bit.cpp
    isosurface/test_isosurface.cpp
    isosurface/test_rmt.cpp
    kriging/test_detrend.cpp
    kriging/test_variogram_calculator.cpp
    krylov/test_krylov.cpp
    numeric/test_conv.cpp
    point_cloud/test_distance_filter.cpp
    point_cloud/test_kdtree.cpp
    point_cloud/test_normal_estimator.cpp
    point_cloud/test_plane_estimator.cpp
    point_cloud/test_sdf_data_generator.cpp
    polynomial/test_lagrange_basis.cpp
    polynomial/test_polynomial_basis_base.cpp
    preconditioner/test_coarse_grid.cpp
    preconditioner/test_domain_divider.cpp
    preconditioner/test_fine_grid.cpp
    rbf/test_rbf.cpp
)

target_link_libraries(${TARGET} PRIVATE
    GTest::gtest GTest::gtest_main
    polatory
)

if(MSVC)
    polatory_target_contents(${TARGET} ${POLATORY_DLLS})
endif()

add_test(
    NAME ${TARGET}
    COMMAND $<TARGET_FILE:${TARGET}>
)
