set(TARGET polatory)

add_library(${TARGET} STATIC
    fmm/impl/biharmonic2d.cpp
    fmm/impl/biharmonic3d.cpp
    fmm/impl/cov_cubic.cpp
    fmm/impl/cov_exponential.cpp
    fmm/impl/cov_gaussian.cpp
    fmm/impl/cov_generalized_cauchy3.cpp
    fmm/impl/cov_generalized_cauchy5.cpp
    fmm/impl/cov_generalized_cauchy7.cpp
    fmm/impl/cov_generalized_cauchy9.cpp
    fmm/impl/cov_spherical.cpp
    fmm/impl/cov_spheroidal3.cpp
    fmm/impl/cov_spheroidal3_direct_part.cpp
    fmm/impl/cov_spheroidal3_fast_part.cpp
    fmm/impl/cov_spheroidal5.cpp
    fmm/impl/cov_spheroidal5_direct_part.cpp
    fmm/impl/cov_spheroidal5_fast_part.cpp
    fmm/impl/cov_spheroidal7.cpp
    fmm/impl/cov_spheroidal7_direct_part.cpp
    fmm/impl/cov_spheroidal7_fast_part.cpp
    fmm/impl/cov_spheroidal9.cpp
    fmm/impl/cov_spheroidal9_direct_part.cpp
    fmm/impl/cov_spheroidal9_fast_part.cpp
    fmm/impl/triharmonic2d.cpp
    fmm/impl/triharmonic3d.cpp
    fmm/make_fmm_evaluator.cpp
    isosurface/cluster.cpp
    isosurface/mesh_defects_finder.cpp
    isosurface/refine.cpp
    isosurface/snap.cpp
    kriging/variogram_calculator.cpp
    kriging/variogram_fitting_1d.cpp
    kriging/variogram_fitting_2d.cpp
    kriging/variogram_fitting_3d.cpp
    krylov/fgmres.cpp
    krylov/gmres_base.cpp
    krylov/gmres.cpp
    krylov/minres.cpp
    point_cloud/kdtree.cpp
    point_cloud/normal_estimator.cpp
    point_cloud/plane_estimator.cpp
    point_cloud/random_points.cpp
    point_cloud/sdf_data_generator.cpp
)

set_target_properties(${TARGET} PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

if(UNIX)
    target_compile_options(${TARGET} PUBLIC -Wall -Wextra -Werror)
    # target_compile_options(${TARGET} PUBLIC -g -fno-omit-frame-pointer -fsanitize=address)
    # target_compile_options(${TARGET} PUBLIC -g -fno-omit-frame-pointer -fsanitize=undefined)
elseif(MSVC)
    target_compile_options(${TARGET} PUBLIC /W4 /WX /wd4702)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(${TARGET} PUBLIC
            -Wno-unused-command-line-argument
        )
    endif()
endif()

target_compile_definitions(${TARGET} PUBLIC
    -DEIGEN_DONT_PARALLELIZE
    -DEIGEN_MPL2_ONLY
    -DEIGEN_USE_BLAS
)

if(MSVC)
    target_compile_definitions(${TARGET} PUBLIC
        -DWIN32_LEAN_AND_MEAN
        -DNOMINMAX
    )
endif()

target_include_directories(${TARGET} PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_include_directories(${TARGET} SYSTEM PUBLIC
    ${CMAKE_BINARY_DIR}/scalfmm/install/include
    ${POCKETFFT_INCLUDE_DIR}
)

target_link_libraries(${TARGET} PUBLIC
    Boost::algorithm
    Boost::container_hash
    Boost::filesystem
    Boost::iterator
    Boost::math
    Boost::range
    Boost::unordered
    Eigen3::Eigen
    FastFloat::fast_float
    OpenMP::OpenMP_CXX
)

target_link_libraries(${TARGET} PRIVATE
    Boost::container
    Ceres::ceres
    igl::igl_core
    nanoflann::nanoflann
)

include(ExternalProject)

set(SCALFMM_CMAKE_ARGS
    -DBLA_VENDOR=${BLA_VENDOR}
    -DCMAKE_TOOLCHAIN_FILE=${PROJECT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake
    -DVCPKG_INSTALLED_DIR=${VCPKG_INSTALLED_DIR}
    -DVCPKG_TARGET_TRIPLET=${VCPKG_TARGET_TRIPLET}
    -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/scalfmm/install
    -Dscalfmm_BUILD_CHECK=OFF
    -Dscalfmm_BUILD_EXAMPLES=OFF
    -Dscalfmm_BUILD_TOOLS=OFF
)

if(MSVC)
    list(APPEND SCALFMM_CMAKE_ARGS
        -DCMAKE_CXX_COMPILER=clang++
        -DCMAKE_CXX_FLAGS=-fopenmp
    )
endif()

if(BLA_VENDOR STREQUAL "Intel10_64lp_seq")
    target_link_libraries(${TARGET} PRIVATE
        $<LINK_ONLY:MKL::MKL>
    )

    list(APPEND SCALFMM_CMAKE_ARGS
        -Dscalfmm_USE_MKL=ON
    )
elseif(BLA_VENDOR STREQUAL "Arm")
    set(BLA_PKGCONFIG_BLAS armpl-lp64-seq)
    set(BLA_PREFER_PKGCONFIG ON)
    find_package(BLAS REQUIRED)

    target_link_libraries(${TARGET} PRIVATE
        BLAS::BLAS
    )

    list(APPEND SCALFMM_CMAKE_ARGS
        -DBLA_PKGCONFIG_BLAS=armpl-lp64-seq
        -DBLA_PKGCONFIG_LAPACK=armpl-lp64-seq
        -DBLA_PREFER_PKGCONFIG=ON
    )
elseif(BLA_VENDOR STREQUAL "Apple")
    target_link_libraries(${TARGET} PRIVATE
        "-framework Accelerate"
    )
endif()

externalproject_add(
    scalfmm
    PREFIX ${CMAKE_BINARY_DIR}/scalfmm
    GIT_REPOSITORY https://github.com/polatory/ScalFMM3.git
    GIT_TAG polatory2
    GIT_SHALLOW ON
    GIT_REMOTE_UPDATE_STRATEGY CHECKOUT
    CMAKE_ARGS ${SCALFMM_CMAKE_ARGS}
)

add_dependencies(polatory scalfmm)

if(UNIX)
    # target_link_options(${TARGET} PUBLIC -fsanitize=address)
    # target_link_options(${TARGET} PUBLIC -fsanitize=undefined)
endif()
