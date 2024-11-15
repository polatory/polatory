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
    fmm/impl/cov_spheroidal5.cpp
    fmm/impl/cov_spheroidal7.cpp
    fmm/impl/cov_spheroidal9.cpp
    fmm/impl/inverse_multiquadric1.cpp
    fmm/impl/multiquadric1.cpp
    fmm/impl/multiquadric3.cpp
    fmm/impl/triharmonic2d.cpp
    fmm/impl/triharmonic3d.cpp
    fmm/make_fmm_evaluator.cpp
    isosurface/mesh_defects_finder.cpp
    kriging/variogram_calculator.cpp
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

if(UNIX)
    target_compile_options(${TARGET} PUBLIC -fPIC -Wall -Wextra -Werror -fno-relaxed-template-template-args -Wno-deprecated-no-relaxed-template-template-args -Wno-unknown-warning-option)
    # target_compile_options(${TARGET} PUBLIC -g -fno-omit-frame-pointer -fsanitize=address)
    # target_compile_options(${TARGET} PUBLIC -g -fno-omit-frame-pointer -fsanitize=undefined)
elseif(MSVC)
    target_compile_options(${TARGET} PUBLIC /W4 /WX /wd4702)
endif()

target_compile_definitions(${TARGET} PUBLIC
    -DEIGEN_DONT_PARALLELIZE
    -DEIGEN_MPL2_ONLY
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
)

target_link_libraries(${TARGET} PUBLIC
    Boost::boost
    Boost::filesystem
    ceres
    Eigen3::Eigen
    FastFloat::fast_float
    OpenMP::OpenMP_CXX
)

if(UNIX)
    target_link_libraries(${TARGET} PUBLIC
        flann::flann_cpp_s
    )
elseif(MSVC)
    target_link_libraries(${TARGET} PUBLIC
        flann::flann_cpp
    )
endif()

include(ExternalProject)

set(SCALFMM_CMAKE_ARGS
    -DCMAKE_TOOLCHAIN_FILE=${PROJECT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake
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

if(USE_MKL)
    target_compile_definitions(${TARGET} PUBLIC
        # -DEIGEN_USE_MKL_ALL
        -DEIGEN_USE_BLAS
    )

    target_include_directories(${TARGET} PUBLIC
        ${VCPKG_DIR}/include/fftw
    )

    target_link_libraries(${TARGET} PUBLIC
        $<LINK_ONLY:MKL::MKL>
    )

    list(APPEND SCALFMM_CMAKE_ARGS
        -Dscalfmm_USE_MKL=ON
    )
elseif(APPLE)
    target_compile_definitions(${TARGET} PUBLIC
        -DEIGEN_USE_BLAS
    )

    target_link_libraries(${TARGET} INTERFACE
        "-framework Accelerate"
        FFTW3::fftw3
    )

    list(APPEND SCALFMM_CMAKE_ARGS
        -DBLA_VENDOR=Apple
    )
endif()

if(APPLE)
    # Avoid linker error with XCode 15
    target_link_options(${TARGET} PUBLIC -ld_classic)
endif()

externalproject_add(
    scalfmm
    PREFIX ${CMAKE_BINARY_DIR}/scalfmm
    GIT_REPOSITORY https://github.com/polatory/ScalFMM3.git
    GIT_TAG polatory
    GIT_SHALLOW ON
    GIT_REMOTE_UPDATE_STRATEGY CHECKOUT
    CMAKE_ARGS ${SCALFMM_CMAKE_ARGS}
)

add_dependencies(polatory scalfmm)

if(UNIX)
    # target_link_options(${TARGET} PUBLIC -fsanitize=address)
    # target_link_options(${TARGET} PUBLIC -fsanitize=undefined)
endif()
