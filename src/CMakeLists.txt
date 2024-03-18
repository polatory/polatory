set(TARGET polatory)

add_library(${TARGET} STATIC
    fmm/impl/biharmonic2d.cpp
    fmm/impl/biharmonic3d.cpp
    fmm/impl/cov_cauchy3.cpp
    fmm/impl/cov_cauchy5.cpp
    fmm/impl/cov_cauchy7.cpp
    fmm/impl/cov_cauchy9.cpp
    fmm/impl/cov_cubic.cpp
    fmm/impl/cov_exponential.cpp
    fmm/impl/cov_gaussian.cpp
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
    kriging/empirical_variogram.cpp
    kriging/weight_function.cpp
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
    target_compile_options(${TARGET} PUBLIC -fPIC -Wall -Wextra -Werror)
    # target_compile_options(${TARGET} PUBLIC -fPIC -g -fno-omit-frame-pointer -Wall -Wextra -Werror -fsanitize=address)
    # target_compile_options(${TARGET} PUBLIC -fPIC -g -fno-omit-frame-pointer -Wall -Wextra -Werror -fsanitize=undefined)
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

    if(MSVC)
        set(COMPILER_SPEC -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_FLAGS=-fopenmp)
    endif()

    externalproject_add(
        scalfmm
        PREFIX ${CMAKE_BINARY_DIR}/scalfmm
        GIT_REPOSITORY https://github.com/polatory/ScalFMM3.git
        GIT_TAG polatory
        GIT_SHALLOW ON
        GIT_PROGRESS ON
        CMAKE_ARGS ${COMPILER_SPEC} -Dscalfmm_USE_MKL=ON -DCMAKE_TOOLCHAIN_FILE=${PROJECT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/scalfmm/install
    )
elseif(APPLE)
    target_compile_definitions(${TARGET} PUBLIC
        -DEIGEN_USE_BLAS
    )

    target_link_libraries(${TARGET} INTERFACE
        "-framework Accelerate"
        FFTW3::fftw3
    )

    externalproject_add(
        scalfmm
        PREFIX ${CMAKE_BINARY_DIR}/scalfmm
        GIT_REPOSITORY https://github.com/polatory/ScalFMM3.git
        GIT_TAG polatory
        GIT_SHALLOW ON
        GIT_PROGRESS ON
        CMAKE_ARGS -DBLA_VENDOR=Apple -DCMAKE_TOOLCHAIN_FILE=${PROJECT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/scalfmm/install
    )
endif()

add_dependencies(polatory scalfmm)

if(UNIX)
    # target_link_libraries(${TARGET} PRIVATE -fsanitize=address)
    # target_link_libraries(${TARGET} PRIVATE -fsanitize=undefined)
endif()
