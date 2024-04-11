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

if(OPENMP_FOUND)
    set(OPENMP_COMPILER_FLAGS
        -fopenmp
    )
    set(OPENMP_LINKER_FLAGS
        -fopenmp
    )
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    execute_process(
        COMMAND brew --prefix libomp
        RESULT_VARIABLE BREW_LIBOMP
        OUTPUT_VARIABLE BREW_LIBOMP_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT BREW_LIBOMP EQUAL 0)
        message(FATAL_ERROR "libomp is not installed. Please run `brew install libomp`.")
    endif()
    set(OPENMP_COMPILER_FLAGS
        -Xclang -fopenmp -I${BREW_LIBOMP_PREFIX}/include
    )
    set(OPENMP_LINKER_FLAGS
        -Xclang -fopenmp -L${BREW_LIBOMP_PREFIX}/lib -lomp
    )
else()
    message(FATAL_ERROR "OpenMP is not found.")
endif()

target_compile_options(${TARGET} PUBLIC ${OPENMP_COMPILER_FLAGS})
target_link_libraries(${TARGET} PUBLIC ${OPENMP_LINKER_FLAGS})

string(REPLACE ";" " " OPENMP_COMPILER_FLAGS_STR "${OPENMP_COMPILER_FLAGS}")
string(REPLACE ";" " " OPENMP_LINKER_FLAGS_STR "${OPENMP_LINKER_FLAGS}")

set(SCALFMM_CMAKE_ARGS
    -DCMAKE_CXX_FLAGS=${OPENMP_COMPILER_FLAGS_STR}
    -DCMAKE_EXE_LINKER_FLAGS=${OPENMP_LINKER_FLAGS_STR}
    -DCMAKE_TOOLCHAIN_FILE=${PROJECT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake
    -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/scalfmm/install
)

if(MSVC)
    list(APPEND SCALFMM_CMAKE_ARGS
        -DCMAKE_CXX_COMPILER=clang++
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

include(ExternalProject)

externalproject_add(
    scalfmm
    PREFIX ${CMAKE_BINARY_DIR}/scalfmm
    GIT_REPOSITORY https://github.com/polatory/ScalFMM3.git
    GIT_TAG appleclang
    GIT_SHALLOW ON
    GIT_PROGRESS ON
    CMAKE_ARGS ${SCALFMM_CMAKE_ARGS}
)

add_dependencies(polatory scalfmm)

if(UNIX)
    # target_link_libraries(${TARGET} PRIVATE -fsanitize=address)
    # target_link_libraries(${TARGET} PRIVATE -fsanitize=undefined)
endif()
