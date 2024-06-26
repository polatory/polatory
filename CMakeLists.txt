cmake_minimum_required(VERSION 3.18)

option(BUILD_BENCHMARKS "Build benchmark programs" ON)
option(BUILD_CLI "Build command-line interface" ON)
option(BUILD_EXAMPLES "Build example programs" ON)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)
option(BUILD_TESTS "Build unit tests" ON)

if(BUILD_PYTHON_BINDINGS)
    list(APPEND VCPKG_MANIFEST_FEATURES "python")
endif()

project(Polatory CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(polatory_target_contents)

set(MKL_ARCH "intel64")
if(UNIX)
    set(MKL_LINK "static")
elseif(MSVC)
    set(MKL_LINK "dynamic")
endif()
set(MKL_THREADING "sequential")
set(MKL_INTERFACE "lp64")

find_package(Boost REQUIRED COMPONENTS filesystem program_options)
find_package(Ceres CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(FastFloat CONFIG REQUIRED)
find_package(FFTW3 CONFIG)
find_package(flann CONFIG REQUIRED)
find_package(GTest CONFIG REQUIRED)
find_package(libigl CONFIG REQUIRED)
find_package(MKL CONFIG)
find_package(OpenMP REQUIRED)

set(VCPKG_DIR "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}")

if(${VCPKG_TARGET_TRIPLET} MATCHES ^x64-)
    set(USE_MKL ON)
else()
    set(BLA_VENDOR "Apple")
endif()

if(MSVC AND USE_MKL)
    # DLLs not copied by MKL::MKL
    # https://www.intel.com/content/www/us/en/develop/documentation/onemkl-windows-developer-guide/top/appendix-c-directory-structure-in-detail/structure-of-the-intel-64-architecture-directories/contents-of-the-redist-intel64-directory.html
    set(POLATORY_DLLS
        ${VCPKG_DIR}/$<$<CONFIG:Debug>:debug>/bin/mkl_def.2.dll
        ${VCPKG_DIR}/$<$<CONFIG:Debug>:debug>/bin/mkl_mc.2.dll
        ${VCPKG_DIR}/$<$<CONFIG:Debug>:debug>/bin/mkl_mc3.2.dll
        ${VCPKG_DIR}/$<$<CONFIG:Debug>:debug>/bin/mkl_avx.2.dll
        ${VCPKG_DIR}/$<$<CONFIG:Debug>:debug>/bin/mkl_avx2.2.dll
        ${VCPKG_DIR}/$<$<CONFIG:Debug>:debug>/bin/mkl_avx512.2.dll
    )
endif()

add_subdirectory(src)

if(BUILD_BENCHMARKS)
    add_subdirectory(benchmark)
endif()

if(BUILD_CLI)
    add_subdirectory(cli)
endif()

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(BUILD_PYTHON_BINDINGS)
    find_package(pybind11 CONFIG REQUIRED)
    add_subdirectory(python)
endif()

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()
