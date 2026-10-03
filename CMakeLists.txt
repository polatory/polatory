cmake_minimum_required(VERSION 3.18)

option(BUILD_BENCHMARKS "Build benchmark programs" ON)
option(BUILD_CLI "Build command-line interface" ON)
option(BUILD_EXAMPLES "Build example programs" ON)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)
option(BUILD_TESTS "Build unit tests" ON)

if(BUILD_PYTHON_BINDINGS)
    list(APPEND VCPKG_MANIFEST_FEATURES "python")
endif()

if(BUILD_TESTS)
    list(APPEND VCPKG_MANIFEST_FEATURES "tests")
endif()

if(BLA_VENDOR)
    set(VCPKG_MANIFEST_NO_DEFAULT_FEATURES ON)
    if(BLA_VENDOR STREQUAL "Arm")
        list(APPEND VCPKG_MANIFEST_FEATURES "armpl")
    elseif(BLA_VENDOR STREQUAL "Intel10_64lp_seq")
        list(APPEND VCPKG_MANIFEST_FEATURES "mkl")
    elseif(NOT BLA_VENDOR STREQUAL "Apple")
        message(FATAL_ERROR "Unsupported BLA_VENDOR: ${BLA_VENDOR}")
    endif()
endif()

project(Polatory CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(polatory_target_contents)

find_package(boost_algorithm CONFIG REQUIRED)
find_package(boost_any CONFIG REQUIRED)
find_package(boost_container CONFIG REQUIRED)
find_package(boost_container_hash CONFIG REQUIRED)
find_package(boost_filesystem CONFIG REQUIRED)
find_package(boost_iterator CONFIG REQUIRED)
find_package(boost_math CONFIG REQUIRED)
find_package(boost_program_options CONFIG REQUIRED)
find_package(boost_range CONFIG REQUIRED)
find_package(boost_unordered CONFIG REQUIRED)
find_package(Ceres CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(FastFloat CONFIG REQUIRED)
find_package(libigl CONFIG REQUIRED)
find_package(nanoflann CONFIG REQUIRED)
find_package(OpenMP REQUIRED)
find_package(xsimd CONFIG REQUIRED)
find_package(xtensor CONFIG REQUIRED)
find_package(xtensor-blas CONFIG REQUIRED)
find_package(xtl CONFIG REQUIRED)
find_path(POCKETFFT_INCLUDE_DIR pocketfft_hdronly.h REQUIRED)

set(VCPKG_DIR "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}")

if(NOT BLA_VENDOR)
    if(${VCPKG_TARGET_TRIPLET} MATCHES ^x64-)
        set(BLA_VENDOR "Intel10_64lp_seq")
    elseif(${VCPKG_TARGET_TRIPLET} MATCHES "^arm64-(linux|windows)")
        set(BLA_VENDOR "Arm")
    else()
        set(BLA_VENDOR "Apple")
    endif()
endif()

if(MSVC)
    string(TOLOWER "${CMAKE_CXX_COMPILER_ARCHITECTURE_ID}" MSVC_ARCH)
    file(TO_CMAKE_PATH "$ENV{VCToolsRedistDir}" MSVC_REDIST_DIR)
    file(GLOB POLATORY_DLLS
        "${MSVC_REDIST_DIR}/debug_nonredist/${MSVC_ARCH}/Microsoft.VC*.OpenMP.LLVM/libomp140.*.dll"
    )
    if(NOT POLATORY_DLLS)
        message(FATAL_ERROR "LLVM OpenMP runtime not found in VCToolsRedistDir.")
    endif()
endif()

if(BLA_VENDOR STREQUAL "Intel10_64lp_seq")
    set(MKL_ARCH "intel64")
    if(UNIX)
        set(MKL_LINK "static")
    elseif(MSVC)
        set(MKL_LINK "dynamic")
    endif()
    set(MKL_THREADING "sequential")
    set(MKL_INTERFACE "lp64")
    find_package(MKL CONFIG REQUIRED)

    if(MSVC)
        # https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-windows/2025-2/contents-of-the-bin-directory.html
        list(APPEND POLATORY_DLLS
            ${VCPKG_DIR}/$<$<CONFIG:Debug>:debug>/bin/mkl_avx2.2.dll
            ${VCPKG_DIR}/$<$<CONFIG:Debug>:debug>/bin/mkl_avx512.2.dll
            ${VCPKG_DIR}/$<$<CONFIG:Debug>:debug>/bin/mkl_core.2.dll
            ${VCPKG_DIR}/$<$<CONFIG:Debug>:debug>/bin/mkl_def.2.dll
            ${VCPKG_DIR}/$<$<CONFIG:Debug>:debug>/bin/mkl_mc3.2.dll
        )
    endif()
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
    find_package(GTest CONFIG REQUIRED)
    enable_testing()
    add_subdirectory(test)
endif()
