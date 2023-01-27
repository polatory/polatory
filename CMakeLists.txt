cmake_minimum_required(VERSION 3.9)

if(NOT CMAKE_BUILD_TYPE)
    message("-- CMAKE_BUILD_TYPE is not set. Defaulting to Release.")
    # Must be set before a toolchain file is loaded by project() command.
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type: Debug or Release." FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(Polatory CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(polatory_common_functions)

if(UNIX)
    set(Boost_USE_STATIC_LIBS ON)
endif()
find_package(absl CONFIG REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem program_options serialization)
find_package(Ceres CONFIG REQUIRED)
find_package(double-conversion CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(flann CONFIG REQUIRED)
find_package(gsl-lite CONFIG REQUIRED)
find_package(GTest CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

option(USE_PREBUILT_BOOST "Use prebuilt Boost libraries specified manually. This option takes effect only on MSVC.")

option(ENABLE_CLANG_TIDY "Enable clang-tidy target. This option should not be used for code generation because it disables OpenMP.")

# vcpkg
set(VCPKG_DIR "${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}")
set(VCPKG_LIB_DIR "${VCPKG_DIR}$<$<CONFIG:Debug>:/debug>/lib")
set(VCPKG_INCLUDE_DIR "${VCPKG_DIR}/include")

if(UNIX)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "^(Clang|GNU)$")
        message(FATAL_ERROR "Not supported compiler.")
    endif()

    set(MKL_ROOT /opt/intel/oneapi/mkl/latest CACHE PATH "Path to mkl")
    set(MKL_INCLUDE_DIR "${MKL_ROOT}/include")
    set(MKL_LIB_DIR "${MKL_ROOT}/lib/intel64")
elseif(MSVC)
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        message(FATAL_ERROR "Not supported compiler.")
    endif()

    set(_PF86 "ProgramFiles(x86)")
    file(TO_CMAKE_PATH "$ENV{${_PF86}}" _PROG_FILES_X86)

    if("$ENV{CI}" STREQUAL "true")
        file(TO_CMAKE_PATH "$ENV{CONDA}" CONDA_DIR)

        set(MKL_ROOT "${CONDA_DIR}/Library")
        set(MKL_DLL_DIR "${MKL_ROOT}/bin")
        set(MKL_INCLUDE_DIR "${MKL_ROOT}/include")
        set(MKL_LIB_DIR "${MKL_ROOT}/lib")
    else()
        set(MKL_ROOT "${_PROG_FILES_X86}/Intel/oneAPI/mkl/latest" CACHE PATH "Path to mkl")
        set(MKL_DLL_DIR "${MKL_ROOT}/redist/intel64")
        set(MKL_INCLUDE_DIR "${MKL_ROOT}/include")
        set(MKL_LIB_DIR "${MKL_ROOT}/lib/intel64")
    endif()

    if(USE_PREBUILT_BOOST)
        if("${Boost_INCLUDE_DIR}" STREQUAL "${VCPKG_INCLUDE_DIR}")
            message(FATAL_ERROR "Boost libraries from vcpkg are detected. Turn off USE_PREBUILT_BOOST.")
        endif()
    endif()

    set(POLATORY_DLLS
        ${MKL_DLL_DIR}/mkl_avx.2.dll
        ${MKL_DLL_DIR}/mkl_avx2.2.dll
        ${MKL_DLL_DIR}/mkl_avx512.2.dll
        ${MKL_DLL_DIR}/mkl_core.2.dll
        ${MKL_DLL_DIR}/mkl_sequential.2.dll
    )
else()
    message(FATAL_ERROR "Not supported system.")
endif()

if(MSVC)
    string(REGEX REPLACE "/W[0-4]" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

if(ENABLE_CLANG_TIDY)
    polatory_enable_clang_tidy()
endif()

add_subdirectory(benchmark)
add_subdirectory(examples)
add_subdirectory(src)

enable_testing()
add_subdirectory(test)
