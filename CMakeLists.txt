cmake_minimum_required(VERSION 3.16)
cmake_policy(VERSION "3.16")

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

if(MSVC)
    if("$ENV{CI}" STREQUAL "true")
        file(TO_CMAKE_PATH "$ENV{CONDA}" CONDA_DIR)
        set(MKL_ROOT "${CONDA_DIR}/Library")
    endif()
endif()
set(MKL_ARCH "intel64")
set(MKL_LINK "dynamic")
set(MKL_THREADING "sequential")
set(MKL_INTERFACE "lp64")

if(UNIX)
    set(Boost_USE_STATIC_LIBS ON)
endif()
find_package(Boost REQUIRED COMPONENTS filesystem program_options serialization)
find_package(Ceres CONFIG REQUIRED)
find_package(double-conversion CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(flann CONFIG REQUIRED)
find_package(GTest CONFIG REQUIRED)
find_package(MKL CONFIG)
find_package(OpenMP REQUIRED)

option(USE_PREBUILT_BOOST "Use prebuilt Boost libraries specified manually. This option takes effect only on MSVC.")

option(ENABLE_CLANG_TIDY "Enable clang-tidy target. This option should not be used for code generation because it disables OpenMP.")

# vcpkg
set(VCPKG_DIR "${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}")
set(VCPKG_LIB_DIR "${VCPKG_DIR}$<$<CONFIG:Debug>:/debug>/lib")
set(VCPKG_INCLUDE_DIR "${VCPKG_DIR}/include")

if(MSVC)
    if(USE_PREBUILT_BOOST)
        if("${Boost_INCLUDE_DIR}" STREQUAL "${VCPKG_INCLUDE_DIR}")
            message(FATAL_ERROR "Boost libraries from vcpkg are detected. Turn off USE_PREBUILT_BOOST.")
        endif()
    endif()
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
