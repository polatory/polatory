cmake_minimum_required(VERSION 3.16)
cmake_policy(VERSION "3.16")

project(Polatory CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(polatory_common_functions)

set(MKL_ARCH "intel64")
set(MKL_LINK "dynamic")
set(MKL_THREADING "sequential")
set(MKL_INTERFACE "lp64")

find_package(Boost REQUIRED COMPONENTS filesystem program_options serialization)
find_package(Ceres CONFIG REQUIRED)
find_package(double-conversion CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(flann CONFIG REQUIRED)
find_package(GTest CONFIG REQUIRED)
find_package(MKL CONFIG)
find_package(OpenMP REQUIRED)

set(VCPKG_DIR "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}")

if(${VCPKG_TARGET_TRIPLET} MATCHES ^x64-)
    set(USE_MKL ON)
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

add_subdirectory(benchmark)
add_subdirectory(examples)
add_subdirectory(src)

enable_testing()
add_subdirectory(test)
