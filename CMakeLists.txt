cmake_minimum_required(VERSION 3.1)

project(polatory CXX)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(polatory_get_boost_dlls)
include(polatory_target_contents)

find_package(Boost 1.67.0 REQUIRED)
find_package(Eigen3 REQUIRED)

option(COVERAGE "Compile with --coverage option" OFF)

if(UNIX)
    set(MKL_ROOT /opt/intel/mkl CACHE PATH "Path to mkl")
    set(MKL_INCLUDE_DIR "${MKL_ROOT}/include")
    set(MKL_LIB_DIR "${MKL_ROOT}/lib/intel64")

    if(CMAKE_CXX_COMPILER_ID MATCHES "^(Clang|GNU)$")
        set(CMAKE_CXX_FLAGS "-std=c++14 -fopenmp")
        set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
        if(COVERAGE)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
        endif()
    else()
        message(FATAL_ERROR "Not supported compiler.")
    endif()
elseif(MSVC)
    set(_PF86 "ProgramFiles(x86)")
    file(TO_CMAKE_PATH "$ENV{${_PF86}}" _PROG_FILES_X86)

    if("$ENV{APPVEYOR}" STREQUAL "True")
        set(MKL_ROOT "$ENV{CONDA_DIR}/Library")
        set(MKL_DLL_DIR "${MKL_ROOT}/bin")
        set(MKL_INCLUDE_DIR "${MKL_ROOT}/include")
        set(MKL_LIB_DIR "${MKL_ROOT}/lib")
    else()
        set(MKL_ROOT "${_PROG_FILES_X86}/IntelSWTools/compilers_and_libraries/windows/mkl" CACHE PATH "Path to mkl")
        set(MKL_DLL_DIR "${MKL_ROOT}/../redist/intel64/mkl")
        set(MKL_INCLUDE_DIR "${MKL_ROOT}/include")
        set(MKL_LIB_DIR "${MKL_ROOT}/lib/intel64")
    endif()
    set(VCPKG_ROOT "" CACHE PATH "Path to vcpkg/installed/x64-windows")
    set(POLATORY_DLLS
        ${MKL_DLL_DIR}/mkl_avx.dll
        ${MKL_DLL_DIR}/mkl_avx2.dll
        ${MKL_DLL_DIR}/mkl_avx512.dll
        ${MKL_DLL_DIR}/mkl_core.dll
        ${MKL_DLL_DIR}/mkl_def.dll
        ${MKL_DLL_DIR}/mkl_mc.dll
        ${MKL_DLL_DIR}/mkl_mc3.dll
        ${MKL_DLL_DIR}/mkl_sequential.dll
    )

    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        set(CMAKE_CXX_FLAGS "/MP /EHsc /std:c++14 /openmp")
        set(CMAKE_CXX_FLAGS_RELEASE "/MD /O2 /Ob2 /DNDEBUG /GL")
    else()
        message(FATAL_ERROR "Not supported compiler.")
    endif()
    
    add_definitions(
        -DBOOST_ALL_DYN_LINK
        -DBOOST_ALL_NO_LIB
    )
else()
    message(FATAL_ERROR "Not supported system.")
endif()

add_definitions(
    -DEIGEN_DONT_PARALLELIZE
    -DEIGEN_MPL2_ONLY
    -DEIGEN_USE_MKL_ALL
    -DPOLATORY_FTZ
)

include_directories(
    include
    ${Boost_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIR}
    ${MKL_INCLUDE_DIR}
    ${VCPKG_ROOT}/include
)

if(UNIX)
    link_libraries(
        "-Wl,--start-group"
        ${MKL_LIB_DIR}/libmkl_intel_lp64.a ${MKL_LIB_DIR}/libmkl_sequential.a ${MKL_LIB_DIR}/libmkl_core.a pthread m dl
        ceres flann_cpp glog
        "-Wl,--end-group"
    )
elseif(MSVC)
    link_libraries(
        ${MKL_LIB_DIR}/mkl_intel_lp64_dll.lib ${MKL_LIB_DIR}/mkl_sequential_dll.lib ${MKL_LIB_DIR}/mkl_core_dll.lib
    )
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        link_libraries(
            ${VCPKG_ROOT}/debug/lib/ceres-debug.lib
            ${VCPKG_ROOT}/debug/lib/flann_cpp-gd.lib
            ${VCPKG_ROOT}/debug/lib/glog.lib
        )
    else()
        link_libraries(
            ${VCPKG_ROOT}/lib/ceres.lib
            ${VCPKG_ROOT}/lib/flann_cpp.lib
            ${VCPKG_ROOT}/lib/glog.lib
        )
    endif()
endif()

add_subdirectory(benchmark)
add_subdirectory(examples)
add_subdirectory(src)

enable_testing()
add_subdirectory(test)
