cmake_minimum_required(VERSION 3.0)

project(polatory CXX)

find_package(Boost 1.60 REQUIRED)
find_package(Eigen3 REQUIRED)

if(UNIX)
    set(MKL_ROOT /opt/intel/mkl)

    set(CMAKE_CXX_FLAGS "-std=c++14 -fopenmp")
    set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
elseif(MSVC)
    set(MKL_ROOT "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/mkl")
    set(MKL_DLL_DIR "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/redist/intel64/mkl")

    set(VCPKG_ROOT "" CACHE STRING "Path to vcpkg/installed/x64-windows")
    set(DLLS
        ${MKL_DLL_DIR}/mkl_avx2.dll
        ${MKL_DLL_DIR}/mkl_core.dll
        ${MKL_DLL_DIR}/mkl_def.dll
        ${MKL_DLL_DIR}/mkl_sequential.dll
    )
    if(CMAKE_BUILD_TYPE MATCHES Debug)
        set(DLLS
            ${DLLS}
            ${VCPKG_ROOT}/debug/bin/ceres-debug.dll
            ${VCPKG_ROOT}/debug/bin/glog.dll
            ${VCPKG_ROOT}/debug/bin/lapack.dll
            ${VCPKG_ROOT}/debug/bin/metis.dll
            ${VCPKG_ROOT}/debug/bin/openblas.dll
        )
    else()
        set(DLLS
            ${DLLS}
            ${VCPKG_ROOT}/bin/ceres.dll
            ${VCPKG_ROOT}/bin/glog.dll
            ${VCPKG_ROOT}/bin/lapack.dll
            ${VCPKG_ROOT}/bin/metis.dll
            ${VCPKG_ROOT}/bin/openblas.dll
        )
    endif()
    
    set(CMAKE_CXX_FLAGS "/MP /fp:strict /fp:except- /Qstd=c++14 /Qopenmp /Qip /QaxAVX")
endif()

include_directories(
    include
    include/polatory/third_party
    include/polatory/third_party/ScalFMM
    ${Boost_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIR}
    ${MKL_ROOT}/include
    ${VCPKG_ROOT}/include
)

add_definitions(
    -DEIGEN_DONT_PARALLELIZE
    -DEIGEN_MPL2_ONLY
    -DEIGEN_USE_MKL_ALL
)

if(UNIX)
    link_libraries(
        "-Wl,--start-group"
        ${MKL_ROOT}/lib/intel64/libmkl_intel_lp64.a ${MKL_ROOT}/lib/intel64/libmkl_sequential.a ${MKL_ROOT}/lib/intel64/libmkl_core.a pthread m dl
        ceres
        "-Wl,--end-group"
    )
elseif(MSVC)
    include_directories(
        ${MKL_ROOT}/include
    )
    link_libraries(
        ${MKL_ROOT}/lib/intel64/mkl_intel_lp64_dll.lib ${MKL_ROOT}/lib/intel64/mkl_sequential_dll.lib ${MKL_ROOT}/lib/intel64/mkl_core_dll.lib
    )
    if(CMAKE_BUILD_TYPE MATCHES Debug)
        link_libraries(
            ${VCPKG_ROOT}/debug/lib/ceres-debug.lib
            ${VCPKG_ROOT}/debug/lib/glog.lib
        )
    else()
        link_libraries(
            ${VCPKG_ROOT}/lib/ceres.lib
            ${VCPKG_ROOT}/lib/glog.lib
        )
    endif()
endif()

add_subdirectory(src)

enable_testing()
add_subdirectory(test)
