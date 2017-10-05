cmake_minimum_required(VERSION 3.0)

project(polatory CXX)

find_package(Boost 1.60 REQUIRED)
find_package(Eigen3 REQUIRED)

if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    set(INTEL_ROOT /opt/intel)
    set(MKL_ROOT ${INTEL_ROOT}/mkl)

    set(CMAKE_CXX_FLAGS "-std=c++14 -fopenmp")
    set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
elseif(CMAKE_SYSTEM_NAME MATCHES "Windows")
    # TODO
endif()

include_directories(
        include
        include/polatory/third_party
        include/polatory/third_party/ScalFMM
        ${Boost_INCLUDE_DIR}
        ${EIGEN3_INCLUDE_DIR}
        ${MKL_ROOT}/include
        )

add_definitions(
        -DEIGEN_DONT_PARALLELIZE
        -DEIGEN_MPL2_ONLY
        -DEIGEN_USE_MKL_ALL
        )

add_subdirectory(src)

enable_testing()
add_subdirectory(test)
