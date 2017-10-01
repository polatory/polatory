cmake_minimum_required(VERSION 3.0)

project(polatory CXX)

if(CMAKE_SYSTEM_NAME MATCHES "Linux")
  set(INTEL_ROOT /opt/intel)
  set(MKL_ROOT ${INTEL_ROOT}/mkl)
  
  set(CMAKE_CXX_FLAGS "-std=c++14 -fopenmp")
  set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
elseif(CMAKE_SYSTEM_NAME MATCHES "Windows")
  # TODO
endif()

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(Boost 1.60 REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(include)
include_directories(include/polatory/third_party)
include_directories(include/polatory/third_party/ScalFMM)
include_directories(${MKL_ROOT}/include)
include_directories(${Boost_INCLUDE_DIR})
include_directories(${EIGEN3_INCLUDE_DIR})

add_definitions(-DEIGEN_DONT_PARALLELIZE)
add_definitions(-DEIGEN_MPL2_ONLY)
add_definitions(-DEIGEN_USE_MKL_ALL)

file(GLOB_RECURSE TEST_SOURCES "test/*.cpp")
file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(Unittest ${SOURCES} ${TEST_SOURCES})

target_link_libraries(Unittest "-Wl,--start-group")
target_link_libraries(Unittest ${MKL_ROOT}/lib/intel64/libmkl_intel_lp64.a ${MKL_ROOT}/lib/intel64/libmkl_sequential.a ${MKL_ROOT}/lib/intel64/libmkl_core.a pthread m dl)
target_link_libraries(Unittest ceres)
target_link_libraries(Unittest gtest_main gtest pthread)
target_link_libraries(Unittest "-Wl,--end-group")
