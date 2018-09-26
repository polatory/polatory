find_package(Boost 1.67.0 COMPONENTS serialization REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)

set(TARGET polatory)

file(GLOB_RECURSE SOURCES "*.cpp")
add_library(${TARGET} STATIC ${SOURCES})

if(UNIX)
    target_compile_options(${TARGET} PUBLIC -std=c++14 ${OpenMP_CXX_FLAGS})
    if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(${TARGET} PUBLIC -O2)
    endif()
elseif(MSVC)
    target_compile_options(${TARGET} PUBLIC /std:c++14 ${OpenMP_CXX_FLAGS})
endif()

target_compile_definitions(${TARGET} PUBLIC
    -DEIGEN_DONT_PARALLELIZE
    -DEIGEN_MPL2_ONLY
    -DEIGEN_USE_MKL_ALL
    -DPOLATORY_FTZ
)

if(USE_PREBUILT_BOOST)
    target_compile_definitions(${TARGET} PUBLIC
        -DBOOST_ALL_DYN_LINK
        -DBOOST_ALL_NO_LIB
    )
endif()

target_include_directories(${TARGET} PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${Boost_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIR}
    ${MKL_INCLUDE_DIR}
    ${VCPKG_INCLUDE_DIR}
)

if(UNIX)
    target_link_libraries(${TARGET} INTERFACE
        "-Wl,--start-group"
        ${MKL_LIB_DIR}/libmkl_intel_lp64.a ${MKL_LIB_DIR}/libmkl_sequential.a ${MKL_LIB_DIR}/libmkl_core.a pthread m dl
        ceres flann_cpp glog
        "-Wl,--end-group"
    )
elseif(MSVC)
    target_link_libraries(${TARGET} INTERFACE
        ${MKL_LIB_DIR}/mkl_intel_lp64_dll.lib ${MKL_LIB_DIR}/mkl_sequential_dll.lib ${MKL_LIB_DIR}/mkl_core_dll.lib
        debug ${VCPKG_LIB_DIR}/ceres-debug.lib optimized ${VCPKG_LIB_DIR}/ceres.lib
        debug ${VCPKG_LIB_DIR}/flann_cpp-gd.lib optimized ${VCPKG_LIB_DIR}/flann_cpp.lib
        ${VCPKG_LIB_DIR}/glog.lib
    )
endif()

target_link_libraries(${TARGET} INTERFACE
    ${Boost_LIBRARIES}
    ${OpenMP_CXX_LIBRARIES}
)

polatory_enable_ipo(${TARGET})

if(MSVC)
    polatory_get_prebuilt_boost_dlls(BOOST_DLLS)
    list(APPEND POLATORY_DLLS ${BOOST_DLLS})
    set(POLATORY_DLLS ${POLATORY_DLLS} PARENT_SCOPE)
endif()

install(TARGETS ${TARGET} ARCHIVE DESTINATION lib)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/polatory DESTINATION include)
