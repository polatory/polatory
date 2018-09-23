find_package(Boost 1.67.0 COMPONENTS serialization REQUIRED)

set(TARGET polatory)

file(GLOB_RECURSE SOURCES "*.cpp")
add_library(${TARGET} STATIC ${SOURCES})

if(UNIX)
    target_link_libraries(${TARGET}
        "-Wl,--start-group"
        ${MKL_LIB_DIR}/libmkl_intel_lp64.a ${MKL_LIB_DIR}/libmkl_sequential.a ${MKL_LIB_DIR}/libmkl_core.a pthread m dl
        ceres flann_cpp glog
        "-Wl,--end-group"
    )
elseif(MSVC)
    target_link_libraries(${TARGET}
        ${MKL_LIB_DIR}/mkl_intel_lp64_dll.lib ${MKL_LIB_DIR}/mkl_sequential_dll.lib ${MKL_LIB_DIR}/mkl_core_dll.lib
        debug ${VCPKG_LIB_DIR}/ceres-debug.lib optimized ${VCPKG_LIB_DIR}/ceres.lib
        debug ${VCPKG_LIB_DIR}/flann_cpp-gd.lib optimized ${VCPKG_LIB_DIR}/flann_cpp.lib
        ${VCPKG_LIB_DIR}/glog.lib
    )
endif()

target_link_libraries(${TARGET}
    ${Boost_LIBRARIES}
)

if(MSVC)
    polatory_get_boost_dlls(BOOST_DLLS)
    list(APPEND POLATORY_DLLS ${BOOST_DLLS})
    set(POLATORY_DLLS ${POLATORY_DLLS} PARENT_SCOPE)
endif()

install(TARGETS ${TARGET} ARCHIVE DESTINATION lib)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/polatory DESTINATION include)
