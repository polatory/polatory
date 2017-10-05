file(GLOB_RECURSE SOURCES "*.cpp")
add_library(${PROJECT_NAME} SHARED ${SOURCES})

target_link_libraries(${PROJECT_NAME}
        "-Wl,--start-group"
        ${MKL_ROOT}/lib/intel64/libmkl_intel_lp64.a ${MKL_ROOT}/lib/intel64/libmkl_sequential.a ${MKL_ROOT}/lib/intel64/libmkl_core.a pthread m dl
        ceres
        "-Wl,--end-group"
        )
