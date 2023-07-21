set(TARGET polatory)

file(GLOB_RECURSE SOURCES
    "*.cpp"
    "${PROJECT_SOURCE_DIR}/third_party/ScalFMM/src/*.cpp"
)
add_library(${TARGET} STATIC ${SOURCES})

if(UNIX)
    target_compile_options(${TARGET} PUBLIC -Wall -Wextra -Werror)
    if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(${TARGET} PUBLIC -O2)
    endif()
elseif(MSVC)
    # C4702(unreachable code) is disabled as it is reported by link.exe in third-party code.
    target_compile_options(${TARGET} PUBLIC /W4 /WX /wd4702)
endif()

target_compile_definitions(${TARGET} PUBLIC
    -DEIGEN_DONT_PARALLELIZE
    -DEIGEN_MPL2_ONLY
    -DEIGEN_USE_BLAS
)

if(MSVC)
    target_compile_definitions(${TARGET} PUBLIC
        -DWIN32_LEAN_AND_MEAN
        -DNOMINMAX
    )
endif()

if(APPLE)
    target_compile_definitions(${TARGET} PUBLIC
        -DPOLATORY_USE_ACCELERATE
    )
else()
    target_compile_definitions(${TARGET} PUBLIC
        -DEIGEN_USE_MKL_ALL
        -DPOLATORY_USE_MKL
    )
endif()


target_include_directories(${TARGET} PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)
if(NOT(APPLE))
    target_include_directories(${TARGET} SYSTEM PUBLIC
        ${MKL_INCLUDE_DIR}
    )
endif()
target_include_directories(${TARGET} SYSTEM PRIVATE
    ${PROJECT_SOURCE_DIR}/third_party/ScalFMM/include
)

target_link_libraries(${TARGET} PUBLIC
    Boost::boost Boost::filesystem Boost::serialization
    ceres
    double-conversion::double-conversion
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
)

if(UNIX)
    target_link_libraries(${TARGET} PUBLIC
        flann::flann_cpp_s
    )
    if(APPLE)
        target_link_libraries(${TARGET} INTERFACE
            "-framework Accelerate"
        )
    else()
        target_link_libraries(${TARGET} INTERFACE
            "-Wl,--start-group"
            ${MKL_LIB_DIR}/libmkl_intel_lp64.a ${MKL_LIB_DIR}/libmkl_sequential.a ${MKL_LIB_DIR}/libmkl_core.a pthread m dl
            "-Wl,--end-group"
        )
    endif()
elseif(MSVC)
    target_link_libraries(${TARGET} PUBLIC
        flann::flann_cpp
    )
    target_link_libraries(${TARGET} INTERFACE
        ${MKL_LIB_DIR}/mkl_intel_lp64_dll.lib ${MKL_LIB_DIR}/mkl_sequential_dll.lib ${MKL_LIB_DIR}/mkl_core_dll.lib
    )
endif()

install(TARGETS ${TARGET} ARCHIVE DESTINATION lib)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/polatory DESTINATION include)
