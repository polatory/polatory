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
)
if(MSVC)
    target_compile_definitions(${TARGET} PUBLIC
        -DWIN32_LEAN_AND_MEAN
        -DNOMINMAX
    )
endif()

target_include_directories(${TARGET} PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)
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
elseif(MSVC)
    target_link_libraries(${TARGET} PUBLIC
        flann::flann_cpp
    )
endif()

if(USE_MKL)
    target_compile_definitions(${TARGET} PUBLIC
        -DEIGEN_USE_MKL_ALL
        -DPOLATORY_USE_MKL
    )

    target_link_libraries(${TARGET} PUBLIC
        MKL::MKL
    )
elseif(APPLE)
    target_compile_definitions(${TARGET} PUBLIC
        -DEIGEN_USE_BLAS
        -DPOLATORY_USE_ACCELERATE
    )

    target_link_libraries(${TARGET} INTERFACE
        "-framework Accelerate"
    )
endif()

install(TARGETS ${TARGET} ARCHIVE DESTINATION lib)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/polatory DESTINATION include)
