set(TARGET polatory)

file(GLOB_RECURSE SOURCES
    "*.cpp"
    "${PROJECT_SOURCE_DIR}/third_party/ScalFMM/src/*.cpp"
)
add_library(${TARGET} STATIC ${SOURCES})

if(UNIX)
    target_compile_options(${TARGET} PUBLIC -fPIC -Wall -Wextra -Werror)
elseif(MSVC)
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
    ${CMAKE_BINARY_DIR}/scalfmm/install/include
)

target_link_libraries(${TARGET} PUBLIC
    Boost::boost
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
