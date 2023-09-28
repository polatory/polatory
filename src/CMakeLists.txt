set(TARGET polatory)

file(GLOB_RECURSE SOURCES
    "*.cpp"
)
add_library(${TARGET} STATIC ${SOURCES})

if(UNIX)
    target_compile_options(${TARGET} PUBLIC -fPIC -Wall -Wextra -Werror)
    # target_compile_options(${TARGET} PUBLIC -fPIC -Wall -Wextra -Werror -fsanitize=address)
    # target_compile_options(${TARGET} PUBLIC -fPIC -Wall -Wextra -Werror -fsanitize=undefined)
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
target_include_directories(${TARGET} SYSTEM PUBLIC
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

include(ExternalProject)

if(USE_MKL)
    target_compile_definitions(${TARGET} PUBLIC
        # -DEIGEN_USE_MKL_ALL
        -DEIGEN_USE_BLAS
    )

    target_include_directories(${TARGET} PUBLIC
        ${VCPKG_DIR}/include/fftw
    )

    target_link_libraries(${TARGET} PUBLIC
        $<LINK_ONLY:MKL::MKL>
    )

    ExternalProject_Add(
        scalfmm
        PREFIX ${CMAKE_BINARY_DIR}/scalfmm
        GIT_REPOSITORY https://github.com/polatory/ScalFMM3.git
        GIT_TAG polatory
        GIT_SHALLOW ON
        GIT_PROGRESS ON
        CMAKE_ARGS -DCMAKE_CXX_COMPILER=clang-cl -DCMAKE_CXX_FLAGS=-fopenmp -Dscalfmm_USE_MKL=ON -DCMAKE_TOOLCHAIN_FILE=${PROJECT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/scalfmm/install
    )
elseif(APPLE)
    target_compile_definitions(${TARGET} PUBLIC
        -DEIGEN_USE_BLAS
    )

    target_link_libraries(${TARGET} INTERFACE
        "-framework Accelerate"
        FFTW3::fftw3
    )

    ExternalProject_Add(
        scalfmm
        PREFIX ${CMAKE_BINARY_DIR}/scalfmm
        GIT_REPOSITORY https://github.com/polatory/ScalFMM3.git
        GIT_TAG polatory
        GIT_SHALLOW ON
        GIT_PROGRESS ON
        CMAKE_ARGS -DBLA_VENDOR=Apple -DCMAKE_TOOLCHAIN_FILE=${PROJECT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/scalfmm/install
    )
endif()

add_dependencies(polatory scalfmm)

if(UNIX)
    # target_link_libraries(${TARGET} PRIVATE -fsanitize=address)
    # target_link_libraries(${TARGET} PRIVATE -fsanitize=undefined)
endif()
