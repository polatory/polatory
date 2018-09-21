find_package(Boost 1.67.0 COMPONENTS filesystem system REQUIRED)

set(TARGET Unittest)

file(GLOB_RECURSE SOURCES "*.cpp")
add_executable(${TARGET} ${SOURCES})

if(UNIX)
    target_link_libraries(${TARGET}
        ${PROJECT_NAME} gtest pthread
    )
elseif(MSVC)
    target_link_libraries(${TARGET}
        ${PROJECT_NAME}
    )
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_link_libraries(${TARGET}
            ${VCPKG_ROOT}/debug/lib/manual-link/gtestd.lib
        )
    else()
        target_link_libraries(${TARGET}
            ${VCPKG_ROOT}/lib/manual-link/gtest.lib
        )
    endif()

    polatory_get_boost_dlls(BOOST_DLLS)
    polatory_target_contents(${TARGET} ${POLATORY_DLLS} ${BOOST_DLLS})
endif()

target_link_libraries(${TARGET}
    ${Boost_LIBRARIES}
)

add_test(
    NAME ${TARGET}
    COMMAND $<TARGET_FILE:${TARGET}>
)
