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
    if(CMAKE_BUILD_TYPE MATCHES Debug)
        target_link_libraries(${TARGET}
            ${VCPKG_ROOT}/debug/lib/manual-link/gtest.lib
        )
    else()
        target_link_libraries(${TARGET}
            ${VCPKG_ROOT}/lib/manual-link/gtest.lib
        )
    endif()

    foreach(dll ${DLLS})
        add_custom_command(TARGET ${TARGET}
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${dll}
            $<TARGET_FILE_DIR:${TARGET}>
        )
    endforeach(dll)
endif()

add_test(
    NAME ${TARGET}
    COMMAND $<TARGET_FILE:${TARGET}>
)
