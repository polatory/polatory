find_package(Boost 1.60 COMPONENTS program_options REQUIRED)

set(TARGET terrain_25d)

file(GLOB_RECURSE SOURCES "*.cpp")
add_executable(${TARGET} ${SOURCES})

target_link_libraries(${TARGET}
    ${PROJECT_NAME}
    ${Boost_PROGRAM_OPTIONS_LIBRARY}
)

foreach(dll ${DLLS})
    add_custom_command(TARGET ${TARGET}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${dll}
        $<TARGET_FILE_DIR:${TARGET}>
    )
endforeach()
