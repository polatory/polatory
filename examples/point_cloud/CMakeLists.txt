set(TARGET point_cloud)

file(GLOB_RECURSE SOURCES "*.cpp")
add_executable(${TARGET} ${SOURCES})

target_link_libraries(${TARGET}
    ${PROJECT_NAME}
)

set(FILES
    ${DLLS}
    ${CMAKE_CURRENT_SOURCE_DIR}/point_cloud.sh
)

foreach(file ${FILES})
    add_custom_command(TARGET ${TARGET}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${file}
        $<TARGET_FILE_DIR:${TARGET}>
    )
endforeach(file)
