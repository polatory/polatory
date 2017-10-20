add_executable(points "points.cpp")
target_link_libraries(points ${PROJECT_NAME})

add_executable(predict "predict.cpp")
target_link_libraries(predict ${PROJECT_NAME})

foreach(dll ${DLLS})
    add_custom_command(TARGET predict
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${dll}
        $<TARGET_FILE_DIR:predict>
    )
endforeach(dll)

set(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmark.sh
    ${CMAKE_CURRENT_SOURCE_DIR}/predict.R
    ${CMAKE_CURRENT_SOURCE_DIR}/simulate.R
)

foreach(file ${FILES})
    add_custom_command(TARGET predict
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${file}
        $<TARGET_FILE_DIR:predict>
    )
endforeach(file)
