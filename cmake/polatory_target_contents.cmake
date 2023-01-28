## # polatory_target_contents
##
## Copies files to the target directory after build.
function(polatory_target_contents TARGET)
    foreach(FILE ${ARGN})
        add_custom_command(TARGET ${TARGET}
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${FILE}
            $<TARGET_FILE_DIR:${TARGET}>
        )
    endforeach()

    add_custom_command ( TARGET ${TARGET} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:MKL::MKL> $<TARGET_FILE_DIR:${TARGET}>
    )
endfunction()
