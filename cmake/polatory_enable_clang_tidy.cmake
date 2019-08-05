function(polatory_enable_clang_tidy)
    find_program(CLANG_TIDY "clang-tidy")
    if(NOT CLANG_TIDY)
        message(AUTHOR_WARNING "clang-tidy not found.")
    else()
        set(CMAKE_EXPORT_COMPILE_COMMANDS 1 PARENT_SCOPE)
        
        add_custom_target(
            clang-tidy
            COMMAND ${PROJECT_SOURCE_DIR}/tools/run-clang-tidy
        )
    endif()
endfunction()
