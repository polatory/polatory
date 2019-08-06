function(polatory_enable_clang_tidy)
    find_program(CLANG_TIDY clang-tidy)
    if(NOT CLANG_TIDY)
        message(FATAL_ERROR "Could not find clang-tidy.")
    endif()

    set(CMAKE_EXPORT_COMPILE_COMMANDS ON PARENT_SCOPE)

    add_custom_target(
        clang-tidy
        COMMAND ${PROJECT_SOURCE_DIR}/tools/run-clang-tidy
    )
endfunction()
