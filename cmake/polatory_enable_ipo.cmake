## # polatory_enable_ipo
## 
## Enables interprocedural optimization for the target.
function(polatory_enable_ipo TARGET)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        return()
    endif()

    # IPO support for MSVC was added in CMake 3.12.
    #   https://gitlab.kitware.com/cmake/cmake/issues/18189
    #   https://gitlab.kitware.com/cmake/cmake/merge_requests/1721
    # However, it was broken in CMake 3.14 and fixed in 3.15.
    #   https://gitlab.kitware.com/cmake/cmake/issues/19571
    if(CMAKE_VERSION VERSION_LESS 3.15)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            target_compile_options(${TARGET} PRIVATE /GL)
            target_link_libraries(${TARGET} PRIVATE -LTCG)
        endif()
    else()
        include(CheckIPOSupported)
        check_ipo_supported(RESULT ipo_supported OUTPUT output)
        if(ipo_supported)
            set_property(TARGET ${TARGET} PROPERTY INTERPROCEDURAL_OPTIMIZATION ON)
        else()
            message(WARNING "${output}")
        endif()
    endif()
endfunction()
