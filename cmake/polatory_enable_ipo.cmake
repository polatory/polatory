## # polatory_enable_ipo
## 
## Enables interprocedural optimization for the target.
function(polatory_enable_ipo TARGET)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        return()
    endif()

    # See https://gitlab.kitware.com/cmake/cmake/merge_requests/1721
    if(CMAKE_VERSION VERSION_LESS 3.12)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            target_compile_options(${TARGET} PRIVATE /GL)
            target_link_libraries(${TARGET} PRIVATE -LTCG)
            # NB: This variable also affects other targets in the same directory.
            set(CMAKE_STATIC_LINKER_FLAGS "${CMAKE_STATIC_LINKER_FLAGS} /LTCG" PARENT_SCOPE)
        endif()
    else()
        include(CheckIPOSupported)
        check_ipo_supported(RESULT ipo_supported OUTPUT output)
        if(ipo_supported)
            set_property(TARGET ${TARGET} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
        else()
            message(WARNING "${output}")
        endif()
    endif()
endfunction()
