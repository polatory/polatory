## # polatory_enable_ipo
## 
## Enables interprocedural optimization for the target, if possible.
function(polatory_enable_ipo TARGET)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        return()
    endif()

    if(MSVC)
        set_property(TARGET ${TARGET} APPEND_STRING PROPERTY
            COMPILE_FLAGS " /GL "
        )
        set_property(TARGET ${TARGET} APPEND_STRING PROPERTY
            LINK_FLAGS " /LTCG "
        )
        message(STATUS "IPO is supported.")
    else()
        include(CheckIPOSupported)
        check_ipo_supported(RESULT ipo_supported OUTPUT output)
        if(ipo_supported)
            set_property(TARGET ${TARGET} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
            message(STATUS "IPO is supported.")
        else()
            message(STATUS "IPO is not supported: ${output}.")
        endif()
    endif()
endfunction()
