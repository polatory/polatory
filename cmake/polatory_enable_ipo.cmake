## # polatory_enable_ipo
##
## Enables interprocedural optimization for the target.
function(polatory_enable_ipo TARGET)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        return()
    endif()

    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_supported OUTPUT output)
    if(ipo_supported)
        set_property(TARGET ${TARGET} PROPERTY INTERPROCEDURAL_OPTIMIZATION ON)
    else()
        message(WARNING "${output}")
    endif()
endfunction()
