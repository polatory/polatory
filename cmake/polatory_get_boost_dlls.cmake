## # polatory_get_boost_dlls
## 
## Returns paths of DLLs which correspond to ${Boost_LIBRARIES}.
function(polatory_get_boost_dlls BOOST_DLLS)
    set(DLLS "")
    set(SKIP_NEXT FALSE)
    foreach(LIB ${Boost_LIBRARIES})
        if(SKIP_NEXT)
            set(SKIP_NEXT FALSE)
            continue()
        elseif(LIB STREQUAL "debug")
            if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
                set(SKIP_NEXT TRUE)
            endif()
            continue()
        elseif(LIB STREQUAL "optimized")
            if(CMAKE_BUILD_TYPE STREQUAL "Debug")
                set(SKIP_NEXT TRUE)
            endif()
            continue()
        elseif(LIB STREQUAL "general")
            continue()
        endif()
        
        get_filename_component(BASE_NAME ${LIB} NAME_WE)
        get_filename_component(DIR ${LIB} PATH)
        
        list(APPEND DLLS "${DIR}/${BASE_NAME}.dll")
    endforeach()
    
    set(${BOOST_DLLS} ${DLLS} PARENT_SCOPE)
endfunction()
