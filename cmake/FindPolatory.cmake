if(NOT POLATORY_FOUND)
    if(UNIX)
        set(SEARCH_PATS
            $ENV{HOME}/.local
            /usr/local
            /usr
        )
    elseif(MSVC)
        set(SEARCH_PATHS
            "$ENV{ProgramW6432}/polatory"
        )
    endif()
    set(SEARCH_PATHS
        ${SEARCH_PATHS}
        ${POLATORY_DIR}
    )

    find_path(POLATORY_INCLUDE_DIR polatory/polatory.hpp
        PATHS ${SEARCH_PATHS}
        PATH_SUFFIXES include
    )

    find_library(POLATORY_LIBRARY polatory
        PATHS ${SEARCH_PATHS}
        PATH_SUFFIXES lib
    )

    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(polatory DEFAULT_MSG
        POLATORY_LIBRARY
        POLATORY_INCLUDE_DIR
    )

    mark_as_advanced(
        POLATORY_INCLUDE_DIR
        POLATORY_LIBRARY
    )
endif()
