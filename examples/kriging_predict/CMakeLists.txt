find_package(Boost 1.60 COMPONENTS program_options REQUIRED)

set(TARGET kriging_predict)

file(GLOB_RECURSE SOURCES "*.cpp")
add_executable(${TARGET} ${SOURCES})

target_link_libraries(${TARGET}
    ${PROJECT_NAME}
    ${Boost_PROGRAM_OPTIONS_LIBRARY}
)

if(MSVC)
    if(CMAKE_BUILD_TYPE MATCHES Debug)
        file(GLOB BOOST_DLLS "${VCPKG_ROOT}/debug/bin/boost_program_options-vc141-mt-*.dll")
    else()
        file(GLOB BOOST_DLLS "${VCPKG_ROOT}/bin/boost_program_options-vc141-mt-*.dll")
    endif()
endif()

set(FILES
    ${DLLS}
    ${BOOST_DLLS}
)

foreach(file ${FILES})
    add_custom_command(TARGET ${TARGET}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${file}
        $<TARGET_FILE_DIR:${TARGET}>
    )
endforeach()
