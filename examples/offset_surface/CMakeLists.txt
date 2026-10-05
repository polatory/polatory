set(TARGET offset_surface)

add_executable(${TARGET} main.cpp)

target_link_libraries(${TARGET} PRIVATE
    Boost::any
    Boost::program_options
    igl::igl_core
    jizai
)

if(MSVC)
    jizai_target_contents(${TARGET} ${JIZAI_DLLS})
endif()
