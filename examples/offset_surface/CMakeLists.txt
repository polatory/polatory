set(TARGET offset_surface)

add_executable(${TARGET}
    main.cpp
)

if(MSVC)
    target_compile_options(${TARGET} PRIVATE -Qunused-arguments)
endif()

target_link_libraries(${TARGET} PRIVATE
    Boost::program_options
    igl::igl_core
    polatory
)

if(MSVC)
    polatory_target_contents(${TARGET} ${POLATORY_DLLS})
endif()
