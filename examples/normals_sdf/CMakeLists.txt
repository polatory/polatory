set(TARGET normals_sdf)

add_executable(${TARGET}
    main.cpp
)

target_link_libraries(${TARGET} PRIVATE
    Boost::program_options
    polatory
)

polatory_enable_ipo(${TARGET})

if(MSVC)
    polatory_target_contents(${TARGET} ${POLATORY_DLLS})
endif()
