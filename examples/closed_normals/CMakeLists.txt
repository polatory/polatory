set(TARGET closed_normals)

add_executable(${TARGET}
    main.cpp
)

target_link_libraries(${TARGET} PRIVATE
    Boost::program_options
    polatory
)

if(MSVC)
    polatory_target_contents(${TARGET} ${POLATORY_DLLS})
endif()
