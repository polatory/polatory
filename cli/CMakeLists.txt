set(TARGET cli)

add_executable(${TARGET}
    create_model_command.cpp
    cross_validate_command.cpp
    estimate_normals_command.cpp
    evaluate_command.cpp
    extract_model_command.cpp
    fit_command.cpp
    fit_model_to_variogram_command.cpp
    isosurface_command.cpp
    main.cpp
    normals_to_sdf_command.cpp
    show_model_command.cpp
    show_variogram_command.cpp
    surface_25d_command.cpp
    unique_command.cpp
    variogram_command.cpp
)

set_target_properties(${TARGET} PROPERTIES
    OUTPUT_NAME "jizai"
)

target_link_libraries(${TARGET} PRIVATE
    Boost::any
    Boost::program_options
    jizai
)

if(MSVC)
    jizai_target_contents(${TARGET} ${JIZAI_DLLS})
endif()
