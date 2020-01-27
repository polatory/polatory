set(TARGET terrain_surface)

add_executable(${TARGET}
    ../common/common.cpp
    main.cpp
)

target_link_libraries(${TARGET} PRIVATE
    ${Boost_LIBRARIES}
    polatory
)

polatory_enable_ipo(${TARGET})

if(MSVC)
    polatory_get_prebuilt_boost_dlls(BOOST_DLLS)
    polatory_target_contents(${TARGET} ${POLATORY_DLLS} ${BOOST_DLLS})
endif()
