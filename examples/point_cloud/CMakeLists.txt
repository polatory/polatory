find_package(Boost 1.67.0 COMPONENTS program_options REQUIRED)

set(TARGET point_cloud)

file(GLOB_RECURSE SOURCES "*.cpp")
add_executable(${TARGET} ${SOURCES})

target_link_libraries(${TARGET}
    ${Boost_LIBRARIES}
    polatory
)

polatory_enable_ipo(${TARGET})

if(MSVC)
    polatory_get_prebuilt_boost_dlls(BOOST_DLLS)
    polatory_target_contents(${TARGET} ${POLATORY_DLLS} ${BOOST_DLLS})
endif()
