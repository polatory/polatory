find_package(Boost 1.67.0 COMPONENTS filesystem system REQUIRED)
find_package(GTest REQUIRED)

set(TARGET Unittest)

file(GLOB_RECURSE SOURCES "*.cpp")
add_executable(${TARGET} ${SOURCES})

if(UNIX)
    target_link_libraries(${TARGET} pthread)
elseif(MSVC)
    polatory_get_prebuilt_boost_dlls(BOOST_DLLS)
    polatory_target_contents(${TARGET} ${POLATORY_DLLS} ${BOOST_DLLS})
endif()

target_link_libraries(${TARGET}
    ${Boost_LIBRARIES}
    GTest::GTest
    polatory
)

polatory_enable_ipo(${TARGET})

add_test(
    NAME ${TARGET}
    COMMAND $<TARGET_FILE:${TARGET}>
)
