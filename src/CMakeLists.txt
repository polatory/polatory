find_package(Boost 1.67.0 COMPONENTS serialization REQUIRED)

set(TARGET polatory)

file(GLOB_RECURSE SOURCES "*.cpp")
add_library(${TARGET} STATIC ${SOURCES})

target_link_libraries(${TARGET}
    ${Boost_LIBRARIES}
)

if(MSVC)
    polatory_get_boost_dlls(BOOST_DLLS)
    list(APPEND POLATORY_DLLS ${BOOST_DLLS})
    set(POLATORY_DLLS ${POLATORY_DLLS} PARENT_SCOPE)
endif()

install(TARGETS ${TARGET} ARCHIVE DESTINATION lib)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/polatory DESTINATION include)
