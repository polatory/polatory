find_package(Boost 1.60 COMPONENTS serialization REQUIRED)

set(TARGET polatory)

file(GLOB_RECURSE SOURCES "*.cpp")
add_library(${TARGET} STATIC ${SOURCES})

target_link_libraries(${TARGET}
    ${Boost_SERIALIZATION_LIBRARY}
)

install(TARGETS ${TARGET} ARCHIVE DESTINATION lib)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/polatory DESTINATION include)
