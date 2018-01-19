set(TARGET polatory)

file(GLOB_RECURSE SOURCES "*.cpp")
add_library(${TARGET} STATIC ${SOURCES})

install(TARGETS ${TARGET} ARCHIVE DESTINATION lib)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/polatory DESTINATION include)
