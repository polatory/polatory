file(GLOB_RECURSE SOURCES "*.cpp")
add_library(${PROJECT_NAME} STATIC ${SOURCES} ../include/polatory/polatory.hpp)
