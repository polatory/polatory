set(UNITTEST Unittest)
file(GLOB_RECURSE SOURCES "*.cpp")
add_executable(${UNITTEST} ${SOURCES})

target_link_libraries(${UNITTEST}
        ${PROJECT_NAME} gtest_main gtest pthread
        )
