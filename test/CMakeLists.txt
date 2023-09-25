set(TARGET Unittest)

file(GLOB_RECURSE SOURCES "*.cpp")
add_executable(${TARGET} ${SOURCES})

target_link_libraries(${TARGET} PRIVATE
    GTest::gtest GTest::gtest_main
    polatory
)

polatory_enable_ipo(${TARGET})

if(MSVC)
    polatory_target_contents(${TARGET} ${POLATORY_DLLS})
endif()

add_test(
    NAME ${TARGET}
    COMMAND $<TARGET_FILE:${TARGET}>
    # COMMAND $<TARGET_FILE:${TARGET}> --gtest_filter=rbf_evaluator.*
    # COMMAND $<TARGET_FILE:${TARGET}> --gtest_filter=rbf_fitter.*
    # COMMAND $<TARGET_FILE:${TARGET}> --gtest_filter=rbf_inequality_fitter.*
    # COMMAND $<TARGET_FILE:${TARGET}> --gtest_filter=rbf_operator.*
    # COMMAND $<TARGET_FILE:${TARGET}> --gtest_filter=rbf_symmetric_evaluator.*
)
