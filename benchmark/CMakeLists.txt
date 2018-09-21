add_executable(points points.cpp)
target_link_libraries(points ${PROJECT_NAME})

add_executable(predict predict.cpp)
target_link_libraries(predict ${PROJECT_NAME})

if(MSVC)
    polatory_get_boost_dlls(BOOST_DLLS)
    polatory_target_contents(predict ${POLATORY_DLLS} ${BOOST_DLLS})
endif()

set(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmark.sh
    ${CMAKE_CURRENT_SOURCE_DIR}/predict.R
    ${CMAKE_CURRENT_SOURCE_DIR}/simulate.R
)

polatory_target_contents(predict ${FILES})
