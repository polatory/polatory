add_executable(points points.cpp)
target_link_libraries(points polatory)
polatory_enable_ipo(points)

add_executable(predict predict.cpp)
target_link_libraries(predict polatory)
polatory_enable_ipo(predict)

if(MSVC)
    polatory_target_contents(predict ${POLATORY_DLLS})
endif()

set(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmark.sh
    ${CMAKE_CURRENT_SOURCE_DIR}/predict.R
    ${CMAKE_CURRENT_SOURCE_DIR}/simulate.R
)

polatory_target_contents(predict ${FILES})
