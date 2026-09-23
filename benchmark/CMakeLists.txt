add_executable(kernel_cost kernel_cost.cpp)
target_link_libraries(kernel_cost PRIVATE polatory)

add_executable(points points.cpp)
target_link_libraries(points PRIVATE polatory)

add_executable(predict predict.cpp)
target_link_libraries(predict PRIVATE polatory)

if(MSVC)
    polatory_target_contents(predict ${POLATORY_DLLS})
endif()

set(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmark.sh
    ${CMAKE_CURRENT_SOURCE_DIR}/predict.R
    ${CMAKE_CURRENT_SOURCE_DIR}/simulate.R
)

polatory_target_contents(predict ${FILES})
