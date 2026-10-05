add_executable(kernel_cost kernel_cost.cpp)
target_link_libraries(kernel_cost PRIVATE jizai)

add_executable(points points.cpp)
target_link_libraries(points PRIVATE jizai)

add_executable(predict predict.cpp)
target_link_libraries(predict PRIVATE jizai)

if(MSVC)
    jizai_target_contents(predict ${JIZAI_DLLS})
endif()

set(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmark.sh
    ${CMAKE_CURRENT_SOURCE_DIR}/predict.R
    ${CMAKE_CURRENT_SOURCE_DIR}/simulate.R
)

jizai_target_contents(predict ${FILES})
