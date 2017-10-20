add_executable(points "points.cpp")
target_link_libraries(points ${PROJECT_NAME})

add_executable(predict "predict.cpp")
target_link_libraries(predict ${PROJECT_NAME})

configure_file("benchmark.sh" "benchmark.sh" COPYONLY)
configure_file("predict.R" "predict.R" COPYONLY)
configure_file("simulate.R" "simulate.R" COPYONLY)
