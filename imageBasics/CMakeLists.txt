cmake_minimum_required(VERSION 3.22)

project(image)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(image imageBasics.cpp)
add_executable(imageDistort undistortImage.cpp)

target_link_libraries(image ${OpenCV_LIBS})
target_link_libraries(imageDistort ${OpenCV_LIBS})