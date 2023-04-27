cmake_minimum_required(VERSION 3.22)

project(stereo)

find_package(OpenCV REQUIRED)
find_package(Pangolin REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("/usr/include/eigen3")
include_directories(${Pangolin_INCLUDE_DIRS})

add_executable(stereo stereoVision.cpp)
target_link_libraries(stereo ${OpenCV_LIBS})
target_link_libraries(stereo ${Pangolin_LIBRARIES})
