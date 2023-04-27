cmake_minimum_required(VERSION 3.22)

project(coordinateTransform)

include_directories("/usr/include/eigen3")
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})
add_executable(coordinateTransform coordinateTransform.cpp)
add_executable(plotTrajectory plotTrajectory.cpp)
target_link_libraries(plotTrajectory ${Pangolin_LIBRARIES})
