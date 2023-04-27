cmake_minimum_required(VERSION 3.22)
project(orb)
set(CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(OpenCV REQUIRED)
find_package(Sophus REQUIRED)
find_package(G2O REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${G2O_INCLUDE_DIRS})
include_directories(${Sophus_INCLUDE_DIRS})
include_directories("/usr/include/eigen3/")
add_executable(orb_cv obr_cv.cpp)
target_link_libraries(orb_cv ${OpenCV_LIBS})

add_executable(orb_self orb_self.cpp)
target_link_libraries(orb_self ${OpenCV_LIBS})

add_executable(pose_estimation_2d2d pose_estimation_2d2d.cpp)
target_link_libraries(pose_estimation_2d2d ${OpenCV_LIBS})

add_executable(triangulation triangulation.cpp)
target_link_libraries(triangulation ${OpenCV_LIBS})

add_executable(pose_estimation_3d2d pose_estimation_3d2d.cpp)
target_link_libraries(pose_estimation_3d2d ${OpenCV_LIBS} ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY} Sophus::Sophus)

add_executable(pose_estimation_3d3d  pose_estimate_3d3d.cpp)
target_link_libraries(pose_estimation_3d3d ${OpenCV_LIBS} ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY} Sophus::Sophus)
