cmake_minimum_required(VERSION 3.22)

find_package(OpenCV REQUIRED)
find_package(Sophus REQUIRED)
find_package(Pangolin REQUIRED)

add_executable(jointMap jointMap.cpp)

target_link_libraries(jointMap ${OpenCV_LIBS} ${Pangolin_LIBRARIES} Sophus::Sophus)