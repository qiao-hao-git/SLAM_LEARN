cmake_minimum_required(VERSION 3.22)

project(gaussNewton)
set(CMAKE_CXX_STANDARD 17)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(OpenCV REQUIRED)
include_directories("/usr/include/eigen3")
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(Ceres REQUIRED)
find_package(G2O REQUIRED)

add_executable(gaussNewton gaussNewton.cpp)
add_executable(ceresCurveFitting ceresCurveFitting.cpp)
add_executable(g2oCurveFitting g2oCurveFitting.cpp)

target_link_libraries(gaussNewton ${OpenCV_LIBS})
target_link_libraries(ceresCurveFitting ${OpenCV_LIBS} ${CERES_LIBRARIES})
target_link_libraries(g2oCurveFitting ${OpenCV_LIBS} ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY})
