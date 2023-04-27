cmake_minimum_required(VERSION 3.22)

set(CMAKE_BUILD_DEBUG "Debug")

project(HelloSLAM)

add_executable(helloSLAM helloSLAM.cpp)

add_library(hello libHelloSLAM.cpp)

add_library(hello_shared SHARED libHelloSLAM.cpp)

add_executable(useHello useHello.cpp)

target_link_libraries(useHello hello_shared)
