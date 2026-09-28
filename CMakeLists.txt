cmake_minimum_required(VERSION 3.12)

project(neuralnet)

file(GLOB SOURCES CONFIGURE_DEPENDS src/*.cpp src/*.c)
add_executable(${PROJECT_NAME} ${SOURCES})