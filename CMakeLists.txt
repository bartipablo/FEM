cmake_minimum_required(VERSION 3.22)
project(Rurki2)

set(CMAKE_CXX_STANDARD 23)

add_executable(Rurki2 main.cpp)

#[[
find_package(PythonLibs 2.7)
target_include_directories(Rurki2 PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(Rurki2 ${PYTHON_LIBRARIES})]]
