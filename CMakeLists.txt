cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(hellotorch)

find_package(Torch REQUIRED)

add_executable(hellotorch hellotorch.cpp)
target_link_libraries(hellotorch "${TORCH_LIBRARIES}")
set_property(TARGET hellotorch PROPERTY CXX_STANDARD 14)