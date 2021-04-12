cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(hellotorch)

find_package(OpenCV REQUIRED)

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS   "-g")

aux_source_directory(. DIR_SRCS)
aux_source_directory(networks/ DIR_SRCS)

add_executable(hellotorch ${DIR_SRCS})
target_include_directories(hellotorch PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(hellotorch ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET hellotorch PROPERTY CXX_STANDARD 14)