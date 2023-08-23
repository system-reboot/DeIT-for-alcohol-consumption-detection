cmake_minimum_required(VERSION 3.22.1)

project(Alcohol-Consumption-Detector)

find_package(Torch REQUIRED)
find_package( OpenCV REQUIRED )

set(CMAKE_PREFIX_PATH "/home/sanskar/libtorch")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3 ${TORCH_CXX_FLAGS}")

add_executable(Alcohol-Consumption-Detector ../src/main.cpp)

target_link_libraries(Alcohol-Consumption-Detector ${TORCH_LIBRARIES} ${OpenCV_LIBS})

set_property(TARGET Alcohol-Consumption-Detector PROPERTY CXX_STANDARD 14)