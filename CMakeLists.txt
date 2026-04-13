cmake_minimum_required(VERSION 3.18)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")

project(parallel_prog LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Настройка архитектуры CUDA под видеокарту
set(CMAKE_CUDA_ARCHITECTURES "native")

add_executable(parallel_prog main.cu)
