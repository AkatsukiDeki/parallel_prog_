cmake_minimum_required(VERSION 3.15)
project(parallel_prog)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)

add_executable(parallel_prog main.cpp)

target_link_libraries(parallel_prog PRIVATE MPI::MPI_CXX)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(parallel_prog PRIVATE "-O3" "-march=native")
    if(MINGW)
        target_link_options(parallel_prog PRIVATE "-static-libgcc" "-static-libstdc++")
    endif()
elseif(MSVC)
    target_compile_options(parallel_prog PRIVATE "/O2" "/arch:AVX2")
endif()
