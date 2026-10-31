cmake_minimum_required(VERSION 3.16)

project(curans LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(CURANS_BUILD_GPU "Build the CUDA executable" ON)
set(CURANS_CUDA_ARCH "" CACHE STRING
    "NVCC -arch value, e.g. native or sm_86 (empty: compiler/CMake default)")

if(CURANS_BUILD_GPU)
    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(WARNING "CUDA compiler not found; the curans executable will not be built")
    else()
        enable_language(CUDA)

        add_executable(curans src/main.cpp src/gpu.cu)
        target_include_directories(curans PRIVATE src)
        target_compile_options(curans PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
            $<$<COMPILE_LANGUAGE:CXX>:-Wall>
            $<$<COMPILE_LANGUAGE:CXX>:-Wextra>)

        if(CURANS_CUDA_ARCH)
            if(NOT CMAKE_VERSION VERSION_LESS 3.18)
                set(CMAKE_CUDA_ARCHITECTURES OFF)
            endif()
            target_compile_options(curans PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:-arch=${CURANS_CUDA_ARCH}>)
        endif()
    endif()
endif()

include(CTest)

if(BUILD_TESTING)
    add_executable(rans_test tests/rans_test.cpp)
    target_include_directories(rans_test PRIVATE src)
    target_compile_options(rans_test PRIVATE -Wall -Wextra)
    add_test(NAME rans COMMAND rans_test)

    add_executable(frame_test tests/frame_test.cpp)
    target_include_directories(frame_test PRIVATE src)
    target_compile_options(frame_test PRIVATE -Wall -Wextra)
    add_test(NAME frame COMMAND frame_test)
endif()
