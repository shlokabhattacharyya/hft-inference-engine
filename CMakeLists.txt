cmake_minimum_required(VERSION 3.16)
project(hft_inference_engine
	VERSION 0.1.0
	DESCRIPTION "real-time market microstructure inference platform"
	LANGUAGES C CXX)

# compiler settings
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # for clangd/IDE tooling

# default to Release if not specified (want optimizations on by default)
if (NOT CMAKE_BUILD_TYPE)
	set (CMAKE_BUILD_TYPE Release)
endif()

# warnings
add_compile_options (-Wall -Wextra -Wpedantic)

# AXV2 for SIMD kernel
add_compile_options(-maxv2 -mfma)

# C kernel (building as static library)
add_library(kernel STATIC
	kernel/engine.c
	kernel/engine_simd.c
)
target_include_directories(kernel PUBLIC kernel)

# C++ main binary
add_executable(hft_inference_engine
	src/main.cpp
)
target_link_libraries(hft_inferece_engine PRIVATE kernel)

# legacy C CLI (for regression testing kernel)
add_executable(hft_inference_engine_cli kernel/main_cli.c)
target_link_libraries(hft_inference_engine_cli PRIVATE kernel)
