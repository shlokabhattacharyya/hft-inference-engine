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

# external dependencies
include(FetchContent)

FetchContent_Declare(
    ixwebsocket
    GIT_REPOSITORY https://github.com/machinezone/IXWebSocket.git
    GIT_TAG v11.4.5
)
set(USE_OPEN_SSL ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(ixwebsocket)

# default to Release if not specified (want optimizations on by default)
if (NOT CMAKE_BUILD_TYPE)
	set (CMAKE_BUILD_TYPE Release)
endif()

# warnings
add_compile_options (-Wall -Wextra -Wpedantic)

# AXV2 + FMA for SIMD kernel
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    add_compile_options(-mavx2 -mfma)
endif()

# C kernel (building as static library)
add_library(kernel STATIC
	kernel/engine.c
	kernel/engine_simd.c
	kernel/kernel_io.c
)
target_include_directories(kernel PUBLIC kernel)
target_link_libraries(kernel PUBLIC m) # engine.c uses expf from libm

# source files / targets
add_executable(ws_test
    src/ws_test.cpp
)
target_link_libraries(ws_test PRIVATE ixwebsocket)

# C++ main binary
add_executable(hft_inference_engine src/main.cpp)
target_compile_features(hft_inference_engine PRIVATE cxx_std_17)
target_link_libraries(hft_inference_engine PRIVATE kernel)

# legacy C CLI (for regression testing kernel)
add_executable(hft_inference_engine_cli kernel/main_cli.c)
target_link_libraries(hft_inference_engine_cli PRIVATE kernel)
