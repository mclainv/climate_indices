cmake_minimum_required(VERSION 3.13)
project(nfdrs4_bench LANGUAGES C CXX)

# Path to the unmodified clone of https://github.com/firelab/NFDRS4 (see setup.ps1).
set(NFDRS4_SRC "" CACHE PATH "Path to the NFDRS4 clone")
if(NOT NFDRS4_SRC)
  message(FATAL_ERROR "pass -DNFDRS4_SRC=<path to NFDRS4 clone>")
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Only the three library targets the NFDRS4 core needs; the clone's top-level
# CMake (CLI apps, fw21, csv_readrow) is deliberately not used.
add_subdirectory(${NFDRS4_SRC}/lib/time64 ${CMAKE_BINARY_DIR}/time64)
add_subdirectory(${NFDRS4_SRC}/lib/utctime ${CMAKE_BINARY_DIR}/utctime)
add_subdirectory(${NFDRS4_SRC}/lib/NFDRS4 ${CMAKE_BINARY_DIR}/NFDRS4)

add_executable(bench bench.cpp)
find_package(Threads REQUIRED)
target_link_libraries(bench PRIVATE NFDRS4 Threads::Threads)
if(MSVC)
  target_compile_definitions(bench PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
