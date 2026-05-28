cmake_minimum_required(VERSION 3.20)
project(Scheduler LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# List all your source files
set(SOURCE_FILES
    src/main.cc
)

set(LIB_SOURCE_FILES
    src/act_malloc.cc
    src/constraint.cc
    src/emitter.cc
    src/globals.cc
    src/parser.cc
    src/scheduler.cc
)

set(ORTOOLS_ROOT "/opt/ortools/")
set(CMAKE_PREFIX_PATH "${ORTOOLS_ROOT}" ${CMAKE_PREFIX_PATH})
find_package(ortools REQUIRED)

set(JSON_INCLUDE_PATH "/opt/nlohmann_json/include")
add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json INTERFACE ${JSON_INCLUDE_PATH})

# Create a static library to be linked with Rust
add_library(malloc_static STATIC ${LIB_SOURCE_FILES})
target_link_libraries(malloc_static PUBLIC ortools::ortools)
target_link_libraries(malloc_static PUBLIC nlohmann_json)
target_include_directories(malloc_static PRIVATE include)

# Also create an executable for debugging/testing
add_executable(malloc ${SOURCE_FILES})
target_link_libraries(malloc PRIVATE malloc_static)
target_include_directories(malloc PRIVATE include)
