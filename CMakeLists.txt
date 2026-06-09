cmake_minimum_required(VERSION 3.20)

project(ann_project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(ann_core
    src/distance.cpp
    src/brute_force_index.cpp
    src/benchmark.cpp
    src/kmeans.cpp
    src/ivf_index.cpp
    src/evaluation.cpp
    src/scalar_quantizer.cpp
    src/sq_brute_force_index.cpp
    src/ivf_sq_index.cpp
    src/product_quantizer.cpp
    src/pq_brute_force_index.cpp
    src/ivf_pq_index.cpp
    src/parallel_benchmark.cpp
    src/benchmark_report.cpp
    src/dataset.cpp
    src/hnsw_index.cpp
)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ann_core PRIVATE -mavx2)
endif()

target_include_directories(ann_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
add_executable(ann_demo
    apps/demo.cpp
)

target_link_libraries(ann_demo
    PRIVATE
        ann_core
)

add_executable(ann_benchmark
    apps/benchmark_sift.cpp
)

target_link_libraries(ann_benchmark
    PRIVATE
        ann_core
)

add_executable(ann_distance_benchmark
    apps/benchmark_distance.cpp
)

target_link_libraries(ann_distance_benchmark
    PRIVATE
        ann_core
)