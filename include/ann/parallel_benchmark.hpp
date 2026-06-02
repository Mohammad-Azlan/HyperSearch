#pragma once

#include "ann/index.hpp"

#include <cstddef>

namespace ann {

struct ParallelBenchmarkResult {
    double total_ms = 0.0;
    double qps = 0.0;
};

ParallelBenchmarkResult benchmark_search_parallel(
    const Index& index,
    const float* queries,
    std::size_t num_queries,
    std::size_t dim,
    std::size_t k,
    std::size_t num_threads
);

}