#pragma once

#include "ann/index.hpp"

#include <cstddef>
#include <vector>

namespace ann {

struct BenchmarkResult {
    double avg_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    double p50_ms = 0.0;
    double p95_ms = 0.0;
};

BenchmarkResult benchmark_search(
    const Index& index,
    const float* queries,
    std::size_t num_queries,
    std::size_t dim,
    std::size_t k
);

}