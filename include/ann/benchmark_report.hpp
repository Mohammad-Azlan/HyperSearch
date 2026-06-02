#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace ann {

struct BenchmarkRow {
    std::string index_name;

    std::size_t nlist = 0;
    std::size_t nprobe = 0;
    std::size_t pq_m = 0;

    double recall_at_k = 0.0;

    double avg_ms = 0.0;
    double p50_ms = 0.0;
    double p95_ms = 0.0;

    double qps = 0.0;

    std::size_t memory_bytes = 0;
};

void write_benchmark_csv(
    const std::string& path,
    const std::vector<BenchmarkRow>& rows
);

}