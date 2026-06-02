#include "ann/benchmark.hpp"

#include "ann/timer.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace ann {

namespace {

double percentile(std::vector<double> values, double p) {
    if (values.empty()) {
        return 0.0;
    }

    std::sort(values.begin(), values.end());

    double rank = (p / 100.0) * static_cast<double>(values.size() - 1);
    std::size_t index = static_cast<std::size_t>(rank);

    return values[index];
}

}

BenchmarkResult benchmark_search(
    const Index& index,
    const float* queries,
    std::size_t num_queries,
    std::size_t dim,
    std::size_t k
) {
    if (queries == nullptr) {
        throw std::invalid_argument("benchmark_search received null queries pointer");
    }

    if (num_queries == 0 || dim == 0) {
        throw std::invalid_argument("num_queries and dim must be greater than 0");
    }

    std::vector<double> latencies_ms;
    latencies_ms.reserve(num_queries);

    for (std::size_t q = 0; q < num_queries; ++q) {
        const float* query_ptr = queries + q * dim;

        Timer timer;
        auto results = index.search(query_ptr, k);
        double elapsed = timer.elapsed_ms();

        latencies_ms.push_back(elapsed);

        // Prevent compiler from aggressively ignoring results in optimized builds.
        volatile std::size_t sink = results.size();
        (void)sink;
    }

    double total = std::accumulate(latencies_ms.begin(), latencies_ms.end(), 0.0);
    double avg = total / static_cast<double>(latencies_ms.size());

    auto [min_it, max_it] = std::minmax_element(latencies_ms.begin(), latencies_ms.end());

    BenchmarkResult result;
    result.avg_ms = avg;
    result.min_ms = *min_it;
    result.max_ms = *max_it;
    result.p50_ms = percentile(latencies_ms, 50.0);
    result.p95_ms = percentile(latencies_ms, 95.0);

    return result;
}

}