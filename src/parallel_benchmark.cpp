#include "ann/parallel_benchmark.hpp"

#include "ann/timer.hpp"

#include <algorithm>
#include <stdexcept>
#include <thread>
#include <vector>

namespace ann {

ParallelBenchmarkResult benchmark_search_parallel(
    const Index& index,
    const float* queries,
    std::size_t num_queries,
    std::size_t dim,
    std::size_t k,
    std::size_t num_threads
) {
    if (queries == nullptr) {
        throw std::invalid_argument("queries cannot be null");
    }

    if (num_queries == 0 || dim == 0 || k == 0 || num_threads == 0) {
        throw std::invalid_argument("invalid benchmark parameters");
    }

    std::vector<std::thread> threads;

    std::size_t queries_per_thread =
        (num_queries + num_threads - 1) / num_threads;

    Timer timer;

    for (std::size_t t = 0; t < num_threads; ++t) {
        std::size_t start = t * queries_per_thread;
        std::size_t end =
            std::min(start + queries_per_thread, num_queries);

        if (start >= num_queries) {
            break;
        }

        threads.emplace_back(
            [&index, queries, dim, k, start, end]() {
                for (std::size_t q = start; q < end; ++q) {
                    const float* query = queries + q * dim;
                    index.search(query, k);
                }
            }
        );
    }

    for (auto& thread : threads) {
        thread.join();
    }

    double total_ms = timer.elapsed_ms();

    ParallelBenchmarkResult result;
    result.total_ms = total_ms;
    result.qps =
        (static_cast<double>(num_queries) * 1000.0) / total_ms;

    return result;
}

}