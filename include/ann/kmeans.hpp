#pragma once

#include <cstddef>
#include <vector>

namespace ann {

struct KMeansResult {
    std::vector<float> centroids;              // k * dim
    std::vector<std::size_t> assignments;      // one cluster per vector
};

KMeansResult run_kmeans(
    const float* data,
    std::size_t num_vectors,
    std::size_t dim,
    std::size_t k,
    std::size_t iterations
);

}