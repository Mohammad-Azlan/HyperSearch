#include "ann/kmeans.hpp"

#include "ann/distance.hpp"

#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace ann {

KMeansResult run_kmeans(
    const float* data,
    std::size_t num_vectors,
    std::size_t dim,
    std::size_t k,
    std::size_t iterations
) {
    if (data == nullptr) {
        throw std::invalid_argument("run_kmeans received null data pointer");
    }

    if (num_vectors == 0 || dim == 0 || k == 0 || iterations == 0) {
        throw std::invalid_argument("num_vectors, dim, k, and iterations must be greater than 0");
    }

    if (k > num_vectors) {
        throw std::invalid_argument("k cannot be greater than num_vectors");
    }

    KMeansResult result;
    result.centroids.resize(k * dim);
    result.assignments.resize(num_vectors, 0);

    // Simple deterministic initialization:
    // choose k random existing vectors as initial centroids.
    std::mt19937 rng(42);
    std::uniform_int_distribution<std::size_t> pick(0, num_vectors - 1);

    for (std::size_t c = 0; c < k; ++c) {
        std::size_t idx = pick(rng);
        const float* src = data + idx * dim;
        float* dst = result.centroids.data() + c * dim;

        for (std::size_t d = 0; d < dim; ++d) {
            dst[d] = src[d];
        }
    }

    std::vector<float> new_centroids(k * dim, 0.0f);
    std::vector<std::size_t> counts(k, 0);

    for (std::size_t iter = 0; iter < iterations; ++iter) {
        std::fill(new_centroids.begin(), new_centroids.end(), 0.0f);
        std::fill(counts.begin(), counts.end(), 0);

        // Assignment step
        for (std::size_t i = 0; i < num_vectors; ++i) {
            const float* vector_i = data + i * dim;

            std::size_t best_cluster = 0;
            float best_distance = std::numeric_limits<float>::max();

            for (std::size_t c = 0; c < k; ++c) {
                const float* centroid = result.centroids.data() + c * dim;
                float distance = l2_distance_squared(vector_i, centroid, dim);

                if (distance < best_distance) {
                    best_distance = distance;
                    best_cluster = c;
                }
            }

            result.assignments[i] = best_cluster;
            counts[best_cluster]++;

            float* centroid_sum = new_centroids.data() + best_cluster * dim;
            for (std::size_t d = 0; d < dim; ++d) {
                centroid_sum[d] += vector_i[d];
            }
        }

        // Update step
        for (std::size_t c = 0; c < k; ++c) {
            float* centroid = result.centroids.data() + c * dim;

            if (counts[c] == 0) {
                // If a cluster became empty, keep old centroid.
                continue;
            }

            const float* centroid_sum = new_centroids.data() + c * dim;

            for (std::size_t d = 0; d < dim; ++d) {
                centroid[d] = centroid_sum[d] / static_cast<float>(counts[c]);
            }
        }
    }

    return result;
}

}