#include "ann/brute_force_index.hpp"

#include "ann/distance.hpp"
#include "ann/topk.hpp"

#include <stdexcept>

namespace ann {

void BruteForceIndex::build(
    const float* data,
    std::size_t num_vectors,
    std::size_t dim
) {
    if (data == nullptr) {
        throw std::invalid_argument("BruteForceIndex::build received null data pointer");
    }

    if (num_vectors == 0 || dim == 0) {
        throw std::invalid_argument("num_vectors and dim must be greater than 0");
    }

    num_vectors_ = num_vectors;
    dim_ = dim;

    data_.assign(data, data + num_vectors * dim);
}

std::vector<SearchResult> BruteForceIndex::search(
    const float* query,
    std::size_t k
) const {
    if (query == nullptr) {
        throw std::invalid_argument("BruteForceIndex::search received null query pointer");
    }

    if (data_.empty()) {
        throw std::runtime_error("BruteForceIndex::search called before build");
    }

    TopK topk(k);

    for (std::size_t i = 0; i < num_vectors_; ++i) {
        const float* vector_i = data_.data() + i * dim_;

        float distance = l2_distance_squared(query, vector_i, dim_);

        topk.push(i, distance);
    }

    return topk.results();
}

}