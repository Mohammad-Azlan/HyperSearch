#include "ann/sq_brute_force_index.hpp"

#include "ann/distance.hpp"
#include "ann/topk.hpp"

#include <stdexcept>
#include <vector>

namespace ann {

void SQBruteForceIndex::build(
    const float* data,
    std::size_t num_vectors,
    std::size_t dim
) {
    if (data == nullptr) {
        throw std::invalid_argument("SQBruteForceIndex::build received null data");
    }

    if (num_vectors == 0 || dim == 0) {
        throw std::invalid_argument("num_vectors and dim must be > 0");
    }

    num_vectors_ = num_vectors;
    dim_ = dim;

    quantizer_.train(data, num_vectors_, dim_);
    codes_ = quantizer_.encode(data, num_vectors_);
}

std::vector<SearchResult> SQBruteForceIndex::search(
    const float* query,
    std::size_t k
) const {
    if (query == nullptr) {
        throw std::invalid_argument("SQBruteForceIndex::search received null query");
    }

    if (codes_.empty()) {
        throw std::runtime_error("SQBruteForceIndex::search called before build");
    }

    TopK topk(k);
    std::vector<float> decoded(dim_);

    for (std::size_t i = 0; i < num_vectors_; ++i) {
        const std::uint8_t* code = codes_.data() + i * dim_;

        quantizer_.decode_vector(code, decoded.data());

        float distance = l2_distance_squared(query, decoded.data(), dim_);

        topk.push(i, distance);
    }

    return topk.results();
}

std::size_t SQBruteForceIndex::memory_usage_bytes() const {
    return codes_.size() * sizeof(std::uint8_t)
         + quantizer_.memory_usage_bytes();
}

}