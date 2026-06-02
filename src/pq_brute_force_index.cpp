#include "ann/pq_brute_force_index.hpp"

#include "ann/topk.hpp"

#include <stdexcept>
#include <vector>

namespace ann {

PQBruteForceIndex::PQBruteForceIndex(
    std::size_t m,
    std::size_t ksub,
    std::size_t iterations
)
    : iterations_(iterations),
      pq_(m, ksub) {
    if (iterations_ == 0) {
        throw std::invalid_argument("iterations must be > 0");
    }
}

void PQBruteForceIndex::build(
    const float* data,
    std::size_t num_vectors,
    std::size_t dim
) {
    if (data == nullptr) {
        throw std::invalid_argument("PQBruteForceIndex::build received null data");
    }

    if (num_vectors == 0 || dim == 0) {
        throw std::invalid_argument("num_vectors and dim must be > 0");
    }

    num_vectors_ = num_vectors;
    dim_ = dim;

    pq_.train(data, num_vectors_, dim_, iterations_);
    codes_ = pq_.encode(data, num_vectors_);
}

std::vector<SearchResult> PQBruteForceIndex::search(
    const float* query,
    std::size_t k
) const {
    if (query == nullptr) {
        throw std::invalid_argument("PQBruteForceIndex::search received null query");
    }

    if (codes_.empty()) {
        throw std::runtime_error("PQBruteForceIndex::search called before build");
    }

    auto distance_table = pq_.compute_distance_table(query);

    TopK topk(k);

    for (std::size_t i = 0; i < num_vectors_; ++i) {
        const std::uint8_t* code =
            codes_.data() + i * pq_.code_size_bytes();

        float distance = pq_.adc_distance(code, distance_table);

        topk.push(i, distance);
    }

    return topk.results();
}

std::size_t PQBruteForceIndex::memory_usage_bytes() const {
    return codes_.size() * sizeof(std::uint8_t)
         + pq_.memory_usage_bytes();
}

}