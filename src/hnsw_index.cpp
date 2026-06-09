#include "ann/hnsw_index.hpp"
#include "ann/distance.hpp"

#include <algorithm>
#include <stdexcept>

namespace ann {

HNSWIndex::HNSWIndex(std::size_t M)
    : M_(M) {

    if (M_ == 0) {
        throw std::invalid_argument("M must be > 0");
    }
}

void HNSWIndex::build(
    const float* data,
    std::size_t num_vectors,
    std::size_t dim
) {
    if (data == nullptr) {
        throw std::invalid_argument("HNSWIndex::build received null data");
    }

    if (num_vectors == 0 || dim == 0) {
        throw std::invalid_argument("num_vectors and dim must be > 0");
    }

    num_vectors_ = num_vectors;
    dim_ = dim;

    data_.assign(data, data + num_vectors_ * dim_);

    graph_.clear();
    graph_.resize(num_vectors_);

    for (std::size_t i = 0; i < num_vectors_; ++i) {
        std::vector<SearchResult> distances;

        const float* vector_i = data_.data() + i * dim_;

        for (std::size_t j = 0; j < num_vectors_; ++j) {
            if (i == j) {
                continue;
            }

            const float* vector_j = data_.data() + j * dim_;

            float distance = l2_distance_squared_avx2(
                vector_i,
                vector_j,
                dim_
            );

            distances.push_back({j, distance});
        }

        std::sort(
            distances.begin(),
            distances.end(),
            [](const SearchResult& a, const SearchResult& b) {
                return a.distance < b.distance;
            }
        );

        std::size_t neighbors_to_keep = std::min(M_, distances.size());

        for (std::size_t n = 0; n < neighbors_to_keep; ++n) {
            graph_[i].push_back(distances[n].index);
        }
    }
}

std::vector<SearchResult> HNSWIndex::search(
    const float* query,
    std::size_t k
) const {
    if (query == nullptr) {
        throw std::invalid_argument("HNSWIndex::search received null query");
    }

    if (data_.empty()) {
        throw std::runtime_error("HNSWIndex::search called before build");
    }

    if (k == 0) {
        return {};
    }

    std::vector<bool> visited(num_vectors_, false);
    std::vector<SearchResult> candidates;

    std::size_t current = 0;
    visited[current] = true;

    float current_distance = l2_distance_squared_avx2(
        query,
        data_.data() + current * dim_,
        dim_
    );

    bool improved = true;

    while (improved) {
        improved = false;

        candidates.push_back({current, current_distance});

        for (std::size_t neighbor : graph_[current]) {
            if (visited[neighbor]) {
                continue;
            }

            visited[neighbor] = true;

            float neighbor_distance = l2_distance_squared_avx2(
                query,
                data_.data() + neighbor * dim_,
                dim_
            );

            candidates.push_back({neighbor, neighbor_distance});

            if (neighbor_distance < current_distance) {
                current = neighbor;
                current_distance = neighbor_distance;
                improved = true;
            }
        }
    }

    std::sort(
        candidates.begin(),
        candidates.end(),
        [](const SearchResult& a, const SearchResult& b) {
            return a.distance < b.distance;
        }
    );

    if (candidates.size() > k) {
        candidates.resize(k);
    }

    return candidates;
}

}