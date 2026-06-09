#include "ann/hnsw_index.hpp"
#include "ann/distance.hpp"

#include <algorithm>
#include <stdexcept>
#include <queue>
#include <unordered_set>

namespace ann {

HNSWIndex::HNSWIndex(std::size_t M, std::size_t ef_search)
    : M_(M),
      ef_search_(ef_search) {

    if (M_ == 0 || ef_search_ == 0) {
        throw std::invalid_argument("M and ef_search must be > 0");
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

    struct MinCandidate {
        float distance;
        std::size_t index;

        bool operator<(const MinCandidate& other) const {
            return distance > other.distance;
        }
    };

    struct MaxResult {
        float distance;
        std::size_t index;

        bool operator<(const MaxResult& other) const {
            return distance < other.distance;
        }
    };

    std::priority_queue<MinCandidate> candidates;
    std::priority_queue<MaxResult> results;
    std::vector<bool> visited(num_vectors_, false);

    std::size_t entry = 0;

    float entry_distance = l2_distance_squared_avx2(
        query,
        data_.data() + entry * dim_,
        dim_
    );

    candidates.push({entry_distance, entry});
    results.push({entry_distance, entry});
    visited[entry] = true;

    while (!candidates.empty()) {
        auto current = candidates.top();
        candidates.pop();

        float worst_result_distance = results.top().distance;

        if (current.distance > worst_result_distance &&
            results.size() >= ef_search_) {
            break;
        }

        for (std::size_t neighbor : graph_[current.index]) {
            if (visited[neighbor]) {
                continue;
            }

            visited[neighbor] = true;

            float distance = l2_distance_squared_avx2(
                query,
                data_.data() + neighbor * dim_,
                dim_
            );

            if (results.size() < ef_search_ ||
                distance < results.top().distance) {

                candidates.push({distance, neighbor});
                results.push({distance, neighbor});

                if (results.size() > ef_search_) {
                    results.pop();
                }
            }
        }
    }

    std::vector<SearchResult> output;

    while (!results.empty()) {
        output.push_back({
            results.top().index,
            results.top().distance
        });
        results.pop();
    }

    std::sort(
        output.begin(),
        output.end(),
        [](const SearchResult& a, const SearchResult& b) {
            return a.distance < b.distance;
        }
    );

    if (output.size() > k) {
        output.resize(k);
    }

    return output;
}

}