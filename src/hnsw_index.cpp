#include "ann/hnsw_index.hpp"
#include "ann/distance.hpp"

#include <algorithm>
#include <queue>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

bool add_unique_neighbor(std::vector<std::size_t>& neighbors, std::size_t neighbor) {
    if (std::find(neighbors.begin(), neighbors.end(), neighbor) != neighbors.end()) {
        return false;
    }

    neighbors.push_back(neighbor);
    return true;
}

}

namespace ann {

HNSWIndex::HNSWIndex(
    std::size_t M,
    std::size_t ef_search,
    std::size_t ef_construction
)
    : M_(M),
      ef_search_(ef_search),
      ef_construction_(ef_construction) {

    if (M_ == 0 || ef_search_ == 0 || ef_construction_ == 0) {
        throw std::invalid_argument("M, ef_search, and ef_construction must be > 0");
    }

    if (ef_construction_ < M_) {
        throw std::invalid_argument("ef_construction must be >= M");
    }
}

std::vector<SearchResult> HNSWIndex::search_layer(
    const float* query,
    std::size_t entry_point,
    std::size_t ef,
    std::size_t level
) const {
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

    float entry_distance = l2_distance_squared_avx2(
        query,
        data_.data() + entry_point * dim_,
        dim_
    );

    candidates.push({entry_distance, entry_point});
    results.push({entry_distance, entry_point});
    visited[entry_point] = true;

    while (!candidates.empty()) {
        auto current = candidates.top();
        candidates.pop();

        if (results.size() >= ef && current.distance > results.top().distance) {
            break;
        }

        for (std::size_t neighbor : layers_[level][current.index]) {
            if (visited[neighbor]) {
                continue;
            }

            visited[neighbor] = true;

            float distance = l2_distance_squared_avx2(
                query,
                data_.data() + neighbor * dim_,
                dim_
            );

            if (results.size() < ef || distance < results.top().distance) {
                candidates.push({distance, neighbor});
                results.push({distance, neighbor});

                if (results.size() > ef) {
                    results.pop();
                }
            }
        }
    }

    std::vector<SearchResult> output;

    while (!results.empty()) {
        output.push_back({results.top().index, results.top().distance});
        results.pop();
    }

    std::sort(
        output.begin(),
        output.end(),
        [](const SearchResult& a, const SearchResult& b) {
            return a.distance < b.distance;
        }
    );

    return output;
}

void HNSWIndex::trim_neighbors(std::size_t node, std::size_t level) {
    auto& neighbors = layers_[level][node];

    neighbors.erase(
        std::remove(neighbors.begin(), neighbors.end(), node),
        neighbors.end()
    );

    std::sort(neighbors.begin(), neighbors.end());
    neighbors.erase(
        std::unique(neighbors.begin(), neighbors.end()),
        neighbors.end()
    );

    if (neighbors.size() <= M_) {
        return;
    }

    const float* node_vector = data_.data() + node * dim_;

    std::vector<SearchResult> scored_neighbors;
    scored_neighbors.reserve(neighbors.size());

    for (std::size_t neighbor : neighbors) {
        const float* neighbor_vector = data_.data() + neighbor * dim_;

        float distance = l2_distance_squared_avx2(
            node_vector,
            neighbor_vector,
            dim_
        );

        scored_neighbors.push_back({neighbor, distance});
    }

    std::sort(
        scored_neighbors.begin(),
        scored_neighbors.end(),
        [](const SearchResult& a, const SearchResult& b) {
            return a.distance < b.distance;
        }
    );

    neighbors.clear();

    std::size_t keep = std::min(M_, scored_neighbors.size());

    for (std::size_t i = 0; i < keep; ++i) {
        neighbors.push_back(scored_neighbors[i].index);
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

    layers_.clear();
    layers_.resize(1);
    layers_[0].resize(num_vectors_);

    entry_point_ = 0;
    max_level_ = 0;
    node_levels_.assign(num_vectors_, 0);

    auto greedy_search_layer = [this](
        const float* query,
        std::size_t entry_point,
        std::size_t level
    ) {
        std::size_t current = entry_point;
        bool improved = true;

        while (improved) {
            improved = false;

            float current_distance = l2_distance_squared_avx2(
                query,
                data_.data() + current * dim_,
                dim_
            );

            for (std::size_t neighbor : layers_[level][current]) {
                float neighbor_distance = l2_distance_squared_avx2(
                    query,
                    data_.data() + neighbor * dim_,
                    dim_
                );

                if (neighbor_distance < current_distance) {
                    current = neighbor;
                    current_distance = neighbor_distance;
                    improved = true;
                }
            }
        }

        return current;
    };

    for (std::size_t i = 1; i < num_vectors_; ++i) {
        std::size_t node_level = random_level();
        node_levels_[i] = node_level;

        std::size_t previous_max_level = max_level_;
        std::size_t current = entry_point_;

        bool becomes_new_entry_point = node_level > previous_max_level;

        if (becomes_new_entry_point) {
            layers_.resize(node_level + 1);

            for (std::size_t level = previous_max_level + 1; level <= node_level; ++level) {
                layers_[level].resize(num_vectors_);
            }
        }

        const float* vector_i = data_.data() + i * dim_;

        for (std::size_t level = previous_max_level; level > node_level; --level) {
            current = greedy_search_layer(vector_i, current, level);
        }

        std::size_t highest_connected_level = std::min(node_level, previous_max_level);

        for (std::size_t l = highest_connected_level + 1; l > 0; --l) {
            std::size_t level = l - 1;

            auto candidates = search_layer(
                vector_i,
                current,
                ef_construction_,
                level
            );

            if (!candidates.empty()) {
                current = candidates[0].index;
            }

            std::size_t keep = std::min(M_, candidates.size());

            for (std::size_t n = 0; n < keep; ++n) {
                std::size_t neighbor = candidates[n].index;

                if (neighbor == i) {
                    continue;
                }

                if (node_levels_[neighbor] < level) {
                    continue;
                }

                add_unique_neighbor(layers_[level][i], neighbor);
                add_unique_neighbor(layers_[level][neighbor], i);

                trim_neighbors(neighbor, level);
            }

            trim_neighbors(i, level);
        }

        if (becomes_new_entry_point) {
            max_level_ = node_level;
            entry_point_ = i;
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

    std::size_t current = entry_point_;

    for (std::size_t level = max_level_; level > 0; --level) {
        bool improved = true;

        while (improved) {
            improved = false;

            float current_distance = l2_distance_squared_avx2(
                query,
                data_.data() + current * dim_,
                dim_
            );

            for (std::size_t neighbor : layers_[level][current]) {
                float neighbor_distance = l2_distance_squared_avx2(
                    query,
                    data_.data() + neighbor * dim_,
                    dim_
                );

                if (neighbor_distance < current_distance) {
                    current = neighbor;
                    current_distance = neighbor_distance;
                    improved = true;
                }
            }
        }
    }

    auto results = search_layer(
        query,
        current,
        std::max(ef_search_, k),
        0
    );

    if (results.size() > k) {
        results.resize(k);
    }

    return results;
}

std::size_t HNSWIndex::random_level() {
    static thread_local std::mt19937 rng(42);
    static thread_local std::uniform_real_distribution<double> dist(0.0, 1.0);

    std::size_t level = 0;
    const double probability = 1.0 / static_cast<double>(M_);

    while (dist(rng) < probability) {
        ++level;
    }

    return level;
}

}