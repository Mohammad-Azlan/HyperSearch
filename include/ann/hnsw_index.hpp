#pragma once

#include "ann/index.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace ann {

class HNSWIndex : public Index {
public:
    HNSWIndex(std::size_t M, std::size_t ef_search, std::size_t ef_construction);

    void build(
        const float* data,
        std::size_t num_vectors,
        std::size_t dim
    ) override;

    std::vector<SearchResult> search(
        const float* query,
        std::size_t k
    ) const override;

    std::string name() const override {
        return "HNSW";
    }

    std::size_t memory_usage_bytes() const;

    void save(const std::string& path) const;
    void load(const std::string& path);

private:
    std::size_t M_;

    std::size_t num_vectors_ = 0;
    std::size_t dim_ = 0;

    std::vector<float> data_;

    // layers_[level][node] stores neighbors of node at that level.
    std::vector<std::vector<std::vector<std::size_t>>> layers_;

    std::size_t ef_search_;
    std::size_t ef_construction_;
    std::size_t entry_point_ = 0;
    std::size_t max_level_ = 0;

    std::vector<std::size_t> node_levels_;

    std::vector<SearchResult> search_layer(
        const float* query,
        std::size_t entry_point,
        std::size_t ef,
        std::size_t level
    ) const;

    void trim_neighbors(std::size_t node, std::size_t level);

    std::size_t random_level();
};

}