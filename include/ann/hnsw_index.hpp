#pragma once

#include "ann/index.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace ann {

class HNSWIndex : public Index {
public:
    HNSWIndex(std::size_t M, std::size_t ef_search);

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

private:
    std::size_t M_;

    std::size_t num_vectors_ = 0;
    std::size_t dim_ = 0;

    std::vector<float> data_;

    std::vector<std::vector<std::size_t>> graph_;
    std::size_t ef_search_;
};

}