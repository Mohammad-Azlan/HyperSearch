#pragma once

#include "ann/index.hpp"
#include "ann/product_quantizer.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ann {

class PQBruteForceIndex : public Index {
public:
    PQBruteForceIndex(
        std::size_t m,
        std::size_t ksub,
        std::size_t iterations
    );

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
        return "PQ-BruteForce";
    }

    std::size_t memory_usage_bytes() const;

private:
    std::size_t iterations_;

    std::size_t num_vectors_ = 0;
    std::size_t dim_ = 0;

    ProductQuantizer pq_;
    std::vector<std::uint8_t> codes_;
};

}