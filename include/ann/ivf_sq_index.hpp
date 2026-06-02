#pragma once

#include "ann/index.hpp"
#include "ann/scalar_quantizer.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ann {

class IVFSQIndex : public Index {
public:
    IVFSQIndex(std::size_t nlist, std::size_t nprobe, std::size_t kmeans_iterations);

    void set_nprobe(std::size_t nprobe);

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
        return "IVF-SQ8";
    }

    std::size_t memory_usage_bytes() const;

private:
    std::size_t nlist_;
    std::size_t nprobe_;
    std::size_t kmeans_iterations_;

    std::size_t num_vectors_ = 0;
    std::size_t dim_ = 0;

    std::vector<float> centroids_;
    std::vector<std::vector<std::size_t>> inverted_lists_;

    ScalarQuantizer quantizer_;
    std::vector<std::uint8_t> codes_;
};

}