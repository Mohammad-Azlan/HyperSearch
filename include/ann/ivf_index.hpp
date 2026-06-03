#pragma once

#include "ann/index.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace ann {

class IVFIndex : public Index {
public:
    IVFIndex(std::size_t nlist, std::size_t nprobe, std::size_t kmeans_iterations);

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
        return "IVF-Flat";
    }

    void save(const std::string& path) const;
    void load(const std::string& path);

private:
    std::size_t nlist_;
    std::size_t nprobe_;
    std::size_t kmeans_iterations_;

    std::size_t num_vectors_ = 0;
    std::size_t dim_ = 0;

    std::vector<float> data_;
    std::vector<float> centroids_;

    // inverted_lists_[c] contains vector indices belonging to cluster c
    std::vector<std::vector<std::size_t>> inverted_lists_;
};

}