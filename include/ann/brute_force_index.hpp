#pragma once

#include "ann/index.hpp"

#include <vector>
#include <string>

namespace ann {

class BruteForceIndex : public Index {
public:
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
        return "BruteForce";
    }

    void save(const std::string& path) const;
    void load(const std::string& path);

private:
    std::vector<float> data_;
    std::size_t num_vectors_ = 0;
    std::size_t dim_ = 0;
};

}