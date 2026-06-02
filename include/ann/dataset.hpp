#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ann {

struct FloatDataset {
    std::vector<float> data;
    std::size_t num_vectors = 0;
    std::size_t dim = 0;
};

struct IntDataset {
    std::vector<std::int32_t> data;
    std::size_t num_vectors = 0;
    std::size_t dim = 0;
};

FloatDataset load_fvecs(
    const std::string& path,
    std::size_t max_vectors = 0
);

IntDataset load_ivecs(
    const std::string& path,
    std::size_t max_vectors = 0
);

}