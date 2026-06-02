#include "ann/dataset.hpp"

#include <fstream>
#include <stdexcept>

namespace ann {

FloatDataset load_fvecs(
    const std::string& path,
    std::size_t max_vectors
) {
    std::ifstream file(path, std::ios::binary);

    if (!file) {
        throw std::runtime_error("Failed to open fvecs file: " + path);
    }

    FloatDataset dataset;

    while (true) {
        std::int32_t dim_i32 = 0;

        file.read(reinterpret_cast<char*>(&dim_i32), sizeof(std::int32_t));

        if (!file) {
            break;
        }

        if (dim_i32 <= 0) {
            throw std::runtime_error("Invalid dimension in fvecs file");
        }

        std::size_t dim = static_cast<std::size_t>(dim_i32);

        if (dataset.dim == 0) {
            dataset.dim = dim;
        } else if (dataset.dim != dim) {
            throw std::runtime_error("Inconsistent vector dimension in fvecs file");
        }

        if (max_vectors != 0 && dataset.num_vectors >= max_vectors) {
            break;
        }

        std::size_t old_size = dataset.data.size();
        dataset.data.resize(old_size + dim);

        file.read(
            reinterpret_cast<char*>(dataset.data.data() + old_size),
            static_cast<std::streamsize>(dim * sizeof(float))
        );

        if (!file) {
            throw std::runtime_error("Unexpected end of fvecs file");
        }

        dataset.num_vectors++;
    }

    return dataset;
}

IntDataset load_ivecs(
    const std::string& path,
    std::size_t max_vectors
) {
    std::ifstream file(path, std::ios::binary);

    if (!file) {
        throw std::runtime_error("Failed to open ivecs file: " + path);
    }

    IntDataset dataset;

    while (true) {
        std::int32_t dim_i32 = 0;

        file.read(reinterpret_cast<char*>(&dim_i32), sizeof(std::int32_t));

        if (!file) {
            break;
        }

        if (dim_i32 <= 0) {
            throw std::runtime_error("Invalid dimension in ivecs file");
        }

        std::size_t dim = static_cast<std::size_t>(dim_i32);

        if (dataset.dim == 0) {
            dataset.dim = dim;
        } else if (dataset.dim != dim) {
            throw std::runtime_error("Inconsistent vector dimension in ivecs file");
        }

        if (max_vectors != 0 && dataset.num_vectors >= max_vectors) {
            break;
        }

        std::size_t old_size = dataset.data.size();
        dataset.data.resize(old_size + dim);

        file.read(
            reinterpret_cast<char*>(dataset.data.data() + old_size),
            static_cast<std::streamsize>(dim * sizeof(std::int32_t))
        );

        if (!file) {
            throw std::runtime_error("Unexpected end of ivecs file");
        }

        dataset.num_vectors++;
    }

    return dataset;
}

}