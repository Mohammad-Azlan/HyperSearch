#include "ann/scalar_quantizer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ann {

void ScalarQuantizer::train(
    const float* data,
    std::size_t num_vectors,
    std::size_t dim
) {
    if (data == nullptr) {
        throw std::invalid_argument("ScalarQuantizer::train received null data");
    }

    if (num_vectors == 0 || dim == 0) {
        throw std::invalid_argument("num_vectors and dim must be > 0");
    }

    dim_ = dim;

    min_.assign(dim_, std::numeric_limits<float>::max());
    std::vector<float> max(dim_, std::numeric_limits<float>::lowest());

    for (std::size_t i = 0; i < num_vectors; ++i) {
        const float* vector = data + i * dim_;

        for (std::size_t d = 0; d < dim_; ++d) {
            min_[d] = std::min(min_[d], vector[d]);
            max[d] = std::max(max[d], vector[d]);
        }
    }

    scale_.resize(dim_);

    for (std::size_t d = 0; d < dim_; ++d) {
        float range = max[d] - min_[d];

        if (range == 0.0f) {
            scale_[d] = 1.0f;
        } else {
            scale_[d] = range / 255.0f;
        }
    }
}

std::vector<std::uint8_t> ScalarQuantizer::encode(
    const float* data,
    std::size_t num_vectors
) const {
    if (data == nullptr) {
        throw std::invalid_argument("ScalarQuantizer::encode received null data");
    }

    if (dim_ == 0) {
        throw std::runtime_error("ScalarQuantizer::encode called before train");
    }

    std::vector<std::uint8_t> codes(num_vectors * dim_);

    for (std::size_t i = 0; i < num_vectors; ++i) {
        const float* vector = data + i * dim_;
        std::uint8_t* code = codes.data() + i * dim_;

        for (std::size_t d = 0; d < dim_; ++d) {
            float normalized = (vector[d] - min_[d]) / scale_[d];
            int q = static_cast<int>(std::round(normalized));

            if(q<0) q=0;
            else if (q>255) q=255;

            code[d] = static_cast<std::uint8_t>(q);
        }
    }

    return codes;
}

void ScalarQuantizer::decode_vector(
    const std::uint8_t* code,
    float* output
) const {
    if (code == nullptr || output == nullptr) {
        throw std::invalid_argument("ScalarQuantizer::decode_vector received null pointer");
    }

    if (dim_ == 0) {
        throw std::runtime_error("ScalarQuantizer::decode_vector called before train");
    }

    for (std::size_t d = 0; d < dim_; ++d) {
        output[d] = min_[d] + static_cast<float>(code[d]) * scale_[d];
    }
}

std::size_t ScalarQuantizer::memory_usage_bytes() const {
    return min_.size() * sizeof(float)
         + scale_.size() * sizeof(float);
}

}