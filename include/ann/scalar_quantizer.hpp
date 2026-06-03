#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <iosfwd>

namespace ann {

class ScalarQuantizer {
public:
    void train(
        const float* data,
        std::size_t num_vectors,
        std::size_t dim
    );

    std::vector<std::uint8_t> encode(
        const float* data,
        std::size_t num_vectors
    ) const;

    void decode_vector(
        const std::uint8_t* code,
        float* output
    ) const;

    std::size_t dim() const {
        return dim_;
    }

    std::size_t memory_usage_bytes() const;
    void save_state(std::ostream& out) const;
    void load_state(std::istream& in);

private:
    std::size_t dim_ = 0;

    std::vector<float> min_;
    std::vector<float> scale_;
};

}