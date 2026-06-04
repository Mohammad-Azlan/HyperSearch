#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <iosfwd>

namespace ann {

class ProductQuantizer {
public:
    ProductQuantizer(std::size_t m, std::size_t ksub);

    void train(
        const float* data,
        std::size_t num_vectors,
        std::size_t dim,
        std::size_t iterations
    );

    std::vector<std::uint8_t> encode(
        const float* data,
        std::size_t num_vectors
    ) const;

    void decode_vector(
        const std::uint8_t* code,
        float* output
    ) const;

    std::vector<float> compute_distance_table(
        const float* query
    ) const;

    float adc_distance(
        const std::uint8_t* code,
        const std::vector<float>& distance_table
    ) const;

    std::size_t m() const {
        return m_;
    }

    std::size_t ksub() const {
        return ksub_;
    }

    std::size_t dim() const {
        return dim_;
    }

    std::size_t dsub() const {
        return dsub_;
    }

    std::size_t code_size_bytes() const {
        return m_;
    }

    std::size_t memory_usage_bytes() const;

    void save_state(std::ostream& out) const;
    void load_state(std::istream& in);

private:
    std::size_t m_;
    std::size_t ksub_;

    std::size_t dim_ = 0;
    std::size_t dsub_ = 0;

    // codebooks_ layout:
    // subquantizer m
    // centroid k
    // dimension d
    //
    // index = sub * ksub * dsub + centroid * dsub + d
    std::vector<float> codebooks_;
};

}