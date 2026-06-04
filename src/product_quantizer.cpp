#include "ann/product_quantizer.hpp"

#include "ann/distance.hpp"

#include <algorithm>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>
#include <istream>
#include <ostream>

namespace ann {

ProductQuantizer::ProductQuantizer(std::size_t m, std::size_t ksub)
    : m_(m),
      ksub_(ksub) {
    if (m_ == 0 || ksub_ == 0) {
        throw std::invalid_argument("ProductQuantizer parameters must be > 0");
    }

    if (ksub_ > 256) {
        throw std::invalid_argument("ksub must be <= 256 because codes use uint8_t");
    }
}

void ProductQuantizer::train(
    const float* data,
    std::size_t num_vectors,
    std::size_t dim,
    std::size_t iterations
) {
    if (data == nullptr) {
        throw std::invalid_argument("ProductQuantizer::train received null data");
    }

    if (num_vectors == 0 || dim == 0 || iterations == 0) {
        throw std::invalid_argument("num_vectors, dim, and iterations must be > 0");
    }

    if (dim % m_ != 0) {
        throw std::invalid_argument("dim must be divisible by m");
    }

    if (ksub_ > num_vectors) {
        throw std::invalid_argument("ksub cannot exceed num_vectors");
    }

    dim_ = dim;
    dsub_ = dim_ / m_;

    codebooks_.assign(m_ * ksub_ * dsub_, 0.0f);

    std::mt19937 rng(42);

    std::vector<float> new_centroids(ksub_ * dsub_, 0.0f);
    std::vector<std::size_t> counts(ksub_, 0);
    std::vector<std::size_t> assignments(num_vectors, 0);

    for (std::size_t sub = 0; sub < m_; ++sub) {
        std::uniform_int_distribution<std::size_t> pick(0, num_vectors - 1);

        // Initialize this subquantizer's codebook using random subvectors.
        for (std::size_t c = 0; c < ksub_; c++) {
            std::size_t idx = pick(rng);

            const float* vector = data + idx * dim_;
            const float* subvector = vector + sub * dsub_;

            float* centroid =
                codebooks_.data() + sub * ksub_ * dsub_ + c * dsub_;

            for (std::size_t d = 0; d < dsub_; ++d) {
                centroid[d] = subvector[d];
            }
        }

        // K-means inside this subspace.
        for (std::size_t iter = 0; iter < iterations; ++iter) {
            std::fill(new_centroids.begin(), new_centroids.end(), 0.0f);
            std::fill(counts.begin(), counts.end(), 0);

            for (std::size_t i = 0; i < num_vectors; ++i) {
                const float* vector = data + i * dim_;
                const float* subvector = vector + sub * dsub_;

                std::size_t best_centroid = 0;
                float best_distance = std::numeric_limits<float>::max();

                for (std::size_t c = 0; c < ksub_; ++c) {
                    const float* centroid =
                        codebooks_.data() + sub * ksub_ * dsub_ + c * dsub_;

                    float distance =
                        l2_distance_squared(subvector, centroid, dsub_);

                    if (distance < best_distance) {
                        best_distance = distance;
                        best_centroid = c;
                    }
                }

                assignments[i] = best_centroid;
                counts[best_centroid]++;

                float* sum = new_centroids.data() + best_centroid * dsub_;

                for (std::size_t d = 0; d < dsub_; ++d) {
                    sum[d] += subvector[d];
                }
            }

            for (std::size_t c = 0; c < ksub_; ++c) {
                if (counts[c] == 0) {
                    continue;
                }

                float* centroid =
                    codebooks_.data() + sub * ksub_ * dsub_ + c * dsub_;

                const float* sum = new_centroids.data() + c * dsub_;

                for (std::size_t d = 0; d < dsub_; ++d) {
                    centroid[d] = sum[d] / static_cast<float>(counts[c]);
                }
            }
        }
    }
}

std::vector<std::uint8_t> ProductQuantizer::encode(
    const float* data,
    std::size_t num_vectors
) const {
    if (data == nullptr) {
        throw std::invalid_argument("ProductQuantizer::encode received null data");
    }

    if (dim_ == 0 || dsub_ == 0 || codebooks_.empty()) {
        throw std::runtime_error("ProductQuantizer::encode called before train");
    }

    std::vector<std::uint8_t> codes(num_vectors * m_);

    for (std::size_t i = 0; i < num_vectors; ++i) {
        const float* vector = data + i * dim_;
        std::uint8_t* code = codes.data() + i * m_;

        for (std::size_t sub = 0; sub < m_; ++sub) {
            const float* subvector = vector + sub * dsub_;

            std::size_t best_centroid = 0;
            float best_distance = std::numeric_limits<float>::max();

            for (std::size_t c = 0; c < ksub_; ++c) {
                const float* centroid =
                    codebooks_.data() + sub * ksub_ * dsub_ + c * dsub_;

                float distance = l2_distance_squared(
                    subvector,
                    centroid,
                    dsub_
                );

                if (distance < best_distance) {
                    best_distance = distance;
                    best_centroid = c;
                }
            }

            code[sub] = static_cast<std::uint8_t>(best_centroid);
        }
    }

    return codes;
}


void ProductQuantizer::decode_vector(
    const std::uint8_t* code,
    float* output
) const {
    if (code == nullptr || output == nullptr) {
        throw std::invalid_argument("ProductQuantizer::decode_vector received null pointer");
    }

    if (dim_ == 0 || dsub_ == 0 || codebooks_.empty()) {
        throw std::runtime_error("ProductQuantizer::decode_vector called before train");
    }

    for (std::size_t sub = 0; sub < m_; ++sub) {
        std::size_t centroid_id = static_cast<std::size_t>(code[sub]);

        const float* centroid =
            codebooks_.data()
            + sub * ksub_ * dsub_
            + centroid_id * dsub_;

        float* output_subvector = output + sub * dsub_;

        for (std::size_t d = 0; d < dsub_; ++d) {
            output_subvector[d] = centroid[d];
        }
    }
}


std::vector<float> ProductQuantizer::compute_distance_table(
    const float* query
) const {
    if (query == nullptr) {
        throw std::invalid_argument("ProductQuantizer::compute_distance_table received null query");
    }

    if (dim_ == 0 || dsub_ == 0 || codebooks_.empty()) {
        throw std::runtime_error("ProductQuantizer::compute_distance_table called before train");
    }

    std::vector<float> table(m_ * ksub_);

    for (std::size_t sub = 0; sub < m_; ++sub) {
        const float* query_subvector = query + sub * dsub_;

        for (std::size_t c = 0; c < ksub_; ++c) {
            const float* centroid =
                codebooks_.data()
                + sub * ksub_ * dsub_
                + c * dsub_;

            table[sub * ksub_ + c] =
                l2_distance_squared(query_subvector, centroid, dsub_);
        }
    }

    return table;
}


float ProductQuantizer::adc_distance(
    const std::uint8_t* code,
    const std::vector<float>& distance_table
) const {
    if (code == nullptr) {
        throw std::invalid_argument("ProductQuantizer::adc_distance received null code");
    }

    if (distance_table.size() != m_ * ksub_) {
        throw std::invalid_argument("distance_table has incorrect size");
    }

    float distance = 0.0f;

    for (std::size_t sub = 0; sub < m_; ++sub) {
        std::size_t centroid_id = static_cast<std::size_t>(code[sub]);
        distance += distance_table[sub * ksub_ + centroid_id];
    }

    return distance;
}

std::size_t ProductQuantizer::memory_usage_bytes() const {
    return codebooks_.size() * sizeof(float);
}

void ProductQuantizer::save_state(std::ostream& out) const {
    out.write(reinterpret_cast<const char*>(&m_), sizeof(m_));
    out.write(reinterpret_cast<const char*>(&ksub_), sizeof(ksub_));
    out.write(reinterpret_cast<const char*>(&dim_), sizeof(dim_));
    out.write(reinterpret_cast<const char*>(&dsub_), sizeof(dsub_));

    std::size_t codebook_size = codebooks_.size();

    out.write(
        reinterpret_cast<const char*>(&codebook_size),
        sizeof(codebook_size)
    );

    out.write(
        reinterpret_cast<const char*>(codebooks_.data()),
        codebook_size * sizeof(float)
    );
}

void ProductQuantizer::load_state(std::istream& in) {
    in.read(reinterpret_cast<char*>(&m_), sizeof(m_));
    in.read(reinterpret_cast<char*>(&ksub_), sizeof(ksub_));
    in.read(reinterpret_cast<char*>(&dim_), sizeof(dim_));
    in.read(reinterpret_cast<char*>(&dsub_), sizeof(dsub_));

    std::size_t codebook_size = 0;

    in.read(
        reinterpret_cast<char*>(&codebook_size),
        sizeof(codebook_size)
    );

    codebooks_.resize(codebook_size);

    in.read(
        reinterpret_cast<char*>(codebooks_.data()),
        codebook_size * sizeof(float)
    );
}

}