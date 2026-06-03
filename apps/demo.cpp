#include "ann/brute_force_index.hpp"
#include "ann/distance.hpp"
#include "ann/timer.hpp"

#include <iostream>
#include <vector>

int main() {
    const std::size_t num_vectors = 4;
    const std::size_t dim = 4;
    const std::size_t k = 2;

    std::vector<float> data = {
        1.0f, 2.0f, 1.0f, 2.0f,
        10.0f, 11.0f, 10.0f, 11.0f,
        2.0f, 1.0f, 2.0f, 1.0f,
        50.0f, 50.0f, 50.0f, 50.0f
    };

    std::vector<float> query = {
        1.5f, 1.8f, 1.5f, 1.8f
    };

    ann::BruteForceIndex index;
    index.build(data.data(), num_vectors, dim);

    index.save("brute_demo.index");

    ann::BruteForceIndex loaded_index;
    loaded_index.load("brute_demo.index");

    auto results = loaded_index.search(query.data(), k);

    std::cout << "Index: " << loaded_index.name() << " loaded from disk\n";
    std::cout << "Top " << k << " results:\n";

    for (const auto& result : results) {
        std::cout << "vector index = " << result.index
                  << ", distance = " << result.distance
                  << "\n";
    }

    // Correctness check on the small 4D demo vector.
    float scalar_distance = ann::l2_distance_squared(
        data.data(),
        query.data(),
        dim
    );

    float avx2_distance = ann::l2_distance_squared_avx2(
        data.data(),
        query.data(),
        dim
    );

    std::cout << "\nDistance check on 4D vector:\n";
    std::cout << "Scalar L2: " << scalar_distance << "\n";
    std::cout << "AVX2 L2: " << avx2_distance << "\n";

    return 0;
}