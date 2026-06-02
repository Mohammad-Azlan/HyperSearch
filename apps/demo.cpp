#include "ann/brute_force_index.hpp"
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

    auto results = index.search(query.data(), k);

    std::cout << "Index: " << index.name() << "\n";
    std::cout << "Top " << k << " results:\n";

    for (const auto& result : results) {
        std::cout << "vector index = " << result.index
                  << ", approximate distance = " << result.distance
                  << "\n";
    }

    return 0;
}