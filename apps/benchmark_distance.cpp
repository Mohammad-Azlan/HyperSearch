#include "ann/brute_force_index.hpp"
#include "ann/distance.hpp"
#include "ann/timer.hpp"

#include <iostream>
#include <vector>

int main() {
    // SIMD performance check on realistic 128D vectors.
    const std::size_t simd_dim = 128;
    const std::size_t repeats = 1000000;

    std::vector<float> simd_a(simd_dim, 1.0f);
    std::vector<float> simd_b(simd_dim, 2.0f);

    {
        ann::Timer timer;
        volatile float sink = 0.0f;

        for (std::size_t i = 0; i < repeats; ++i) {
            sink += ann::l2_distance_squared(
                simd_a.data(),
                simd_b.data(),
                simd_dim
            );
        }

        std::cout << "\nScalar 128D distance benchmark: "
                  << timer.elapsed_ms()
                  << " ms\n";
    }

    {
        ann::Timer timer;
        volatile float sink = 0.0f;

        for (std::size_t i = 0; i < repeats; ++i) {
            sink += ann::l2_distance_squared_avx2(
                simd_a.data(),
                simd_b.data(),
                simd_dim
            );
        }

        std::cout << "AVX2 128D distance benchmark: "
                  << timer.elapsed_ms()
                  << " ms\n";
    }

    return 0;
}