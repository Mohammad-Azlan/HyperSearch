#include "ann/brute_force_index.hpp"
#include "ann/distance.hpp"
#include "ann/timer.hpp"
#include "ann/ivf_index.hpp"
#include "ann/ivf_sq_index.hpp"
#include "ann/ivf_pq_index.hpp"
#include "ann/hnsw_index.hpp"

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

    std::cout << "\nIVF serialization test:\n";

    ann::IVFIndex ivf(2, 1, 3);
    ivf.build(data.data(), num_vectors, dim);
    ivf.save("ivf_demo.index");

    ann::IVFIndex loaded_ivf(1, 1, 1);
    loaded_ivf.load("ivf_demo.index");

    auto ivf_results = loaded_ivf.search(query.data(), k);

    std::cout << "Index: " << loaded_ivf.name()
            << " loaded from disk\n";

    for (const auto& result : ivf_results) {
        std::cout << "vector index = " << result.index
                << ", distance = " << result.distance
                << "\n";
    }

    std::cout << "\nIVF-SQ serialization test:\n";

    ann::IVFSQIndex ivf_sq(2, 1, 3);
    ivf_sq.build(data.data(), num_vectors, dim);
    ivf_sq.save("ivf_sq_demo.index");

    ann::IVFSQIndex loaded_ivf_sq(1, 1, 1);
    loaded_ivf_sq.load("ivf_sq_demo.index");

    auto ivf_sq_results = loaded_ivf_sq.search(query.data(), k);

    std::cout << "Index: " << loaded_ivf_sq.name()
            << " loaded from disk\n";

    for (const auto& result : ivf_sq_results) {
        std::cout << "vector index = " << result.index
                << ", distance = " << result.distance
                << "\n";
    }

    std::cout << "\nIVF-PQ serialization test:\n";

    ann::IVFPQIndex ivf_pq(
        2,  // nlist
        1,  // nprobe
        3,  // kmeans iterations
        2,  // pq_m
        2,  // pq_ksub
        3   // pq_iterations
    );

    ivf_pq.build(data.data(), num_vectors, dim);
    ivf_pq.save("ivf_pq_demo.index");

    ann::IVFPQIndex loaded_ivf_pq(
        1, 1, 1,
        1, 2, 1
    );

    loaded_ivf_pq.load("ivf_pq_demo.index");

    auto ivf_pq_results = loaded_ivf_pq.search(query.data(), k);

    std::cout << "Index: " << loaded_ivf_pq.name()
            << " loaded from disk\n";

    for (const auto& result : ivf_pq_results) {
        std::cout << "vector index = " << result.index
                << ", distance = " << result.distance
                << "\n";
    }

    std::cout << "\nHNSW serialization-free search test:\n";

    ann::HNSWIndex hnsw(2,4, 4);
    hnsw.build(data.data(), num_vectors, dim);

    auto hnsw_results = hnsw.search(query.data(), k);

    std::cout << "Index: " << hnsw.name() << "\n";

    for (const auto& result : hnsw_results) {
        std::cout << "vector index = " << result.index
                << ", distance = " << result.distance
                << "\n";
    }

    return 0;
}