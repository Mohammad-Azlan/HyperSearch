#include "ann/benchmark.hpp"
#include "ann/brute_force_index.hpp"
#include "ann/ivf_index.hpp"
#include "ann/timer.hpp"
#include "ann/evaluation.hpp"
#include "ann/scalar_quantizer.hpp"
#include "ann/sq_brute_force_index.hpp"
#include "ann/ivf_sq_index.hpp"
#include "ann/product_quantizer.hpp"
#include "ann/pq_brute_force_index.hpp"
#include "ann/ivf_pq_index.hpp"
#include "ann/parallel_benchmark.hpp"
#include "ann/benchmark_report.hpp"
#include "ann/dataset.hpp"

#include <iostream>
#include <random>
#include <vector>
#include <fstream>
#include <algorithm>

void print_benchmark(
    const ann::Index& index,
    const ann::BenchmarkResult& benchmark,
    std::size_t num_vectors,
    std::size_t dim,
    std::size_t num_queries,
    std::size_t k
) {
    std::cout << "\nBenchmark summary:\n";
    std::cout << "Index type: " << index.name() << "\n";
    std::cout << "Database vectors: " << num_vectors << "\n";
    std::cout << "Dimension: " << dim << "\n";
    std::cout << "Queries: " << num_queries << "\n";
    std::cout << "k: " << k << "\n";
    std::cout << "Average search time: " << benchmark.avg_ms << " ms\n";
    std::cout << "Min search time: " << benchmark.min_ms << " ms\n";
    std::cout << "Max search time: " << benchmark.max_ms << " ms\n";
    std::cout << "P50 search time: " << benchmark.p50_ms << " ms\n";
    std::cout << "P95 search time: " << benchmark.p95_ms << " ms\n";
}


double evaluate_recall(
    const ann::Index& exact_index,
    const ann::Index& approx_index,
    const float* queries,
    std::size_t num_queries,
    std::size_t dim,
    std::size_t k
) {
    double total_recall = 0.0;

    for (std::size_t q = 0; q < num_queries; ++q) {
        const float* query_ptr = queries + q * dim;

        auto exact_results = exact_index.search(query_ptr, k);
        auto approx_results = approx_index.search(query_ptr, k);

        total_recall += ann::recall_at_k(
            exact_results,
            approx_results,
            k
        );
    }

    return total_recall / static_cast<double>(num_queries);
}

void print_memory_stats(
    std::size_t float_memory,
    std::size_t compressed_memory
) {
    std::cout << "Float memory: " << float_memory << " bytes\n";
    std::cout << "Compressed memory: " << compressed_memory << " bytes\n";
    std::cout << "Memory reduction: "
              << static_cast<double>(float_memory) /
                     static_cast<double>(compressed_memory)
              << "x\n";
}

int main() {
    // const std::size_t num_vectors = 100000;
    // const std::size_t dim = 128;
    // const std::size_t k = 10;
    // const std::size_t num_queries = 100;

    const bool run_brute = true;
    const bool run_sq_brute = true;
    const bool run_pq_brute = true;
    const bool run_ivf_pq = true;
    const bool run_ivf_sq = true;
    const bool run_ivf_flat_sweep = true;
    const bool run_parallel_ivf_pq = true;

    const std::string base_path = "data/sift1m/sift_base.fvecs";
    const std::string query_path = "data/sift1m/sift_query.fvecs";

    const std::size_t max_base_vectors = 1000000;
    const std::size_t max_query_vectors = 100;
    const std::size_t k = 10;

    const std::size_t nlist = 256;
    const std::size_t ivf_pq_nprobe = 8;
    const std::size_t ivf_sq_nprobe = 64;

    const std::size_t pq_m = 32;
    const std::size_t pq_ksub = 256;

    const std::size_t ivf_kmeans_iterations = 10;
    const std::size_t pq_iterations = 10;

    std::vector<ann::BenchmarkRow> report_rows;

    
    auto base = ann::load_fvecs(base_path, max_base_vectors);
    auto query = ann::load_fvecs(query_path, max_query_vectors);

    std::vector<float> data = std::move(base.data);
    std::vector<float> queries = std::move(query.data);

    const std::size_t num_vectors = base.num_vectors;
    const std::size_t dim = base.dim;
    const std::size_t num_queries = query.num_vectors;

    std::cout << "Loaded base vectors: " << base.num_vectors
            << ", dim: " << base.dim << "\n";

    std::cout << "Loaded query vectors: " << query.num_vectors
            << ", dim: " << query.dim << "\n";
    

    // std::vector<float> data(num_vectors * dim);
    // std::vector<float> queries(num_queries * dim);

    // std::mt19937 rng(42);
    // std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    // for (float& x : data) {
    //     x = dist(rng);
    // }

    // for (float& x : queries) {
    //     x = dist(rng);
    // }


    ann::BruteForceIndex brute;
    ann::BenchmarkResult brute_bench{};

    if (run_brute) {
        {
            ann::Timer timer;
            brute.build(data.data(), num_vectors, dim);
            std::cout << "Brute build time: " << timer.elapsed_ms() << " ms\n";
        }

        brute_bench = ann::benchmark_search(
            brute,
            queries.data(),
            num_queries,
            dim,
            k
        );

        print_benchmark(brute, brute_bench, num_vectors, dim, num_queries, k);

        report_rows.push_back({
            brute.name(),
            0,
            0,
            0,
            1.0,
            brute_bench.avg_ms,
            brute_bench.p50_ms,
            brute_bench.p95_ms,
            0.0,
            data.size() * sizeof(float)
        });
    }

    if (run_sq_brute) {
        ann::SQBruteForceIndex sq_brute;

        {
            ann::Timer timer;
            sq_brute.build(data.data(), num_vectors, dim);
            std::cout << "SQ brute build time: " << timer.elapsed_ms() << " ms\n";
        }

        auto sq_bench = ann::benchmark_search(
            sq_brute,
            queries.data(),
            num_queries,
            dim,
            k
        );

        print_benchmark(sq_brute, sq_bench, num_vectors, dim, num_queries, k);

        double sq_avg_recall = evaluate_recall(
            brute,
            sq_brute,
            queries.data(),
            num_queries,
            dim,
            k
        );


        std::cout << "\nSQ-BruteForce evaluation:\n";
        std::cout << "Recall@" << k << ": " << sq_avg_recall << "\n";
        print_memory_stats(data.size() * sizeof(float), sq_brute.memory_usage_bytes());

        report_rows.push_back({
            sq_brute.name(),
            0,
            0,
            0,
            sq_avg_recall,
            sq_bench.avg_ms,
            sq_bench.p50_ms,
            sq_bench.p95_ms,
            0.0,
            sq_brute.memory_usage_bytes()
        });
    }

    if (run_pq_brute) {
        ann::PQBruteForceIndex pq_brute(
            pq_m,
            pq_ksub,
            pq_iterations
        );

        {
            ann::Timer timer;
            pq_brute.build(data.data(), num_vectors, dim);
            std::cout << "PQ brute build time: "
                    << timer.elapsed_ms()
                    << " ms\n";
        }

        auto pq_bench = ann::benchmark_search(
            pq_brute,
            queries.data(),
            num_queries,
            dim,
            k
        );

        print_benchmark(pq_brute, pq_bench, num_vectors, dim, num_queries, k);


        double pq_avg_recall = evaluate_recall(
            brute,
            pq_brute,
            queries.data(),
            num_queries,
            dim,
            k
        );

        std::cout << "\nPQ-BruteForce evaluation:\n";
        std::cout << "Recall@" << k << ": "
                << pq_avg_recall << "\n";

        print_memory_stats(
            data.size() * sizeof(float),
            pq_brute.memory_usage_bytes()
        );

        report_rows.push_back({
            pq_brute.name(),
            0,
            0,
            32,
            pq_avg_recall,
            pq_bench.avg_ms,
            pq_bench.p50_ms,
            pq_bench.p95_ms,
            0.0,
            pq_brute.memory_usage_bytes()
        });
    }

    const std::vector<std::size_t> ivf_pq_nprobe_values = {4, 8, 16, 32, 64, 128};

    if (run_ivf_pq) {
        ann::IVFPQIndex ivf_pq(
            nlist,
            1,
            ivf_kmeans_iterations,
            pq_m,
            pq_ksub,
            pq_iterations
        );

        {
            ann::Timer timer;
            ivf_pq.build(data.data(), num_vectors, dim);
            std::cout << "\nIVF-PQ build time: "
                    << timer.elapsed_ms()
                    << " ms\n";
        }

        for (std::size_t current_nprobe : ivf_pq_nprobe_values) {
            ivf_pq.set_nprobe(current_nprobe);

            auto ivf_pq_bench = ann::benchmark_search(
                ivf_pq,
                queries.data(),
                num_queries,
                dim,
                k
            );

            print_benchmark(
                ivf_pq,
                ivf_pq_bench,
                num_vectors,
                dim,
                num_queries,
                k
            );

            double ivf_pq_avg_recall = evaluate_recall(
                brute,
                ivf_pq,
                queries.data(),
                num_queries,
                dim,
                k
            );

            std::cout << "\nIVF-PQ sweep result:\n";
            std::cout << "nlist: " << nlist << "\n";
            std::cout << "nprobe: " << current_nprobe << "\n";
            std::cout << "Recall@" << k << ": "
                    << ivf_pq_avg_recall << "\n";

            print_memory_stats(
                data.size() * sizeof(float),
                ivf_pq.memory_usage_bytes()
            );

            double best_qps = 0.0;

            if (run_parallel_ivf_pq && current_nprobe == 8) {
                std::vector<std::size_t> thread_counts = {1, 2, 4, 8, 10};

                std::ofstream throughput_file("throughput_results.csv");
                throughput_file << "index_name,nprobe,threads,total_ms,qps\n";

                std::cout << "\nParallel throughput benchmark for IVF-PQ"
                        << " nprobe=" << current_nprobe << ":\n";

                for (std::size_t threads : thread_counts) {
                    auto parallel_result = ann::benchmark_search_parallel(
                        ivf_pq,
                        queries.data(),
                        num_queries,
                        dim,
                        k,
                        threads
                    );

                    best_qps = std::max(best_qps, parallel_result.qps);

                    std::cout << "Threads: " << threads
                            << ", Total time: " << parallel_result.total_ms << " ms"
                            << ", QPS: " << parallel_result.qps
                            << "\n";

                    throughput_file << ivf_pq.name() << ","
                                    << current_nprobe << ","
                                    << threads << ","
                                    << parallel_result.total_ms << ","
                                    << parallel_result.qps << "\n";
                }
            }

            report_rows.push_back({
                ivf_pq.name(),
                nlist,
                current_nprobe,
                pq_m,
                ivf_pq_avg_recall,
                ivf_pq_bench.avg_ms,
                ivf_pq_bench.p50_ms,
                ivf_pq_bench.p95_ms,
                best_qps,
                ivf_pq.memory_usage_bytes()
            });
        }
    }


    if (run_ivf_sq) {
        ann::IVFSQIndex ivf_sq(
            nlist,
            ivf_sq_nprobe,
            ivf_kmeans_iterations
        );

        {
            ann::Timer timer;
            ivf_sq.build(data.data(), num_vectors, dim);
            std::cout << "\nIVF-SQ build time: "
                    << timer.elapsed_ms()
                    << " ms\n";
        }

        auto ivf_sq_bench = ann::benchmark_search(
            ivf_sq,
            queries.data(),
            num_queries,
            dim,
            k
        );

        print_benchmark(ivf_sq, ivf_sq_bench, num_vectors, dim, num_queries, k);


        double ivf_sq_avg_recall = evaluate_recall(brute,ivf_sq,queries.data(),num_queries,dim,k);

        std::cout << "\nIVF-SQ evaluation:\n";
        std::cout << "Recall@" << k << ": "
                << ivf_sq_avg_recall << "\n";

        print_memory_stats(
            data.size() * sizeof(float),
            ivf_sq.memory_usage_bytes()
        );

        report_rows.push_back({
            ivf_sq.name(),
            nlist,
            ivf_sq_nprobe,
            0,
            ivf_sq_avg_recall,
            ivf_sq_bench.avg_ms,
            ivf_sq_bench.p50_ms,
            ivf_sq_bench.p95_ms,
            0.0,
            ivf_sq.memory_usage_bytes()
        });
    }

    std::vector<std::size_t> nlist_values = {256};
    std::vector<std::size_t> nprobe_values = {1, 2, 4, 8, 16, 32, 64, 128};

    if (run_ivf_flat_sweep) {
        for (std::size_t nlist_value : nlist_values) {
            ann::IVFIndex ivf(nlist_value , 1, ivf_kmeans_iterations);

            {
                ann::Timer timer;
                ivf.build(data.data(), num_vectors, dim);
                std::cout << "\n====================================\n";
                std::cout << "IVF build complete\n";
                std::cout << "nlist: " << nlist_value  << "\n";
                std::cout << "Build time: " << timer.elapsed_ms() << " ms\n";
            }

            for (std::size_t nprobe : nprobe_values) {
                if (nprobe > nlist_value ) {
                    continue;
                }

                ivf.set_nprobe(nprobe);

                auto ivf_bench = ann::benchmark_search(
                    ivf,
                    queries.data(),
                    num_queries,
                    dim,
                    k
                );



                double avg_recall =evaluate_recall(brute,ivf,queries.data(),num_queries,dim,k);

                double speedup = brute_bench.avg_ms / ivf_bench.avg_ms;

                std::cout << "\nIVF-Flat sweep result:\n";
                std::cout << "nlist: " << nlist_value  << "\n";
                std::cout << "nprobe: " << nprobe << "\n";
                std::cout << "Average search time: " << ivf_bench.avg_ms << " ms\n";
                std::cout << "P50 search time: " << ivf_bench.p50_ms << " ms\n";
                std::cout << "P95 search time: " << ivf_bench.p95_ms << " ms\n";
                std::cout << "Recall@" << k << ": " << avg_recall << "\n";
                std::cout << "Speedup over brute force: " << speedup << "x\n";

                report_rows.push_back({
                    ivf.name(),
                    nlist_value,
                    nprobe,
                    0,
                    avg_recall,
                    ivf_bench.avg_ms,
                    ivf_bench.p50_ms,
                    ivf_bench.p95_ms,
                    0.0,
                    data.size() * sizeof(float)
                });
            }
        }
    }

    ann::write_benchmark_csv("benchmark_results.csv", report_rows);

    std::cout << "\nWrote benchmark_results.csv\n";

    return 0;
}
