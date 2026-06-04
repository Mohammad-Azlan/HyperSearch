# HyperSearch

A high-performance approximate nearest neighbor (ANN) vector search engine written from scratch in modern C++.

This project implements exact and approximate vector search methods used in retrieval systems, recommendation systems, semantic search, and vector databases. The goal is to study and build the core infrastructure behind FAISS-style vector search engines, with emphasis on algorithmic depth, memory efficiency, multithreading, and benchmarking discipline.

## Highlights

- Built a vector search engine from scratch in modern C++
- Implemented IVF, SQ8, PQ, IVF-SQ, and IVF-PQ
- Added AVX2 SIMD-optimized L2 distance kernels
- Benchmarked on the full **SIFT1M** dataset (1 million vectors)
- Achieved **73x latency speedup** at low-recall IVF settings
- Achieved **0.989 Recall@10 at 5.6x speedup**
- Achieved **12.7x memory reduction** with IVF-PQ while maintaining sub-2 ms query latency
- Reached **2064 queries/sec** with multithreaded IVF-PQ
- Added binary index serialization and persistence
- Save trained indexes and reload without retraining

## Current Features

- Exact brute-force k-nearest-neighbor search
- Squared L2 distance kernel
- Heap-based top-k selection
- Abstract `Index` interface
- AVX2 SIMD-optimized distance kernels
- IVF-Flat index with k-means clustering
- Scalar Quantization (SQ8)
- IVF-SQ8 index
- Product Quantization (PQ)
- IVF-PQ index with asymmetric distance computation (ADC)
- Recall@k evaluation
- P50 / P95 latency benchmarking
- Memory usage reporting
- Multithreaded batch throughput benchmark
- CSV benchmark export
- SIFT1M `.fvecs` / `.ivecs` dataset loader
### Persistence
- BruteForce index serialization
- IVF-Flat index serialization
- IVF-SQ index serialization
- IVF-PQ index serialization
- Binary save/load support for quantizer state and ANN structures

#### Example
```
ann::IVFPQIndex index(...);

index.build(data.data(), num_vectors, dim);
index.save("sift_ivfpq.index");

ann::IVFPQIndex loaded(...);
loaded.load("sift_ivfpq.index");

auto results = loaded.search(query.data(), 10);
```

## Benchmark Setup

Final benchmark configuration:

- Dataset: SIFT1M
- Base vectors: 1,000,000
- Query vectors: 100
- Dimension: 128
- Metric: Squared L2 distance
- SIMD Acceleration: AVX2
- k: 10
- Build type: Release
- Compiler: GCC 14.2.0 (MSYS2 UCRT64)
- Platform: Windows
- CPU: 10-core laptop CPU

## Benchmark Results

### Brute Force Baseline

| Index | Recall@10 | Avg Latency | P50 | P95 | Memory |
|---|---:|---:|---:|---:|---:|
| BruteForce | 1.000 | 29.16 ms | 28.30 ms | 33.06 ms | 512 MB |

### Quantized Indexes

| Index | Recall@10 | Avg Latency | P50 | P95 | Memory | Memory Reduction |
|---|---:|---:|---:|---:|---:|---:|
| SQ-BruteForce | 0.990 | 19.99 ms | 19.30 ms | 21.84 ms | 128 MB | 4.0x |
| PQ-BruteForce | 0.700 | 13.06 ms | 13.02 ms | 13.64 ms | 32.1 MB | 15.9x |
| IVF-SQ8 | 0.990 | 22.97 ms | 23.05 ms | 25.94 ms | 136.1 MB | 3.76x |
| IVF-PQ (nprobe=8) | 0.693 | 1.70 ms | 1.65 ms | 2.46 ms | 40.3 MB | 12.7x |

### IVF-Flat Sweep

| nlist | nprobe | Recall@10 | Avg Latency | Speedup vs Brute |
|---:|---:|---:|---:|---:|
| 256 | 1 | 0.535 | 0.40 ms | 73.10x |
| 256 | 2 | 0.708 | 0.72 ms | 40.29x |
| 256 | 4 | 0.871 | 1.40 ms | 20.81x |
| 256 | 8 | 0.965 | 2.69 ms | 10.84x |
| 256 | 16 | 0.989 | 5.23 ms | 5.58x |
| 256 | 32 | 0.998 | 10.14 ms | 2.88x |
| 256 | 64 | 0.999 | 19.88 ms | 1.47x |
| 256 | 128 | 0.999 | 39.35 ms | 0.74x |

### Multithreaded IVF-PQ Throughput (nprobe=8)

| Threads | QPS |
|---:|---:|
| 1 | 577.94 |
| 2 | 1010.69 |
| 4 | 1225.35 |
| 8 | 2028.95 |
| 10 | 2064.32 |


## Benchmark Plots

### Recall-Latency Scatter Plot

![Recall Latency Scatter Plot](results/recall_vs_latency.png)

This plot shows the ANN tradeoff between Recall@k (here k=10) and latency.

### IVF-Flat Recall-Latency Tradeoff

![IVF Recall Latency Curve](results/ivf_recall_latency_curve.png)

This plot shows the core ANN tradeoff: increasing `nprobe` improves Recall@10 but increases query latency.

### Memory Usage

![Memory Usage](results/memory_usage.png)

Quantized indexes significantly reduce memory usage. PQ-based indexes provide the largest compression.

### Parallel Throughput Scaling

![QPS Scaling](results/qps_scaling.png)

IVF-PQ parallel batch search scales with thread count on the full SIFT1M benchmark.

## Key Results

On the full SIFT1M benchmark:

- IVF-Flat demonstrated the classic ANN recall-latency tradeoff, reaching **0.965 Recall@10 at 2.69 ms/query (10.84x speedup over brute-force)** and **0.989 Recall@10 at 5.23 ms/query (5.58x speedup over brute-force)**.
- IVF-PQ achieved **12.7x memory reduction** with **1.70 ms average latency** and **0.693 Recall@10** , providing a strong memory-latency tradeoff for compressed ANN search.
- PQ-BruteForce achieved **15.9x memory reduction** while remaining **2.23x faster than brute force**.
- IVF-PQ multithreaded batch search reached **2064 QPS** on 10 CPU cores.

## SIMD Optimizations

HyperSearch includes AVX2 vectorized L2 distance kernels for float-vector search.

Compared to the original scalar implementation on SIFT1M:

| Index | Scalar | AVX2 | Speedup |
|---|---:|---:|---:|
| BruteForce | 50.73 ms | 29.16 ms | 1.74x |
| SQ-BruteForce | 58.76 ms | 19.99 ms | 2.94x |
| IVF-Flat (nprobe=16) | 11.51 ms | 5.23 ms | 2.20x |

The SIMD implementation uses AVX2 256-bit vector instructions to accelerate squared L2 distance computation across 128-dimensional vectors.

### Current Version

HyperSearch v3.0

Major additions:
- Added index serialization and persistence
- Added save/load support for IVF, IVF-SQ, and IVF-PQ
- Validated serialized indexes on SIFT1M

## Version History

### v3.0
- Added index serialization and persistence
- Added save/load support for IVF, IVF-SQ, and IVF-PQ
- Validated serialized indexes on SIFT1M

### v2.0
- Added AVX2 SIMD acceleration
- Improved search latency by up to ~3x

### v1.0
- Initial ANN engine release
- IVF, SQ, PQ, IVF-SQ, IVF-PQ
- SIFT1M benchmarking

## Architecture

The project is organized into reusable modules:

```text
apps/
include/ann/
src/
scripts/
results/
```

## Core Components

- `Index`: common abstract interface for all indexes
- `BruteForceIndex`: exact search baseline
- `IVFIndex`: inverted file index
- `ScalarQuantizer`: per-dimension SQ8 compression
- `SQBruteForceIndex`: brute-force search over SQ8 compressed vectors
- `IVFSQIndex`: IVF search over SQ8 compressed vectors
- `ProductQuantizer`: PQ training, encoding, decoding, and ADC distance
- `PQBruteForceIndex`: brute-force search over PQ codes
- `IVFPQIndex`: IVF search over PQ codes
- `benchmark`: latency benchmark utilities
- `parallel_benchmark`: multithreaded throughput benchmark
- `evaluation`: Recall@k evaluation
- `dataset`: `.fvecs` and `.ivecs` dataset loading
- `benchmark_report`: CSV benchmark output

## Build

This project uses CMake and Ninja.

```bash
cmake -B build-release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-release
.\build-release\ann_demo.exe
.\build-release\ann_benchmark.exe
```

## Dataset

The project supports SIFT1M `.fvecs` and `.ivecs` files.

Expected layout:

```text
data/sift1m/sift_base.fvecs
data/sift1m/sift_query.fvecs
data/sift1m/sift_groundtruth.ivecs
data/sift1m/sift_learn.fvecs
```

## Current Limitations

- Ground truth is computed using brute-force search during evaluation
- IVF/PQ training is single-threaded
- No Python bindings
- No graph-based ANN index (e.g. HNSW)

## Roadmap

Planned next steps:

- Python bindings via pybind11
- HNSW graph-based ANN index
- Multi-threaded index training
- Additional benchmark datasets

## Project Goal

The goal of this project is to demonstrate systems-level AI infrastructure ability: implementing retrieval algorithms from scratch, optimizing memory layout, measuring recall-latency tradeoffs, and building a benchmarkable C++ vector search engine.
