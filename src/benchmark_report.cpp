#include "ann/benchmark_report.hpp"

#include <fstream>
#include <stdexcept>

namespace ann {

void write_benchmark_csv(
    const std::string& path,
    const std::vector<BenchmarkRow>& rows
) {
    std::ofstream file(path);

    if (!file) {
        throw std::runtime_error("Failed to open benchmark CSV file: " + path);
    }

    file << "index_name,nlist,nprobe,pq_m,recall_at_k,avg_ms,p50_ms,p95_ms,qps,memory_bytes\n";

    for (const auto& row : rows) {
        file << row.index_name << ","
             << row.nlist << ","
             << row.nprobe << ","
             << row.pq_m << ","
             << row.recall_at_k << ","
             << row.avg_ms << ","
             << row.p50_ms << ","
             << row.p95_ms << ","
             << row.qps << ","
             << row.memory_bytes << "\n";
    }
}

}