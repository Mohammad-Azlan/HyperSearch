#include "ann/evaluation.hpp"

#include <algorithm>
#include <stdexcept>
#include <unordered_set>

namespace ann {

double recall_at_k(
    const std::vector<SearchResult>& ground_truth,
    const std::vector<SearchResult>& approximate,
    std::size_t k
) {
    if (k == 0) {
        throw std::invalid_argument("k must be greater than 0");
    }

    std::unordered_set<std::size_t> truth_ids;

    std::size_t truth_count = std::min(k, ground_truth.size());
    std::size_t approx_count = std::min(k, approximate.size());

    for (std::size_t i = 0; i < truth_count; ++i) {
        truth_ids.insert(ground_truth[i].index);
    }

    std::size_t matches = 0;

    for (std::size_t i = 0; i < approx_count; ++i) {
        if (truth_ids.find(approximate[i].index) != truth_ids.end()) {
            matches++;
        }
    }

    return static_cast<double>(matches) / static_cast<double>(truth_count);
}

}