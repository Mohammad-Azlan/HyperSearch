#pragma once

#include "ann/index.hpp"

#include <vector>

namespace ann {

double recall_at_k(
    const std::vector<SearchResult>& ground_truth,
    const std::vector<SearchResult>& approximate,
    std::size_t k
);

}