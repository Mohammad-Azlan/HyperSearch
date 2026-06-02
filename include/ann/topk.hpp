#pragma once

#include "ann/index.hpp"

#include <algorithm>
#include <queue>
#include <vector>

namespace ann {

class TopK {
public:
    explicit TopK(std::size_t k) : k_(k) {}

    void push(std::size_t index, float distance) {
        if (k_ == 0) {
            return;
        }

        SearchResult result{index, distance};

        if (heap_.size() < k_) {
            heap_.push(result);
        } else if (distance < heap_.top().distance) {
            heap_.pop();
            heap_.push(result);
        }
    }

    std::vector<SearchResult> results() {
        std::vector<SearchResult> output;

        while (!heap_.empty()) {
            output.push_back(heap_.top());
            heap_.pop();
        }

        std::sort(output.begin(), output.end(),
                  [](const SearchResult& a, const SearchResult& b) {
                      return a.distance < b.distance;
                  });

        return output;
    }

private:
    struct WorseFirst {
        bool operator()(const SearchResult& a, const SearchResult& b) const {
            return a.distance < b.distance;
        }
    };

    std::size_t k_;
    std::priority_queue<SearchResult, std::vector<SearchResult>, WorseFirst> heap_;
};

}