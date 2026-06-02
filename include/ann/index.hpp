#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace ann {

struct SearchResult {
    std::size_t index;
    float distance;
};

class Index {
public:
    virtual ~Index() = default;

    virtual void build(
        const float* data,
        std::size_t num_vectors,
        std::size_t dim
    ) = 0;

    virtual std::vector<SearchResult> search(
        const float* query,
        std::size_t k
    ) const = 0;

    virtual std::string name() const = 0;
};

}