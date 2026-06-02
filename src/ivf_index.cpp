#include "ann/ivf_index.hpp"

#include "ann/distance.hpp"
#include "ann/kmeans.hpp"
#include "ann/topk.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace ann {

IVFIndex::IVFIndex(
    std::size_t nlist,
    std::size_t nprobe,
    std::size_t kmeans_iterations
)
    : nlist_(nlist),
      nprobe_(nprobe),
      kmeans_iterations_(kmeans_iterations) {
    if (nlist_ == 0 || nprobe_ == 0 || kmeans_iterations_ == 0) {
        throw std::invalid_argument("IVF parameters must be > 0");
    }

    if (nprobe_ > nlist_) {
        throw std::invalid_argument("nprobe cannot exceed nlist");
    }
}

void IVFIndex::set_nprobe(std::size_t nprobe) {
    if (nprobe == 0) {
        throw std::invalid_argument("nprobe must be > 0");
    }

    if (nprobe > nlist_) {
        throw std::invalid_argument("nprobe cannot exceed nlist");
    }

    nprobe_ = nprobe;
}

void IVFIndex::build(
    const float* data,
    std::size_t num_vectors,
    std::size_t dim
) {
    if (data == nullptr) {
        throw std::invalid_argument("IVF build received null data");
    }

    if (num_vectors == 0 || dim == 0) {
        throw std::invalid_argument("num_vectors and dim must be > 0");
    }

    num_vectors_ = num_vectors;
    dim_ = dim;

    data_.assign(data, data + num_vectors * dim);

    auto km = run_kmeans(
        data_.data(),
        num_vectors_,
        dim_,
        nlist_,
        kmeans_iterations_
    );

    centroids_ = std::move(km.centroids);

    inverted_lists_.clear();
    inverted_lists_.resize(nlist_);

    for (std::size_t i = 0; i < num_vectors_; ++i) {
        std::size_t cluster = km.assignments[i];
        inverted_lists_[cluster].push_back(i);
    }
}

std::vector<SearchResult> IVFIndex::search(
    const float* query,
    std::size_t k
) const {
    if (query == nullptr) {
        throw std::invalid_argument("IVF search received null query");
    }

    if (data_.empty()) {
        throw std::runtime_error("IVF search called before build");
    }

    std::vector<SearchResult> centroid_distances;

    for (std::size_t c = 0; c < nlist_; ++c) {
        const float* centroid = centroids_.data() + c * dim_;

        float distance = l2_distance_squared(query, centroid, dim_);

        centroid_distances.push_back({c, distance});
    }

    std::sort(
        centroid_distances.begin(),
        centroid_distances.end(),
        [](const SearchResult& a, const SearchResult& b) {
            return a.distance < b.distance;
        }
    );

    TopK topk(k);

    for (std::size_t probe = 0; probe < nprobe_; ++probe) {
        std::size_t cluster = centroid_distances[probe].index;

        for (std::size_t vector_idx : inverted_lists_[cluster]) {
            const float* vector_ptr = data_.data() + vector_idx * dim_;

            float distance = l2_distance_squared(query, vector_ptr, dim_);

            topk.push(vector_idx, distance);
        }
    }

    return topk.results();
}

}