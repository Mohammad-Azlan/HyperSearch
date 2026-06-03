#include "ann/ivf_index.hpp"

#include "ann/distance.hpp"
#include "ann/kmeans.hpp"
#include "ann/topk.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>
#include <fstream>

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

        float distance = l2_distance_squared_avx2(query, centroid, dim_);

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

            float distance = l2_distance_squared_avx2(query, vector_ptr, dim_);

            topk.push(vector_idx, distance);
        }
    }

    return topk.results();
}

void IVFIndex::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);

    if (!out) {
        throw std::runtime_error("Failed to open IVF index file for writing");
    }

    out.write(reinterpret_cast<const char*>(&nlist_), sizeof(nlist_));
    out.write(reinterpret_cast<const char*>(&nprobe_), sizeof(nprobe_));
    out.write(reinterpret_cast<const char*>(&kmeans_iterations_), sizeof(kmeans_iterations_));
    out.write(reinterpret_cast<const char*>(&num_vectors_), sizeof(num_vectors_));
    out.write(reinterpret_cast<const char*>(&dim_), sizeof(dim_));

    std::size_t data_size = data_.size();
    out.write(reinterpret_cast<const char*>(&data_size), sizeof(data_size));
    out.write(reinterpret_cast<const char*>(data_.data()), data_size * sizeof(float));

    std::size_t centroids_size = centroids_.size();
    out.write(reinterpret_cast<const char*>(&centroids_size), sizeof(centroids_size));
    out.write(reinterpret_cast<const char*>(centroids_.data()), centroids_size * sizeof(float));

    std::size_t num_lists = inverted_lists_.size();
    out.write(reinterpret_cast<const char*>(&num_lists), sizeof(num_lists));

    for (const auto& list : inverted_lists_) {
        std::size_t list_size = list.size();
        out.write(reinterpret_cast<const char*>(&list_size), sizeof(list_size));
        out.write(reinterpret_cast<const char*>(list.data()), list_size * sizeof(std::size_t));
    }
}

void IVFIndex::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);

    if (!in) {
        throw std::runtime_error("Failed to open IVF index file for reading");
    }

    in.read(reinterpret_cast<char*>(&nlist_), sizeof(nlist_));
    in.read(reinterpret_cast<char*>(&nprobe_), sizeof(nprobe_));
    in.read(reinterpret_cast<char*>(&kmeans_iterations_), sizeof(kmeans_iterations_));
    in.read(reinterpret_cast<char*>(&num_vectors_), sizeof(num_vectors_));
    in.read(reinterpret_cast<char*>(&dim_), sizeof(dim_));

    std::size_t data_size = 0;
    in.read(reinterpret_cast<char*>(&data_size), sizeof(data_size));
    data_.resize(data_size);
    in.read(reinterpret_cast<char*>(data_.data()), data_size * sizeof(float));

    std::size_t centroids_size = 0;
    in.read(reinterpret_cast<char*>(&centroids_size), sizeof(centroids_size));
    centroids_.resize(centroids_size);
    in.read(reinterpret_cast<char*>(centroids_.data()), centroids_size * sizeof(float));

    std::size_t num_lists = 0;
    in.read(reinterpret_cast<char*>(&num_lists), sizeof(num_lists));
    inverted_lists_.resize(num_lists);

    for (auto& list : inverted_lists_) {
        std::size_t list_size = 0;
        in.read(reinterpret_cast<char*>(&list_size), sizeof(list_size));

        list.resize(list_size);
        in.read(reinterpret_cast<char*>(list.data()), list_size * sizeof(std::size_t));
    }
}

}