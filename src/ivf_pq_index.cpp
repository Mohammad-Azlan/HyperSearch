#include "ann/ivf_pq_index.hpp"

#include "ann/distance.hpp"
#include "ann/kmeans.hpp"
#include "ann/topk.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>
#include <fstream>

namespace ann {

IVFPQIndex::IVFPQIndex(
    std::size_t nlist,
    std::size_t nprobe,
    std::size_t kmeans_iterations,
    std::size_t pq_m,
    std::size_t pq_ksub,
    std::size_t pq_iterations
)
    : nlist_(nlist),
      nprobe_(nprobe),
      kmeans_iterations_(kmeans_iterations),
      pq_iterations_(pq_iterations),
      pq_(pq_m, pq_ksub) {
    if (nlist_ == 0 || nprobe_ == 0 ||
        kmeans_iterations_ == 0 || pq_iterations_ == 0) {
        throw std::invalid_argument("IVFPQIndex parameters must be > 0");
    }

    if (nprobe_ > nlist_) {
        throw std::invalid_argument("nprobe cannot exceed nlist");
    }
}

void IVFPQIndex::set_nprobe(std::size_t nprobe) {
    if (nprobe == 0) {
        throw std::invalid_argument("nprobe must be > 0");
    }

    if (nprobe > nlist_) {
        throw std::invalid_argument("nprobe cannot exceed nlist");
    }

    nprobe_ = nprobe;
}

void IVFPQIndex::build(
    const float* data,
    std::size_t num_vectors,
    std::size_t dim
) {
    if (data == nullptr) {
        throw std::invalid_argument("IVFPQIndex::build received null data");
    }

    if (num_vectors == 0 || dim == 0) {
        throw std::invalid_argument("num_vectors and dim must be > 0");
    }

    num_vectors_ = num_vectors;
    dim_ = dim;

    auto km = run_kmeans(
        data,
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

    pq_.train(data, num_vectors_, dim_, pq_iterations_);
    codes_ = pq_.encode(data, num_vectors_);
}

std::vector<SearchResult> IVFPQIndex::search(
    const float* query,
    std::size_t k
) const {
    if (query == nullptr) {
        throw std::invalid_argument("IVFPQIndex::search received null query");
    }

    if (codes_.empty()) {
        throw std::runtime_error("IVFPQIndex::search called before build");
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

    auto distance_table = pq_.compute_distance_table(query);

    TopK topk(k);

    for (std::size_t probe = 0; probe < nprobe_; ++probe) {
        std::size_t cluster = centroid_distances[probe].index;

        for (std::size_t vector_idx : inverted_lists_[cluster]) {
            const std::uint8_t* code =
                codes_.data() + vector_idx * pq_.code_size_bytes();

            float distance = pq_.adc_distance(code, distance_table);

            topk.push(vector_idx, distance);
        }
    }

    return topk.results();
}

std::size_t IVFPQIndex::memory_usage_bytes() const {
    std::size_t centroid_bytes = centroids_.size() * sizeof(float);
    std::size_t code_bytes = codes_.size() * sizeof(std::uint8_t);
    std::size_t pq_bytes = pq_.memory_usage_bytes();

    std::size_t list_bytes = 0;
    for (const auto& list : inverted_lists_) {
        list_bytes += list.size() * sizeof(std::size_t);
    }

    return centroid_bytes + code_bytes + pq_bytes + list_bytes;
}

void IVFPQIndex::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);

    if (!out) {
        throw std::runtime_error("Failed to open IVF-PQ index file for writing");
    }

    out.write(reinterpret_cast<const char*>(&nlist_), sizeof(nlist_));
    out.write(reinterpret_cast<const char*>(&nprobe_), sizeof(nprobe_));
    out.write(reinterpret_cast<const char*>(&kmeans_iterations_), sizeof(kmeans_iterations_));
    out.write(reinterpret_cast<const char*>(&pq_iterations_), sizeof(pq_iterations_));
    out.write(reinterpret_cast<const char*>(&num_vectors_), sizeof(num_vectors_));
    out.write(reinterpret_cast<const char*>(&dim_), sizeof(dim_));

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

    pq_.save_state(out);

    std::size_t codes_size = codes_.size();
    out.write(reinterpret_cast<const char*>(&codes_size), sizeof(codes_size));
    out.write(reinterpret_cast<const char*>(codes_.data()), codes_size * sizeof(std::uint8_t));
}

void IVFPQIndex::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);

    if (!in) {
        throw std::runtime_error("Failed to open IVF-PQ index file for reading");
    }

    in.read(reinterpret_cast<char*>(&nlist_), sizeof(nlist_));
    in.read(reinterpret_cast<char*>(&nprobe_), sizeof(nprobe_));
    in.read(reinterpret_cast<char*>(&kmeans_iterations_), sizeof(kmeans_iterations_));
    in.read(reinterpret_cast<char*>(&pq_iterations_), sizeof(pq_iterations_));
    in.read(reinterpret_cast<char*>(&num_vectors_), sizeof(num_vectors_));
    in.read(reinterpret_cast<char*>(&dim_), sizeof(dim_));

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

    pq_.load_state(in);

    std::size_t codes_size = 0;
    in.read(reinterpret_cast<char*>(&codes_size), sizeof(codes_size));
    codes_.resize(codes_size);
    in.read(reinterpret_cast<char*>(codes_.data()), codes_size * sizeof(std::uint8_t));
}

}