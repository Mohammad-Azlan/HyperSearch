#include "ann/distance.hpp"

namespace ann {

float l2_distance_squared(const float* a, const float* b, std::size_t dim) {
    float sum = 0.0f;

    for (std::size_t i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }

    return sum;
}

}