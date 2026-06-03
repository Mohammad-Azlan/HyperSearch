#pragma once

#include <cstddef>

namespace ann {

float l2_distance_squared(const float* a, const float* b, std::size_t dim);

float l2_distance_squared_avx2(const float* a, const float* b, std::size_t dim);

}