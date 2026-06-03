#include "ann/distance.hpp"
#include <immintrin.h>

namespace ann {

    float l2_distance_squared(const float* a, const float* b, std::size_t dim) {
        float sum = 0.0f;

        for (std::size_t i = 0; i < dim; ++i) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }

        return sum;
    }

    float l2_distance_squared_avx2(
        const float* a,
        const float* b,
        std::size_t dim
    ) {
        __m256 sum = _mm256_setzero_ps();

        std::size_t i = 0;

        for (; i + 8 <= dim; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);

            __m256 diff = _mm256_sub_ps(va, vb);
            __m256 sq = _mm256_mul_ps(diff, diff);

            sum = _mm256_add_ps(sum, sq);
        }

        alignas(32) float temp[8];
        _mm256_store_ps(temp, sum);

        float result = temp[0] + temp[1] + temp[2] + temp[3]
                    + temp[4] + temp[5] + temp[6] + temp[7];

        for (; i < dim; ++i) {
            float diff = a[i] - b[i];
            result += diff * diff;
        }

        return result;
    }

}