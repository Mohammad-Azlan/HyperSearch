#pragma once

#include <chrono>

namespace ann {

class Timer {
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}

    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start_;

        return elapsed.count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
};

}