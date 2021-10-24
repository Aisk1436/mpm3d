#include <iostream>
#include <chrono>
#include "Eigen/Dense"
#include "mpm3d.cuh"

int main()
{
    auto start_time = std::chrono::high_resolution_clock::now();

    mpm::init();
    for (auto runs = 0; runs < 2048; runs++)
    {
        mpm::advance();
        auto x = mpm::to_numpy();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto used_time = (end_time - start_time) / std::chrono::nanoseconds(1);

    std::cout << double(used_time) / 1e9 << "s\n";
    return 0;
}
