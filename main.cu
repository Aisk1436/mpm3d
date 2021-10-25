#include <iostream>
#include <chrono>
#include "mpm3d.cuh"
#include "gui.cuh"

#define USE_GUI // comment this line to disable gui

int main()
{
#ifdef USE_GUI
    gui::init();
#endif
    mpm::init();

    auto start_time = std::chrono::high_resolution_clock::now();

    for (auto runs = 0; runs < 2048; runs++)
    {
        mpm::advance();
        auto x = mpm::to_numpy();
#ifdef USE_GUI
        gui::render(x);
#endif
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto used_time = (end_time - start_time) / std::chrono::nanoseconds(1);

    std::cout << double(used_time) / 1e9 << "s\n";
    return 0;
}
