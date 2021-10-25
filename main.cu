#include <iostream>
#include <chrono>
#include "mpm3d.cuh"

#define USE_GUI // comment this line to disable gui
#ifdef USE_GUI

#include <thread>
#include "gui.cuh"

constexpr int fps = 65;
constexpr auto frame_interval = std::chrono::nanoseconds(int(1e9)) / fps;

#endif

int main()
{
#ifdef USE_GUI
    gui::init();
#endif
    using namespace std::chrono_literals;
    auto now = std::chrono::high_resolution_clock::now;
    mpm::init();

    auto start_time = now();

    for (auto runs = 0; runs < 2048; runs++)
    {
#ifdef USE_GUI
        auto frame_start = now();
#endif
        mpm::advance();
        auto x = mpm::to_numpy();
#ifdef USE_GUI
        gui::render(x);
        // limit fps
        auto rest_time = frame_interval - (now() - frame_start);
        if (rest_time.count() > 0)
        {
            std::this_thread::sleep_for(rest_time); // not very precise
        }
#endif
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto used_time = (end_time - start_time) / 1ns;

    std::cout << double(used_time) / 1e9 << "s\n";
    return 0;
}
