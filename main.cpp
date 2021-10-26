#include <iostream>
#include <fstream>
#include <chrono>
#include <fmt/core.h>
#include "mpm3d.cuh"

#define USE_GUI // comment this line to disable gui
#ifdef USE_GUI

#include <memory>
#include <thread>
#include "gui.h"

constexpr int fps = 65;
constexpr auto frame_interval = std::chrono::nanoseconds(int(1e9)) / fps;

#endif

constexpr auto input_file_name = "../py/x_init";
constexpr auto output_file_name = "../py/x_cuda";

int main()
{
#ifdef USE_GUI
    gui::init();
#endif
    using namespace std::chrono_literals;
    auto now = std::chrono::high_resolution_clock::now;

    auto x_host = std::make_shared<mpm::Vector[]>(mpm::n_particles);
    auto ifs = std::ifstream(input_file_name);
    auto ofs = std::ofstream(output_file_name);

    for (auto i = 0; i < mpm::n_particles; i++)
    {
        for (auto j = 0; j < mpm::dim; j++)
        {
            ifs >> x_host[i][j];
        }
    }
    
    mpm::init(x_host);

    decltype(mpm::to_numpy()) x_cuda;
    auto start_time = now();
    for (auto runs = 0; runs < 2048; runs++)
    {
#ifdef USE_GUI
        auto frame_start = now();
#endif
        mpm::advance();
        x_cuda = mpm::to_numpy();
        (void)x_cuda;
#ifdef USE_GUI
        gui::render(x_cuda);
        // limit fps
        auto rest_time = frame_interval - (now() - frame_start);
        if (rest_time.count() > 0)
        {
            std::this_thread::sleep_for(rest_time); // not very precise
        }
#endif
    }
    auto used_time = (now() - start_time) / 1ns;
    std::cout << double(used_time) / 1e9 << "s\n";

    for (auto i = 0; i < mpm::n_particles; i++)
    {
        fmt::print(ofs, "{}: [{}]\n", i, x_cuda[i].transpose());
    }
    return 0;
}
