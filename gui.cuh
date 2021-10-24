//
// Created by acacia on 10/25/21.
//

#ifndef MPM3D_GUI_CUH
#define MPM3D_GUI_CUH

#include <GLFW/glfw3.h>
#include "mpm3d.cuh"

namespace gui
{
    void init();

    void render(const std::unique_ptr<mpm::Vector[]>&);
}

#endif //MPM3D_GUI_CUH
