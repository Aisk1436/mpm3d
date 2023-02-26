#ifndef MPM3D_GUI_CUH
#define MPM3D_GUI_CUH

#include "mpm3d.cuh"
#include <GLFW/glfw3.h>

namespace gui {
void init();

void render(const mpm::Vector *);
}// namespace gui

#endif//MPM3D_GUI_CUH
