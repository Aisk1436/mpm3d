#ifndef MPM3D_GUI_CUH
#define MPM3D_GUI_CUH

#include <GLFW/glfw3.h>
#include "mpm3d.cuh"

namespace gui {
void init();

void render(const mpm::Vector *);
}

#endif //MPM3D_GUI_CUH
