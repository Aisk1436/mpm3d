//
// Created by acacia on 10/23/21.
//

#ifndef MPM3D_MPM3D_CUH
#define MPM3D_MPM3D_CUH

#include <memory>
#include "Eigen/Dense"
#include "utils.h"

namespace mpm
{
    using Vector = Eigen::Vector2f;

    void init();

    void advance();

    std::unique_ptr<Vector[]> to_numpy();     // dummy
}



#endif //MPM3D_MPM3D_CUH
