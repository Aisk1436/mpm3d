//
// Created by acacia on 10/23/21.
//

#ifndef MPM3D_MPM3D_CUH
#define MPM3D_MPM3D_CUH

#include <memory>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include "Eigen/Dense"
#include "utils.h"

namespace mpm
{
    using Real = float;
//    using Real = double;

    constexpr int dim = 3, n_grid = 32, steps = 25;
    constexpr Real dt = 4e-4;

//    constexpr __device__ int dim_dev = 2, n_grid_dev = 128, steps_dev = 20;
//    constexpr __device__ Real dt_dev = 2e-4;

    using Vector = std::conditional_t<std::is_same_v<Real, float>, std::conditional_t<
            dim == 2, Eigen::Vector2f, Eigen::Vector3f>, std::conditional_t<
            dim == 2, Eigen::Vector2d, Eigen::Vector3d>>;
    using Matrix = std::conditional_t<std::is_same_v<Real, float>, std::conditional_t<
            dim == 2, Eigen::Matrix2f, Eigen::Matrix3f>, std::conditional_t<
            dim == 2, Eigen::Matrix2d, Eigen::Matrix3d>>;
    using Vectori = std::conditional_t<
            dim == 2, Eigen::Vector2i, Eigen::Vector3i>;

    constexpr int n_particles =
            utils::power(n_grid, dim) / utils::power(2, dim - 1);

    void init(std::shared_ptr<mpm::Vector[]> = nullptr);

    void advance();

    std::unique_ptr<Vector[]> to_numpy();     // dummy
}

#endif //MPM3D_MPM3D_CUH
