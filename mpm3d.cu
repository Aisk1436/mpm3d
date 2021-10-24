//
// Created by acacia on 10/23/21.
//

#include "mpm3d.cuh"

using namespace utils;

namespace mpm
{
    using Real = float;
    using Vector = Eigen::Vector2f;
    using Matrix = Eigen::Matrix2f;
    using Vectori = Eigen::Vector2i;

    constexpr __device__ int dim = 2, n_grid = 128, steps = 20;
    constexpr __device__ Real dt = 2e-4;
    constexpr __device__ int n_particles =
            power(n_grid, dim) / power(2, dim - 1);
    constexpr __device__ int neighbour = power(3, dim);

    constexpr __device__ Real dx = 1.0 / n_grid;
    constexpr Real p_rho = 1.0;
    constexpr Real p_vol = power(dx * 0.5, 2);
    constexpr __device__ Real p_mass = p_vol * p_rho;
    constexpr __device__ Real gravity = 9.8;
    constexpr __device__ int bound = 3;
    constexpr __device__ Real E = 400;

    Vector* x_dev;
    Vector* v_dev;
    Matrix* C_dev;
    Real* J_dev;
    Vector* grid_v_dev;
    Real* grid_m_dev;

    int max_threads_per_block;

    template<class R, class A>
    __device__ R narrow_cast(const A& a)
    {
        R r = R(a);
        if (A(r) != a) printf("warning: info loss in narrow_cast");
        return r;
    }

    __device__ Vectori get_indices(size_t idx)
    {
        Vectori indices;
        for (auto i = dim - 1; i >= 0; i--)
        {
            indices[i] = narrow_cast<int, size_t>(idx % 3 - 1);
            idx /= 3;
        }
        return indices;
    }

    __global__ void init_kernel(Real* J)
    {
        auto idx = blockIdx.x * blockDim.x + threadIdx.x;
        J[idx] = 1;
    }

    __global__ void reset_kernel(Vector* grid_v, Real* grid_m)
    {
        auto idx = blockIdx.x * blockDim.x + threadIdx.x;
        grid_v[idx].setZero();
        grid_m[idx] = 0;
    }

    __global__ void
    particle_to_grid_kernel(Vector* x, Vector* v, Matrix* C, const Real* J,
            Vector* grid_v, Real* grid_m)
    {
        auto idx = blockIdx.x * blockDim.x + threadIdx.x;
        // do not use the auto keyword with Eigen's expressions
        Vector Xp = x[idx] / dx;
        Vectori base = (Xp.array() - 0.5).cast<int>();
        Vector fx = Xp - base.cast<Real>();
        std::array<Vector, 3> w{ 0.5 * (1.5 - fx.array()).pow(2),
                                 0.75 - (fx.array() - 1.0).pow(2),
                                 0.5 * (fx.array() - 0.5).pow(2) };
        auto stress = -dt * 4 * E * p_vol * (J[idx] - 1) / pow(dx, 2);
        Matrix affine = Matrix::Identity() * stress + p_mass * C[idx];
        for (auto offset_idx = 0; offset_idx < neighbour; offset_idx++)
        {
            Vectori offset = get_indices(offset_idx).array() - 1;
            Vector dpos = (offset.cast<Real>() - fx) * dx;
            Real weight = 1.0;
            for (auto i = 0; i < dim; i++)
            {
                weight *= w[offset[i]][i];
            }

            // TODO: evaluate performance of atomic operations
            Vector grid_v_add = weight * (p_mass * v[idx] + affine * dpos);
            auto grid_m_add = weight * p_mass;
            Vectori grid_idx_vector = base + offset;
            auto grid_idx = 0;
            for (auto i = 0; i < dim; i++)
            {
                grid_idx = grid_idx * n_grid + grid_idx_vector[i];
            }
            for (auto i = 0; i < dim; i++)
            {
                atomicAdd(&(grid_v[grid_idx][i]), grid_v_add[i]);
            }
            atomicAdd(&(grid_m[grid_idx]), grid_m_add);
        }
    }

    __global__ void grid_update_kernel(Vector* grid_v, Real* grid_m)
    {
        auto idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (grid_m[idx] > 0)
        {
            grid_v[idx] /= grid_m[idx];
        }
        grid_v[idx][1] -= dt * gravity;
        Vectori indices = get_indices(idx);
        for (auto i = 0; i < dim; i++)
        {
            if ((indices[i] < bound && grid_v[idx][i] < 0) ||
                (indices[i] > n_grid - bound && grid_v[idx][i] > 0))
            {
                grid_v[idx][i] = 0;
            }
        }
    }

    __global__ void
    grid_to_particle_kernel(Vector* x, Vector* v, Matrix* C, Real* J,
            Vector* grid_v)
    {
        auto idx = blockIdx.x * blockDim.x + threadIdx.x;
        Vector Xp = x[idx] / dx;
        Vectori base = (Xp.array() - 0.5).cast<int>();
        Vector fx = Xp - base.cast<Real>();
        std::array<Vector, 3> w{ 0.5 * (1.5 - fx.array()).pow(2),
                                 0.75 - (fx.array() - 1.0).pow(2),
                                 0.5 * (fx.array() - 0.5).pow(2) };

        Vector new_v = Vector::Zero();
        Matrix new_C = Matrix::Zero();
        for (auto offset_idx = 0; offset_idx < neighbour; offset_idx++)
        {
            Vectori offset = get_indices(offset_idx).array() - 1;
            Vector dpos = (offset.cast<Real>() - fx) * dx;
            Real weight = 1.0;
            for (auto i = 0; i < dim; i++)
            {
                weight *= w[offset[i]][i];
            }
            Vectori grid_idx_vector = base + offset;
            auto grid_idx = 0;
            for (auto i = 0; i < dim; i++)
            {
                grid_idx = grid_idx * n_grid + grid_idx_vector[i];
            }
            new_v += weight * grid_v[grid_idx];
            new_C += 4.0 * weight * grid_v[grid_idx] * dpos.transpose() /
                     pow(dx, 2);
        }
        v[idx] = new_v;
        x[idx] += dt * v[idx];
        J[idx] *= Real(1.0) + dt * new_C.trace();
        C[idx] = new_C;
    }

    __host__ void init()
    {
        cudaFree(x_dev);
        cudaFree(v_dev);
        cudaFree(C_dev);
        cudaFree(J_dev);
        cudaFree(grid_v_dev);
        cudaFree(grid_m_dev);

        cudaMalloc(&x_dev, n_particles * sizeof(Vector));
        cudaMalloc(&v_dev, n_particles * sizeof(Vector));
        cudaMalloc(&C_dev, n_particles * sizeof(Matrix));
        cudaMalloc(&J_dev, n_particles * sizeof(Real));
        cudaMalloc(&grid_v_dev, power(n_grid, dim) * sizeof(Vector));
        cudaMalloc(&grid_m_dev, power(n_grid, dim) * sizeof(Real));
        cuda_check_error();

        // initialize x on host and copy to device
        auto x_host = std::make_unique<Vector[]>(n_particles);
        for (auto i = 0; i < n_particles; i++)
        {
            x_host[i].setRandom();
            x_host[i] = (x_host[i] * 0.5).array() + 1.0;
            x_host[i] = (x_host[i] * 0.4).array() + 0.15;
        }
        cudaMemcpy(x_dev, x_host.get(), n_particles * sizeof(Vector),
                cudaMemcpyHostToDevice);

        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, 0);
        max_threads_per_block = prop.maxThreadsPerBlock;
        int block_num = get_block_num(n_particles,
                max_threads_per_block);
        init_kernel<<<block_num, max_threads_per_block>>>(J_dev);
        cuda_check_error();
    }

    void advance()
    {
        auto T = steps;
        int particle_block_num = get_block_num(n_particles,
                max_threads_per_block);
        int grid_block_num = get_block_num(power(n_grid, dim),
                max_threads_per_block);
        while (T--)
        {
            reset_kernel<<<grid_block_num, max_threads_per_block>>>(grid_v_dev,
                    grid_m_dev);

            particle_to_grid_kernel<<<particle_block_num, max_threads_per_block>>>(
                    x_dev, v_dev, C_dev, J_dev, grid_v_dev, grid_m_dev);

            grid_update_kernel<<<grid_block_num, max_threads_per_block>>>(
                    grid_v_dev, grid_m_dev);

            grid_to_particle_kernel<<<particle_block_num, max_threads_per_block>>>(
                    x_dev, v_dev, C_dev, J_dev, grid_v_dev);

            cuda_check_error();
        }
    }

    std::unique_ptr<Vector[]> to_numpy()
    {
        auto x_host = std::make_unique<Vector[]>(n_particles);
        cudaMemcpy(x_host.get(), x_dev, n_particles * sizeof(Vector),
                cudaMemcpyDeviceToHost);
        return x_host;
    }
}