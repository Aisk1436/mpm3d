//
// Created by acacia on 10/23/21.
//

#include "mpm3d.cuh"

#define UM_FLAG

using namespace utils;

namespace mpm
{
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

    int warp_size;

    inline void cuda_check_error()
    {
        cudaDeviceSynchronize();
        auto status = cudaGetLastError();
        if (status != cudaSuccess)
        {
            fmt::print(std::cerr, "{}: {}\n", cudaGetErrorName(status),
                    cudaGetErrorString(status));
            exit(1);
        }
    }

    template<class T, class ...A>
    void
    cuda_launch_kernel(T func, int n_thread, A& ...args)
    {
        int grid_size;
        int block_size;
        int min_grid_size;
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, func, 0,
                n_thread);
        // reduce the block size to increase the performance
        block_size =
                (block_size + warp_size * 2 - 1) / (warp_size * 2) * warp_size;
        grid_size = (n_thread + block_size - 1) / block_size;
        func<<<grid_size, block_size>>>(args...);
    }

    template<class R, class A>
    __device__ R narrow_cast(const A& a)
    {
        R r = R(a);
        if (A(r) != a) printf("warning: info loss in narrow_cast\n");
        return r;
    }

    __device__ Vectori get_offset(size_t idx)
    {
        Vectori offset;
        for (auto i = dim - 1; i >= 0; i--)
        {
            offset[i] = narrow_cast<int, size_t>(idx % 3);
            idx /= 3;
        }
        return offset;
    }

    __device__ Vectori get_indices(size_t idx)
    {
        Vectori indices;
        for (auto i = dim - 1; i >= 0; i--)
        {
            indices[i] = narrow_cast<int, size_t>(idx % n_grid);
            idx /= n_grid;
        }
        return indices;
    }

    __global__ void init_kernel(Real* J)
    {
        auto idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_particles) return;
        J[idx] = 1;
    }

    __global__ void reset_kernel(Vector* grid_v, Real* grid_m)
    {
        auto idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= power(n_grid, dim)) return;
        grid_v[idx].setZero();
        grid_m[idx] = 0;
    }

    __global__ void
    particle_to_grid_kernel(Vector* x, Vector* v, Matrix* C, const Real* J,
            Vector* grid_v, Real* grid_m)
    {
        auto idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_particles) return;
        // do not use the auto keyword with Eigen's expressions
        Vector Xp = x[idx] / dx;
        Vectori base = (Xp.array() - 0.5).cast<int>();
        Vector fx = Xp - base.cast<Real>();
        std::array<Vector, 3> w{ 0.5 * (1.5 - fx.array()).pow(2),
                                 0.75 - (fx.array() - 1.0).pow(2),
                                 0.5 * (fx.array() - 0.5).pow(2) };
        auto stress = -dt * 4 * E * p_vol * (J[idx] - 1) / std::pow(dx, 2);
        Matrix affine = Matrix::Identity() * stress + p_mass * C[idx];
        for (auto offset_idx = 0; offset_idx < neighbour; offset_idx++)
        {
            Vectori offset = get_offset(offset_idx);
            Vector dpos = (offset.cast<Real>() - fx) * dx;
            Real weight = 1.0;
            for (auto i = 0; i < dim; i++)
            {
                weight *= w[offset[i]][i];
            }

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
//                grid_v[grid_idx][i] += grid_v_add[i];
            }
            atomicAdd(&(grid_m[grid_idx]), grid_m_add);
//            grid_m[grid_idx] += grid_m_add;
        }
    }

    __global__ void grid_update_kernel(Vector* grid_v, Real* grid_m)
    {
        auto idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= power(n_grid, dim)) return;
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
        if (idx >= n_particles) return;
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
            Vectori offset = get_offset(offset_idx);
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

    void init(std::shared_ptr<mpm::Vector[]> x_init)
    {
        cudaFree(x_dev);
        cudaFree(v_dev);
        cudaFree(C_dev);
        cudaFree(J_dev);
        cudaFree(grid_v_dev);
        cudaFree(grid_m_dev);

#ifdef UM_FLAG
        cudaMallocManaged(&x_dev, n_particles * sizeof(Vector));
        cudaMallocManaged(&v_dev, n_particles * sizeof(Vector));
        cudaMallocManaged(&C_dev, n_particles * sizeof(Matrix));
        cudaMallocManaged(&J_dev, n_particles * sizeof(Real));
        cudaMallocManaged(&grid_v_dev, power(n_grid, dim) * sizeof(Vector));
        cudaMallocManaged(&grid_m_dev, power(n_grid, dim) * sizeof(Real));
        cuda_check_error();
#else
        cudaMalloc(&x_dev, n_particles * sizeof(Vector));
        cudaMalloc(&v_dev, n_particles * sizeof(Vector));
        cudaMalloc(&C_dev, n_particles * sizeof(Matrix));
        cudaMalloc(&J_dev, n_particles * sizeof(Real));
        cudaMalloc(&grid_v_dev, power(n_grid, dim) * sizeof(Vector));
        cudaMalloc(&grid_m_dev, power(n_grid, dim) * sizeof(Real));
        cuda_check_error();
#endif

        if (!x_init)
        {
            x_init = std::make_unique<Vector[]>(n_particles);
            // initialize x on the host and copy to the device
            for (auto i = 0; i < n_particles; i++)
            {
                for (auto j = 0; j < dim; j++)
                {
                    x_init[i][j] = Real(rand_real());
                }
                x_init[i] = (x_init[i] * 0.4).array() + 0.15;
            }
        }
#ifdef UM_FLAG
        for (auto i = 0; i < n_particles; i++)
        {
            x_dev[i] = x_init[i];
        }
#else
        cudaMemcpy(x_dev, x_init.get(), n_particles * sizeof(Vector),
                cudaMemcpyHostToDevice);
#endif

        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, 0);
        warp_size = prop.warpSize;
        cuda_launch_kernel(init_kernel, n_particles, J_dev);
        cuda_check_error();
    }

    void advance()
    {
        auto T = steps;
        auto n_grid_thread = power(n_grid, dim);
        while (T--)
        {
            cuda_launch_kernel(reset_kernel, n_grid_thread, grid_v_dev,
                    grid_m_dev);

            cuda_launch_kernel(particle_to_grid_kernel, n_particles, x_dev,
                    v_dev, C_dev, J_dev, grid_v_dev, grid_m_dev);

            cuda_launch_kernel(grid_update_kernel, n_grid_thread, grid_v_dev,
                    grid_m_dev);

            cuda_launch_kernel(grid_to_particle_kernel, n_particles, x_dev,
                    v_dev, C_dev, J_dev, grid_v_dev);

            cuda_check_error();
        }
    }

    Vector* to_numpy()
    {
#ifdef UM_FLAG
        auto& x_host = x_dev;
#else
        auto x_host = new Vector[n_particles];
        cudaMemcpy(x_host, x_dev, n_particles * sizeof(Vector),
                cudaMemcpyDeviceToHost);
#endif
        return x_host;
    }

} // namespace mpm

