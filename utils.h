//
// Created by acacia on 10/23/21.
//

#ifndef MPM3D_UTILS_H
#define MPM3D_UTILS_H

#include <iostream>
#include <random>

namespace utils
{
    inline void cuda_check_error()
    {
        cudaDeviceSynchronize();
        auto status = cudaGetLastError();
        if (status != cudaSuccess)
        {
            std::cerr << cudaGetErrorName(status) << ": "
                      << cudaGetErrorString(status);
            exit(1);
        }
    }

    inline double rand_real(double min, double max)
    {
        static std::random_device rd;
        static std::default_random_engine gen(rd());

        return std::uniform_real_distribution<>{ min, max }(gen);
    }

    inline double rand_real()
    {
        return rand_real(0.0, 1.0);
    }

    template<typename T>
    constexpr T _sqr(T a)
    {
        return a * a;
    }

    template<typename T>
    constexpr T power(T a, std::size_t n)
    {
        return n == 0 ? 1 : _sqr(power(a, n / 2)) * (n % 2 == 0 ? 1 : a);
    }

    inline int get_block_num(int thread_num, int max_threads_per_block)
    {
        return (thread_num + max_threads_per_block - 1) / max_threads_per_block;
    }

//    template<class T, int Dim>
//    class Matrix    // square matrix
//    {
//        std::array<T, Dim * Dim> mat;
//
//    public:
//
//        T& operator()(size_t x_dev, size_t y)
//        {
//            return mat[x_dev * Dim + y];
//        }
//
//        const T& operator()(size_t x_dev, size_t y) const
//        {
//            return mat[x_dev * Dim + y];
//        }
//
//        Matrix operator+=(Matrix&& rhs)
//        {
//            for (auto x_dev = 0; x_dev < Dim; x_dev++)
//                for (auto y = 0; y < Dim; y++)
//                {
//                    mat[x_dev * Dim + y] += rhs(x_dev, y);
//                }
//        }
//
//        Matrix operator+=(T rhs)
//        {
//            for (auto x_dev = 0; x_dev < Dim; x_dev++)
//                for (auto y = 0; y < Dim; y++)
//                {
//                    mat[x_dev * Dim + y] += rhs;
//                }
//        }
//
//
//        Matrix operator*=(Matrix&& rhs)
//        {
//            for (auto x_dev = 0; x_dev < Dim; x_dev++)
//                for (auto y = 0; y < Dim; y++)
//                {
//                    mat[x_dev * Dim + y] += rhs(x_dev, y);
//                }
//        }
//
//        Matrix operator*=(T rhs)
//        {
//            for (auto x_dev = 0; x_dev < Dim; x_dev++)
//                for (auto y = 0; y < Dim; y++)
//                {
//                    mat[x_dev * Dim + y] *= rhs;
//                }
//        }    };
} // namespace utils

#endif //MPM3D_UTILS_H
