//
// Created by acacia on 10/25/21.
//

#include "gui.h"

namespace gui
{
    constexpr int window_x = 512;
    constexpr int window_y = 512;
    constexpr char window_name[] = "mpm";
    GLFWwindow* window;

    void error_callback(int error, const char* description)
    {
        fprintf(stderr, "Error %d: %s\n", error, description);
    }

    void draw_dot(GLdouble x, GLdouble y, GLdouble size)
    {
        constexpr auto n_side = 8;
        static GLdouble sin_tab[n_side];
        static GLdouble cos_tab[n_side];

        static auto init_flag = true;
        if (init_flag)
        {
            GLdouble pi = 3.1415926;
            for (auto i = 0; i < n_side; i++)
            {
                sin_tab[i] = sin(i * pi * 2 / n_side);
                cos_tab[i] = cos(i * pi * 2 / n_side);
            }
            init_flag = false;
            (void)init_flag;
        }

        glColor3d(0.616, 0.8, 0.878);
        auto radius_x = size / window_x / 2;
        auto radius_y = size / window_y / 2;
        glBegin(GL_POLYGON);
        for (auto i = 0; i < n_side; i++)
        {
            glVertex2d(x + radius_x * cos_tab[i],
                    y + radius_y * sin_tab[i]);
        }
        glEnd();
    }

    template<class T>
    std::enable_if_t<std::is_same_v<T, Eigen::Vector2f> ||
                     std::is_same_v<T, Eigen::Vector2d>, std::tuple<mpm::Real, mpm::Real>>
    transform(const T& a)
    {
        return { a[0], a[1] };
    }

    template<class T>
    std::enable_if_t<std::is_same_v<T, Eigen::Vector3f> ||
                     std::is_same_v<T, Eigen::Vector3d>, std::tuple<mpm::Real, mpm::Real>>
    transform(const T& a)
    {
        using mpm::Real;
        using mpm::Vector;
        Real phi = 0.49;
        Real theta = 0.56;
        Real x = a[0] - Real(0.5);
        Real y = a[1] - Real(0.5);
        Real z = a[2] - Real(0.5);
        Real c = std::cos(phi);
        Real s = std::sin(phi);
        Real C = std::cos(theta);
        Real S = std::sin(theta);
        x = x * c + z * s;
        z = z * c - x * s;
        Real u = x + Real(0.5);
        Real v = y * C + z * S + Real(0.5);
        return { u, v };
    }

    void init()
    {
        glfwSetErrorCallback(error_callback);
        if (!glfwInit())
        {
            std::cerr << "glfwInit failure\n";
            exit(1);
        }
        glfwWindowHint(GLFW_SAMPLES, 8);
        window = glfwCreateWindow(window_x, window_y, window_name, nullptr,
                nullptr);
        if (!window)
        {
            std::cerr << "glfwCreateWindow failure\n";
            glfwTerminate();
            exit(1);
        }
        glfwMakeContextCurrent(window);
        glfwSwapInterval(1);
        glClearColor(0.25, 0.25, 0.25, 1.0);
    }

    void render(const std::unique_ptr<mpm::Vector[]>& x)
    {
        glClear(GL_COLOR_BUFFER_BIT);
        double dot_size = 6.0;

        for (auto i = 0; i < mpm::n_particles; i++)
        {
            auto& p = x[i];
            auto[u, v] = transform(p);
            draw_dot(u * 2 - 1, v * 2 - 1, dot_size);
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
} // namespace gui