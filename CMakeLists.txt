cmake_minimum_required(VERSION 3.20)
project(MPM3D CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

add_subdirectory(fmt)

add_executable(MPM3D main.cu mpm3d.cu gui.cu)

set_target_properties(MPM3D PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES 60)

target_compile_options(MPM3D PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr --diag-suppress 20236,20012>)

find_package(glfw3 3.3 REQUIRED)
target_link_libraries(MPM3D PRIVATE glfw)

find_package(OpenGL REQUIRED)
target_link_libraries(MPM3D PRIVATE OpenGL::GL)

target_link_libraries(MPM3D PRIVATE fmt::fmt)

if (CMAKE_BUILD_TYPE EQUAL "RELEASE")
    target_compile_definitions(MPM3D PUBLIC EIGEN_NO_DEBUG)
endif (CMAKE_BUILD_TYPE EQUAL "RELEASE")
