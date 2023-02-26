cmake_minimum_required(VERSION 3.24)
project(mpm3d CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

add_executable(mpm3d main.cu mpm3d.cu gui.cu)

set_target_properties(mpm3d PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES 60)

target_compile_options(mpm3d PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr>)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

add_subdirectory(deps/glfw)
target_link_libraries(mpm3d glfw)

find_package(OpenGL REQUIRED)
target_link_libraries(mpm3d OpenGL::GL)

add_subdirectory(deps/fmt)
target_link_libraries(mpm3d fmt::fmt)

target_include_directories(mpm3d PRIVATE deps/eigen)

if (CMAKE_BUILD_TYPE EQUAL "RELEASE")
    target_compile_definitions(mpm3d PRIVATE EIGEN_NO_DEBUG)
endif (CMAKE_BUILD_TYPE EQUAL "RELEASE")
