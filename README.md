# mpm3d

Implementing mpm3d using CUDA C++. 

The [mpm3d](https://github.com/taichi-dev/taichi/blob/ae61418d66b79c4af2173acb00ba8a8b28b5dd54/python/taichi/examples/simulation/mpm3d.py) was originally written in [Taichi](https://github.com/taichi-dev/taichi).

You can change configures in mpm3d.cuh (e.g., change 'dim' from 3 to 2, or
change 'Real' from float to double). It should compile correctly.

## How to build

Please clone the repository recursively.

```
git clone --recursive https://github.com/jiajunhanh/mpm3d.git
```

### Windows

Open this project with either Visual Studio or CLion.

See [CUDA projects in CLion
](https://www.jetbrains.com/help/clion/cuda-projects.html).

Make sure you have the newest version of MSVC, or you may get errors when building Eigen.

### Linux

You can open this project with CLion (recommended).

See [CUDA projects in CLion
](https://www.jetbrains.com/help/clion/cuda-projects.html).

You can also build with Makefile.

```
make run
```

```
make run-release
```

```
make run-debug
```

Make sure you have installed [CMake](https://cmake.org/).
