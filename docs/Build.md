# Building
This document describes how to build the project from source.

## Dependencies
- [CCfits](https://heasarc.gsfc.nasa.gov/fitsio/CCfits/)
    - [CFITSIO](https://heasarc.gsfc.nasa.gov/fitsio/)
        - [ZLib](https://www.zlib.net/)
- [OpenCL SDK](https://github.com/KhronosGroup/OpenCL-SDK)

On Windows, it is unfortunaly probably required to build all those from source, which may be a pain to do. In addition, if multiconfig builds are needed, all dependencies need to be compiled in both release and debug mode, and stored separately.

CCfits and OpenCL SDK are the only required dependencies. However, CCfits depends on more dependencies. CCfits also has to be built as a dynamic library, which at the time of writing requires it to be built as a static library first.
To do this, first compile CCfits as per their instruction, then update the dynamic library flag:

```
cmake -S <your source folder> -B <your build folder> -DBUILD_SHARED_LIBS=ON
```

and build again.

## Windows
MSVC is required to compile on Windows. First of all clone the repository. Make a copy of `CMakeListsUser.txt.example` and name it `CMakeListsUser.txt`. Change the paths in the copied file to point to folders where `CFITSIO`, `CCfits` and `OpenCL` are located. There are comments in the CMake file for guidance.

Create the build files with:

```
cmake -S . -B build
```

You may wish to append the flag `-G <generator>` and/or `-A <platform>`, where `<generator>` is the generator to use (for instance Visual Studio 16 2019) and `<platform>` is the targetted platform (for instance x64).

Then build the project with:

```
cmake --build build --config <config>
```

where `<config>` is `Debug` or `Release`. The executable will be available in `/build/Debug` or `/build/Release`, depending on the chosen config.

## Linux
idk what is going on above but this section is the only relevant part if you are on linux. Ignore cmake

Dependencies on Ubuntu with Nvidia graphics card:
```bash
apt install ocl-icd-opencl-dev # don't know
apt install libccfits-dev # fits images
apt install nvidia-opencl-dev # to use nvidia grahics card with opencl
apt install intel-opencl-icd # to use intel cpu with opencl
```

GCC is required to compile on Linux. First, clone the repository. Run `make`. Now, compilation should be done. If there are errors, check the Makefile and make sure you have the dependencies installed.
