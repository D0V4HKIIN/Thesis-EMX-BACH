# Building
This document describes how to build the project from source.

## Dependencies
- [CCfits](https://heasarc.gsfc.nasa.gov/fitsio/CCfits/)
    - [CFITSIO](https://heasarc.gsfc.nasa.gov/fitsio/)
        - [ZLib](https://www.zlib.net/)
- [OpenCL SDK](https://github.com/KhronosGroup/OpenCL-SDK)

CCfits and OpenCL SDK are the only required dependencies. However, CCfits depends on more dependencies. CCfits also has to be built as a dynamic library, which at the time of writing requires it to be built as a static library first.
To do this, first compile CCfits as per their instruction, then run `cmake -B <your build folder> -DBUILD_SHARED_LIBS=ON` and build again.

On Windows, it is unfortunaly probably required to build all those from source, which may be a pain to do.

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
GCC is required to compile on Linux. First, clone the repository. Run `make`. Now, compilation should be done. If there are errors, check the Makefile and make sure you have the dependencies installed.
