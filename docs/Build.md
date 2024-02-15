# Building
This document describes how to build the project from source.

First of all clone the repository. Make a copy of `CMakeListsUser.txt.example` and name it `CMakeListsUser.txt`. Change the paths in the copied file to point to folders where `CFITSIO`, `CCfits` and `OpenCL` are located. There are comments in the CMake file for guidance.

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
