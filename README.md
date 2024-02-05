
Start original BACH README:

---

BACH was created as a part of a master thesis conducted at Linköping University during the spring of 2023 by [Annie Wång](https://github.com/th3tard1sparadox) and [Victor Lells](https://github.com/vollells). It is a PSF-alignment and image subtraction tool based on [HOTPANTS](https://github.com/acbecker/hotpants).

BACH is currently capable at aligning the PSFs of two FITS images, but has trouble with the subtraction. It cannot handle too large images, but in our general testing up to around 5K x 5K should work fine.

# Installation

Install all dependencies and then clone the repo by running the following command:

    git clone git@github.com:MindRoadAB/Exjobb-HOTPANTS.git

You can then use the program. See how to use BACH further below.

## Dependencies

BACH depends on the following packages:
- [CCfits](https://heasarc.gsfc.nasa.gov/fitsio/CCfits/)
- [C++ for OpenCL 2021](https://www.khronos.org/opencl/assets/CXX_for_OpenCL.html)

## Requirements

- CMake or Make depending on your OS
    - If you are using CMake, Visual Studio 17 2022 is also required
- C++20

# Usage

BACH will be compiled differently depending on what system you are on, follow the guide for your system below.

There will always be two output images, the convolved image and the subtracted image. The subtracted image will always be named `sub.fits`, while the convolved is `diff.fits` by default.

## Windows

We recommend that you use CMake if you run Windows. 

To compile BACH with CMake, make sure that `CMakeLists.txt` contain the correct paths to the libraries used. These can be found under `Include directories` and `Dependencies`.

When the paths are set correctly, open a terminal and navigate into the folder containing `CMakeLists.txt` and run the commands

    cmake -S . -B "build" -G "Visual Studio 17 2022" -A "x64"
    cmake --build .\build\

The exe file can be found in the folder `build/Debug` named `BACH.exe`. Move this file to the folder containing all `.cl` files. This can be done using the following command if you are standing in the `BACH` folder.

    mv .\build\Debug\BACH.exe .

Then run the file:

    .\BACH.exe -t [template image filename] -s [science image filename]

## Linux

If you are running Linux, we recommend that you use Make. Navigate to the folder containing `Makefile` and run

    make

Then execute the file by running

    ./BACH -t [template image filename] -s [science image filename]

## Command line options

These are the available command line options. The values within the parenthesis are the default values.

### Required

    -t [template image filename] // needs to be a FITS image
    -s [science image filename]  // needs to be a FITS image

### Optional

    -o [output image filename] // will output the convolved FITS image (diff.fits)
    -op [output path]          // (out/)
    -ip [input path]           // (res/)
    -v                         // turn on verbose mode
    -vt                        // turn on timers
