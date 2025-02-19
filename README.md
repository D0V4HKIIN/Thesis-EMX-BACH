# X-BACH
X-BACH (Extended Basic Accelerated C++ HOTPANTS) is an astronomical image subtraction software created by Gustav Arneving and Hugo Wilhelmsson, as part of a master's thesis conducted at MindRoad Öst AB. It is based on [BACH](https://github.com/MindRoadAB/Thesis-BACH), an earlier master's thesis by [Annie Wång](https://github.com/th3tard1sparadox) and [Victor Lells](https://github.com/vollells).

X-BACH is a parallelization of the popular image subtraction tool [HOTPANTS](https://github.com/acbecker/hotpants), rewritten in C++, and using OpenCL for task acceleration. It operates on two FITS images, one called science and one called template, generates a difference image by subtraction. The purpose of X-BACH was to explore the parallelization potential of non-trivial parallelizable tasks in the HOTPANTS algorithm and check how it would perform.

## Build
See [here](docs/Build.md).

## Usage
The usage of X-BACH is highlighted below:

```
BACH -t <template image name> -s <science image name>
```

X-BACH also supports some optional arguments. These arguments are presented below:

- `-o <convolved output name>`: name of the convolved output FITS image. Defaults to `diff.fits`.
- `-op <output path>`: name of the output folder, where the output images will be stored. Defaults to `out/`.
- `-ip <input path>`: name of the input folder, where the input images are located. Defaults to `res/`.
- `-v`: turns on verbose mode.
- `-vt`: prints execution time.
- `-p`: integer to choose which platform to use. Platforms are listed when in verbose mode. Defaults to 0.
- `-d`: integer to choose which device to use. Devices are listed when in verbose mode. Defaults to 0.

For instance, if the input files are stored in `C:\in`, called `science.fits` and `template.fits`, and the output files would be written to `C:\out`, the following command would be used:

```
BACH -t template.fits -s science.fits -ip "C:\in\" -op "C:\out\"
```

This would generate two files, `diff.fits` (convolved image) and `sub.fits` (subtracted image) in `C:\out`.

## Known Issues
- Input and output path arguments are glitchy. Always put '/' (or '\\') at the end of the path.
- Non-deterministic behaviour is observed between computers in some rare test cases.
