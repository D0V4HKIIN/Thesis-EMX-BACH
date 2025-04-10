# EMX-BACH
EMX-BACH (Even More Extended Basic Accelerated C++ HOTPANTS) is an astronomical image subtraction software created by Jonas Bonnaudet, as part of a master's thesis conducted at MindRoad Öst AB. It is based on [X-BACH](https://github.com/MindRoadAB/Thesis-X-BACH) by Gustav Arneving and Hugo Wilhelmsson which itself is based on [BACH](https://github.com/MindRoadAB/Thesis-BACH), an earlier master's thesis by [Annie Wång](https://github.com/th3tard1sparadox) and [Victor Lells](https://github.com/vollells).

EMX-BACH is a parallelization of the popular image subtraction tool [HOTPANTS](https://github.com/acbecker/hotpants), rewritten in C++, and using OpenCL and OpenMP for task acceleration. It operates on two FITS images, one called science and one called template, generates a difference image by subtraction. The purpose of EMX-BACH was to explore the parallelization potential of non-trivial parallelizable tasks in the HOTPANTS algorithm and check how it would perform.

## Build
See [here](docs/Build.md).

## Usage
The usage of EMX-BACH is highlighted below:

```
BACH -t <template image name> -s <science image name>
```

X-BACH also supports some optional arguments. These arguments are presented below:

- `-o <convolved output name>`: name of the convolved output FITS image. Defaults to `conv.fits`.
- `-op <output path>`: name of the output folder, where the output images will be stored. Defaults to `out/`.
- `-ip <input path>`: name of the input folder, where the input images are located. Defaults to `res/`.
- `-v`: turns on verbose mode.
- `-vt`: prints execution time.
- `-p <int>`: integer to choose which platform to use. Platforms are listed when in verbose mode. Defaults to `0`.
- `-d <int>`: integer to choose which device to use. Devices are listed when in verbose mode. Defaults to `0`.
- `-sss <mp|cl>` : use openmp or opencl to compute Stamp and SubStamps. Defaults to `mp`.
- `--cpuPart <float>` : How much of the work is offloaded to the cpu during convolution. Defaults to `0.2`.
- `--accelerators <[int:int:float]>`: Takes a comma separated list of three values separated by colons that defines an opencl devices to be used to accelerate convolution. The first value defines the platform id, the second the device id and the last value defines how much work should be attributed to it. Defaults to an empty list.

For instance, if the input files are stored in `./res`, called `science.fits` and `template.fits`, and the output files would be written to `./out`, the following command would be used:

```
EMXBACH -t template.fits -s science.fits -ip "res/" -op "out/" --cpuPart 0.1 --accelerators 1:0:0.2,1:1:0.3
```

This would generate two files, `conv.fits` (convolved image) and `sub.fits` (subtracted image) in `./out` and split 10% of the work to the cpu, 20% to opencl device 1:0, 30% to opencl device 1:1 and 40% to the default gpu (0:0).

## Known Issues
- Input and output path arguments are glitchy. Always put '/' (or '\\') at the end of the path.
- Non-deterministic behaviour is observed between computers in some rare test cases.
