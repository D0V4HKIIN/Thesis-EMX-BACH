#pragma once

#include <CL/opencl.hpp>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

struct Arguments {
  std::string templateName;
  std::string scienceName;
  std::string outName = "diff.fits";

  std::string inputPath = "res/";
  std::string outPath = "out/";

  int stampsx = 10;
  int stampsy = 10;

  double threshLow = 0.0;
  double threshHigh = 25000.0;
  double threshKernFit = 20.0;
  double sigKernFit = 2.0;

  double sigClipAlpha = 3.0;
  double iqRange = 1.35;  // interquartile range

  cl_int maxKSStamps = 3;

  cl_int nPSF = 49;  // nPSF

  cl_int hSStampWidth = 15;  // half substamp width
  cl_int fSStampWidth = 31;  // full substamp width
  cl_int hKernelWidth = 10;  // half kernel width
  cl_int fKernelWidth = 21;  // full kernel width
  cl_int hStampWidth = 0;    // half stamp width
  cl_int fStampWidth = 0;    // half stamp width

  double inSpreadMaskFactor = 1.0;
  bool normalizeTemplate = true; // If false, normalize science

  cl_int backgroundOrder = 1;
  cl_int kernelOrder = 2;

  std::vector<cl_int> dg = {6, 4, 2};  // ngauss = length of dg
  std::vector<float> bg = {
      (1.0 / (2.0 * 0.7 * 0.7)),
      (1.0 / (2.0 * 1.5 * 1.5)),
      (1.0 / (2.0 * 3.0 * 3.0)),
  };

  bool verbose = false;
  bool verboseTime = false;
};

inline const char* getCmdOption(const char** begin, const char** end, const std::string& option) {
  const char** itr = std::find(begin, end, option);
  if(itr != end && ++itr != end) {
    return *itr;
  }
  return 0;
}

inline bool cmdOptionExists(const char** begin, const char** end,
                            const std::string& option) {
  return std::find(begin, end, option) != end;
}

inline Arguments args{};

inline void getArguments(const int argc, const char* argv[]) {
  if(cmdOptionExists(argv, argv + argc, "-o")) {
    args.outName = getCmdOption(argv, argv + argc, "-o");
  }

  if(cmdOptionExists(argv, argv + argc, "-op")) {
    args.outPath = getCmdOption(argv, argv + argc, "-op");
  }

  if(cmdOptionExists(argv, argv + argc, "-ip")) {
    args.inputPath = getCmdOption(argv, argv + argc, "-ip");
  }

  if(cmdOptionExists(argv, argv + argc, "-v")) {
    args.verbose = true;
  }

  if(cmdOptionExists(argv, argv + argc, "-vt")) {
    args.verboseTime = true;
  }

  if(cmdOptionExists(argv, argv + argc, "-t")) {
    args.templateName = getCmdOption(argv, argv + argc, "-t");
  } else {
    throw std::invalid_argument("Template file Input is required!");
    return;
  }

  if(cmdOptionExists(argv, argv + argc, "-s")) {
    args.scienceName = getCmdOption(argv, argv + argc, "-s");
  } else {
    throw std::invalid_argument("Science file input is required!");
    return;
  }
}
