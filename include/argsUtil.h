#pragma once

#include <array>
#include <CL/opencl.hpp>
#include <string>

struct Arguments {
  std::string templateName;
  std::string scienceName;
  std::string outName = "diff.fits";

  std::string inputPath = "res/";
  std::string outPath = "out/";

  int platform = 0; // default platform and devices
  int device = 0;

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
  cl_int fStampWidth = 0;    // full stamp width

  double inSpreadMaskFactor = 1.0;
  bool normalizeTemplate = true; // If false, normalize science

  cl_int backgroundOrder = 1;
  cl_int kernelOrder = 2;

  std::array<cl_int, 3> dg = {6, 4, 2};  // ngauss = length of dg
  std::array<cl_float, 3> bg = {
      (1.0 / (2.0 * 0.7 * 0.7)),
      (1.0 / (2.0 * 1.5 * 1.5)),
      (1.0 / (2.0 * 3.0 * 3.0)),
  };

  bool verbose = false;
  bool verboseTime = false;
};

const char* getCmdOption(const char** begin, const char** end, const std::string& option);

bool cmdOptionExists(const char** begin, const char** end, const std::string& option);

void getArguments(const int argc, const char* argv[], Arguments& args);
