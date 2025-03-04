#pragma once

#include <filesystem>

#include "datatypeUtil.h"

struct ClStampsData {
  cl::Buffer stampCoords;  // (x, y) coordinates // changed
  cl::Buffer stampSizes;   // changed
  struct {
    cl::Buffer skyEsts;  // changed
    cl::Buffer fwhms;    // changed
  } stats;

  cl::Buffer currentSubStamps;  // filled with 0 in currentSubStamps, then used
                                // in fillstamps (cmv)
  cl::Buffer subStampCoords;    // (x, y) coordinates //changed // needed by
                                // fillstamps (cmv)
  cl::Buffer subStampValues;    // changed // never referenced later?
  cl::Buffer subStampCounts;    // changed // needed by fillstamps (cmv)
  cl::Buffer w;
  cl::Buffer q;
  cl::Buffer b;
  unsigned int stampCount;  // changed (?)
};

struct ClData {
  cl::Device &device;
  cl::Context &context;
  cl::Program &program;
  cl::CommandQueue &queue;

  cl::Buffer tImgBuf;
  cl::Buffer sImgBuf;
  cl::Buffer maskBuf;  // changed
  cl::Buffer convImg;

  struct {
    cl::Buffer xy;
    cl::Buffer filterX;
    cl::Buffer filterY;
    cl::Buffer vec;
    cl::Buffer solution;
  } kernel;

  struct {
    cl::Buffer yConvTmp;
  } cmv;

  struct {
    cl::Buffer xy;
  } bg;

  struct {
    cl::Buffer kernelXy;
  } cd;

  int gaussCount;
  int qCount;
  int bCount;
  int wRows;
  int wColumns;

  ClStampsData tmpl;  // changed
  ClStampsData sci;   // changed
};

void init(Image &templateImg, Image &scienceImg, ClData &clData,
          const Arguments &args);
void sssCl(const std::pair<cl_int, cl_int> &axis,
           std::vector<Stamp> &templateStamps, std::vector<Stamp> &sciStamps,
           Arguments &args, ClData &clData);
void sssMp(std::vector<Stamp> &templateStamps, const Image &templateImg,
           std::vector<Stamp> &sciStamps, const Image &scienceImg,
           ImageMask &mask, Arguments &args);
void cmv(const std::pair<cl_int, cl_int> &axis,
         std::vector<Stamp> &templateStamps, std::vector<Stamp> &sciStamps,
         ClData &clData, const Arguments &args);
bool cd(Image &templateImg, Image &scienceImg,
        std::vector<Stamp> &templateStamps, std::vector<Stamp> &sciStamps,
        ClData &clData, const Arguments &args);
void ksc(std::vector<Stamp> &templateStamps, Kernel &convolutionKernel,
         const Image &sImg, const cl::Buffer &tImgBuf,
         const cl::Buffer &sImgBuf, ClData &clData,
         const ClStampsData &stampData, const Arguments &args);
double conv(const std::pair<cl_int, cl_int> &imgSize, Image &convImg,
            Kernel &convolutionKernel, bool convTemplate, ClData &clData,
            const Arguments &args);
void sub(const std::pair<cl_int, cl_int> &imgSize, Image &diffImg,
         bool convTemplate, double kernSum, const ClData &clData,
         const Arguments &args);
void fin(const Image &convImg, const Image &diffImg, const Arguments &args);
