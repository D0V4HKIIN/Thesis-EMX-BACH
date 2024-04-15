#pragma once

#include <CL/opencl.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <utility>

#include "argsUtil.h"
#include "bach.h"
#include "datatypeUtil.h"

/* Utils */
void maskInput(const std::pair<cl_long, cl_long> &axis, const ClData& clData, const Arguments& args);
void sigmaClip(const cl::Buffer &data, int dataOffset, int dataCount, double *mean, double *stdDev, int maxIter, const ClData &clData, const Arguments& args);

void calcStats(const std::pair<cl_long, cl_long> &axis, const Arguments& args, const cl::Buffer& imgBuf, const ClStampsData& stampsData, const ClData& clData);

void ludcmp(const cl::Buffer &matrix, int matrixSize, int stampCount, const cl::Buffer &index, const cl::Buffer &vv, const ClData &clData);
void lubksb(const cl::Buffer &matrix, int matrixSize, int stampCount, const cl::Buffer &index, const cl::Buffer &result, const ClData &clData);
int ludcmp(std::vector<std::vector<double>>& matrix, const int matrixSize,
           std::vector<int>& index, double& rowInter, const Arguments& args);
void lubksb(std::vector<std::vector<double>>& matrix, const int matrixSize,
            const std::vector<int>& index, std::vector<double>& result);
double makeKernel(const cl::Buffer &kernel, const cl::Buffer &kernSolution, const std::pair<cl_long, cl_long> &imgSize, const int x, const int y, const Arguments& args, const ClData &clData);
double makeKernel(Kernel& kern, const std::pair<cl_long, cl_long> &imgSize, const int x,
                  const int y, const Arguments& args);

/* SSS */
void createStamps(std::vector<Stamp>& stamps, const int w, const int h, ClStampsData& stampsData, const ClData& clData, const Arguments& args);
cl_int findSStamps(const std::pair<cl_long, cl_long> &axis, const bool isTemplate, const Arguments& args, const cl::Buffer& imgBuf, const ClStampsData& stampsData, const ClData& clData);
void removeEmptyStamps(const Arguments& args, ClStampsData& stampsData, const ClData& clData);
void identifySStamps(const std::pair<cl_long, cl_long> &axis, const Arguments& args, ClData& clData);
void resetSStampSkipMask(const int w, const int h, const ClData& clData);
void readFinalStamps(std::vector<Stamp>& stamps, const ClStampsData& stampsData, const ClData& clData, const Arguments& args);

/* CMV */
void initFillStamps(std::vector<Stamp>& stamps, const std::pair<cl_long, cl_long> &axis, const cl::Buffer& tImgBuf, const cl::Buffer& sImgBuf,
               const Kernel& k, ClData& clData, ClStampsData& stampData, const Arguments& args);
void fillStamps(std::vector<Stamp>& stamps, const std::pair<cl_long, cl_long> &axis, const cl::Buffer& tImgBuf, const cl::Buffer& sImgBuf,
               int stampOffset, int stampCount, const Kernel& k, const ClData& clData, const ClStampsData& stampData, const Arguments& args);

/* CD && KSC */
double testFit(std::vector<Stamp>& stamps, const std::pair<cl_long, cl_long> &axis, const cl::Buffer &tImgBuf, const cl::Buffer &sImgBuf, ClData& clData, ClStampsData& stampData, const Arguments& args);
void createMatrix(const cl::Buffer &matrix, const cl::Buffer &weights, const ClData &clData, const ClStampsData &stampData, const std::pair<cl_long, cl_long>& imgSize, const Arguments& args);
std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
createMatrix(const std::vector<Stamp>& stamps, const std::pair<cl_long, cl_long>& imgSize, const Arguments& args);
void createScProd(const cl::Buffer &res, const cl::Buffer &weights, const cl::Buffer &img, const std::pair<cl_long, cl_long>& imgSize, const ClData &clData, const ClStampsData &stampData, const Arguments& args);
std::vector<double> createScProd(const std::vector<Stamp>& stamps, const Image& img,
                                 const std::vector<std::vector<double>>& weight, const Arguments& args);
void calcSigs(const cl::Buffer &tImgBuf, const cl::Buffer &sImgBuf, const std::pair<cl_long, cl_long> &axis,
              const cl::Buffer &model, const cl::Buffer &kernSol, const cl::Buffer &sigma,
              const ClStampsData &stampData, const ClData &clData, const Arguments& args);
void fitKernel(Kernel& k, std::vector<Stamp>& stamps, const Image &sImg, const cl::Buffer &tImgBuf, const cl::Buffer &sImgBuf,
               ClData &clData, const ClStampsData &stampData, const Arguments& args);
bool checkFitSolution(const Kernel& k, std::vector<Stamp>& stamps, const std::pair<cl_long, cl_long> &axis, const ClData &clData, const ClStampsData &stampData,
                      const cl::Buffer &tImgBuf, const cl::Buffer &sImgBuf, const cl::Buffer &kernSol, const Arguments& args);
void removeBadSubStamps(bool *check, const ClStampsData &stampData, std::vector<Stamp> &stamps, const std::vector<cl_uchar> &invalidatedSubStamps, const std::pair<cl_long, cl_long> &axis,
                        const cl::Buffer &sImgBuf, const cl::Buffer &tImgBuf, const Kernel &k, const ClData &clData, const Arguments &args);
