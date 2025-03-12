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
void maskInput(const std::pair<cl_int, cl_int>& axis, const ClData& clData,
               const Arguments& args);
void sigmaClip(const cl::Buffer& data, int dataOffset, int dataCount,
               double* mean, double* stdDev, int maxIter, const ClData& clData,
               const Arguments& args);

void calcStats(const std::pair<cl_int, cl_int>& axis, const Arguments& args,
               const cl::Buffer& imgBuf, const ClStampsData& stampsData,
               const ClData& clData);
int timeDiff(std::chrono::time_point<std::chrono::steady_clock> end,
             std::chrono::time_point<std::chrono::steady_clock> start);

void ludcmp(const cl::Buffer& matrix, int matrixSize, int stampCount,
            const cl::Buffer& index, const cl::Buffer& vv,
            const ClData& clData);
void lubksb(const cl::Buffer& matrix, int matrixSize, int stampCount,
            const cl::Buffer& index, const cl::Buffer& result,
            const ClData& clData);
int ludcmp(std::vector<std::vector<double>>& matrix, const int matrixSize,
           std::vector<int>& index, double& rowInter, const Arguments& args);
void lubksb(std::vector<std::vector<double>>& matrix, const int matrixSize,
            const std::vector<int>& index, std::vector<double>& result);
double makeKernel(const cl::Buffer& kernel, const cl::Buffer& kernSolution,
                  const std::pair<cl_int, cl_int>& imgSize, const int x,
                  const int y, const Arguments& args, const ClData& clData);
double makeKernel(const Kernel& kern, std::vector<double>& currKernel,
                  const std::pair<cl_int, cl_int>& imgSize, const int x,
                  const int y, const Arguments& args);
void convCl(const int w, const int h, const std::vector<cl_double> convKernels,
            const int xSteps, const bool scaleConv, const double invKernSum,
            Image& convImg, const Arguments& args, ClData& clData);
void convMp(const int w, const int h, const std::vector<cl_double> convKernels,
            const int xSteps, const bool scaleConv, const double invKernSum,
            Image& convImg, const Image& templateImage,
            const std::vector<double> kernSolution, ImageMask& mask,
            const Arguments& args);

/* SSS for OpenCl*/
void createStamps(const int w, const int h, ClStampsData& stampsData,
                  const ClData& clData, const Arguments& args);
cl_int findSStamps(const std::pair<cl_int, cl_int>& axis, const bool isTemplate,
                   const Arguments& args, const cl::Buffer& imgBuf,
                   const ClStampsData& stampsData, const ClData& clData);
void removeEmptyStamps(const Arguments& args, ClStampsData& stampsData,
                       const ClData& clData);
void identifySStamps(const std::pair<cl_int, cl_int>& axis,
                     const Arguments& args, const ClData& clData);
void resetSStampSkipMask(const int w, const int h, const ClData& clData);
void readFinalStamps(std::vector<Stamp>& stamps, const ClStampsData& stampsData,
                     const ClData& clData, const Arguments& args);

/* SSS for OpenMP*/
void createStampsMp(const int stampX, const int stampY, const int w,
                    const int h, Stamp& stamp, const Arguments& args);
void identifySStampsMp(Stamp& templStamp, const Image& templImage,
                       Stamp& scienceStamp, const Image& scienceImage,
                       ImageMask& mask, const Arguments& args);
void calcStatsMp(Stamp& stamp, const Image& image, ImageMask& mask,
                 const Arguments& args);
int findSStampsMp(Stamp& stamp, const Image& image, ImageMask& mask,
                  const bool isTemplate, const Arguments& args);
double checkSStampMp(const SubStamp& sstamp, const Image& image,
                     ImageMask& mask, const Stamp& stamp,
                     const ImageMasks badMask, const ImageMasks skipMask,
                     const Arguments& args);
void computeStamps(const int w, const int h, const Image& templateImg,
                   std::vector<Stamp>& templateStamps, const Image& scienceImg,
                   std::vector<Stamp>& scienceStamps, ImageMask& mask,
                   const Arguments& args);
void moveSssToGpu(const std::vector<Stamp>& templateStamps,
                  const std::vector<Stamp>& scienceStamps,
                  const ImageMask& mask, ClData& clData, const Arguments& args);
void moveStamps(const std::vector<Stamp>& stamps, ClStampsData& stampsData,
                ClData& clData, const Arguments& args);

template <typename T>
void uploadBuffer(const std::vector<T>& v, cl::Buffer& buffer,
                  cl::CommandQueue& queue) {
  queue.enqueueWriteBuffer(buffer, CL_TRUE, 0, sizeof(T) * v.size(), v.data());
}
template <typename T>
void printBuffer(const cl::Buffer& buffer, size_t size, cl::CommandQueue& queue,
                 const std::string& separator = ',') {
  std::vector<T> v(size);
  queue.enqueueReadBuffer(buffer, CL_TRUE, 0, sizeof(T) * size, v.data());

  std::copy(v.begin(), v.end(),
            std::ostream_iterator<T>(std::cout, separator.c_str()));
  std::cout << std::endl;
}

/* CMV */
void initFillStamps(std::vector<Stamp>& stamps,
                    const std::pair<cl_int, cl_int>& axis,
                    const cl::Buffer& tImgBuf, const cl::Buffer& sImgBuf,
                    ClData& clData, ClStampsData& stampData,
                    const Arguments& args);
void fillStamps(std::vector<Stamp>& stamps,
                const std::pair<cl_int, cl_int>& axis,
                const cl::Buffer& tImgBuf, const cl::Buffer& sImgBuf,
                int stampOffset, int stampCount, const ClData& clData,
                const ClStampsData& stampData, const Arguments& args);

/* CD && KSC */
double testFit(std::vector<Stamp>& stamps,
               const std::pair<cl_int, cl_int>& axis, const cl::Buffer& tImgBuf,
               const cl::Buffer& sImgBuf, ClData& clData,
               ClStampsData& stampData, const Arguments& args);
void createMatrix(const cl::Buffer& matrix, const cl::Buffer& weights,
                  const ClData& clData, const ClStampsData& stampData,
                  const std::pair<cl_int, cl_int>& imgSize,
                  const Arguments& args);
std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
createMatrix(const std::vector<Stamp>& stamps,
             const std::pair<cl_int, cl_int>& imgSize, const Arguments& args);
void createScProd(const cl::Buffer& res, const cl::Buffer& weights,
                  const cl::Buffer& img,
                  const std::pair<cl_int, cl_int>& imgSize,
                  const ClData& clData, const ClStampsData& stampData,
                  const Arguments& args);
std::vector<double> createScProd(const std::vector<Stamp>& stamps,
                                 const Image& img,
                                 const std::vector<std::vector<double>>& weight,
                                 const Arguments& args);
void calcSigs(const cl::Buffer& tImgBuf, const cl::Buffer& sImgBuf,
              const std::pair<cl_int, cl_int>& axis, const cl::Buffer& model,
              const cl::Buffer& kernSol, const cl::Buffer& sigma,
              const ClStampsData& stampData, const ClData& clData,
              const Arguments& args);
void fitKernel(Kernel& k, std::vector<Stamp>& stamps, const Image& sImg,
               const cl::Buffer& tImgBuf, const cl::Buffer& sImgBuf,
               ClData& clData, const ClStampsData& stampData,
               const Arguments& args);
bool checkFitSolution(std::vector<Stamp>& stamps,
                      const std::pair<cl_int, cl_int>& axis,
                      const ClData& clData, const ClStampsData& stampData,
                      const cl::Buffer& tImgBuf, const cl::Buffer& sImgBuf,
                      const cl::Buffer& kernSol, const Arguments& args);
void removeBadSubStamps(bool* check, const ClStampsData& stampData,
                        std::vector<Stamp>& stamps,
                        const std::vector<cl_uchar>& invalidatedSubStamps,
                        const std::pair<cl_int, cl_int>& axis,
                        const cl::Buffer& sImgBuf, const cl::Buffer& tImgBuf,
                        const ClData& clData, const Arguments& args);
