#include "bach.h"

#include <CL/opencl.hpp>
#include <iostream>
#include <iterator>
#include <vector>

#include "argsUtil.h"
#include "bachUtil.h"
#include "clUtil.h"
#include "fitsUtil.h"
#include "mathUtil.h"

void init(Image &templateImg, Image &scienceImg, ClData &clData,
          const Arguments &args) {
  if(templateImg.axis != scienceImg.axis) {
    std::cout << "Template image and science image must be the same size!"
              << std::endl;
    std::exit(1);
  }

  int pixelCount = templateImg.axis.first * templateImg.axis.second;

  // Upload buffers
  clData.queue.enqueueWriteBuffer(clData.tImgBuf, CL_TRUE, 0,
                                  sizeof(cl_double) * pixelCount, &templateImg);
  clData.queue.enqueueWriteBuffer(clData.sImgBuf, CL_TRUE, 0,
                                  sizeof(cl_double) * pixelCount, &scienceImg);

  maskInput(templateImg.axis, clData, args);
}

void sssCl(const std::pair<cl_int, cl_int> &axis,
           std::vector<Stamp> &templateStamps, std::vector<Stamp> &sciStamps,
           const Arguments &args, ClData &clData) {
  std::cout << "\nCreating stamps..." << std::endl;

  const auto [w, h] = axis;

  templateStamps.reserve(args.stampsx * args.stampsy);
  sciStamps.reserve(args.stampsx * args.stampsy);

  createStamps(w, h, clData.tmpl, clData, args);
  createStamps(w, h, clData.sci, clData, args);
  if(args.verbose) {
    std::cout << "Stamps created for template image" << std::endl;
    std::cout << "Stamps created for science image" << std::endl;
  }

  /* == Check Template Stamps  == */

  identifySStamps(axis, args, clData);

  int oldCount = args.stampsx * args.stampsy;
  removeEmptyStamps(args, clData.tmpl, clData);
  removeEmptyStamps(args, clData.sci, clData);

  double filledTempl{static_cast<double>(clData.tmpl.stampCount) / oldCount};
  double filledScience{static_cast<double>(clData.sci.stampCount) / oldCount};

  if(args.verbose) {
    std::cout << "Non-Empty template stamps: " << clData.tmpl.stampCount
              << std::endl;
    std::cout << "Non-Empty science stamps: " << clData.sci.stampCount
              << std::endl;
  }

  if(filledTempl < 0.1 || filledScience < 0.1) {
    if(args.verbose)
      std::cout << "Not enough substamps found in images, "
                << "trying again with lower thresholds..." << std::endl;
    exit(-1);
    // args.threshLow *= 0.5;

    templateStamps.clear();
    sciStamps.clear();

    resetSStampSkipMask(w, h, clData);

    createStamps(w, h, clData.tmpl, clData, args);
    createStamps(w, h, clData.sci, clData, args);

    identifySStamps(axis, args, clData);

    removeEmptyStamps(args, clData.tmpl, clData);
    removeEmptyStamps(args, clData.sci, clData);
    // args.threshLow /= 0.5;
  }

  readFinalStamps(templateStamps, clData.tmpl, clData, args);
  readFinalStamps(sciStamps, clData.sci, clData, args);

  if(templateStamps.size() == 0 && sciStamps.size() == 0) {
    std::cout << "No substamps found" << std::endl;
    std::exit(1);
  }
}

void sssMp(std::vector<Stamp> &templateStamps, const Image &templateImg,
           std::vector<Stamp> &scienceStamps, const Image &scienceImg,
           ImageMask &mask, const Arguments &args) {
  std::cout << "\nCreating stamps..." << std::endl;

  const auto [w, h] = templateImg.axis;

  computeStamps(w, h, templateImg, templateStamps, scienceImg, scienceStamps,
                mask, args);

  int maxStampsCount = args.stampsx * args.stampsy;

  templateStamps.erase(
      std::remove_if(templateStamps.begin(), templateStamps.end(),
                     [](Stamp &s) { return s.subStamps.empty(); }),
      templateStamps.end());
  scienceStamps.erase(
      std::remove_if(scienceStamps.begin(), scienceStamps.end(),
                     [](Stamp &s) { return s.subStamps.empty(); }),
      scienceStamps.end());

  double filledTemplate =
      static_cast<double>(templateStamps.size()) / maxStampsCount;
  double filledScience =
      static_cast<double>(scienceStamps.size()) / maxStampsCount;

  if(args.verbose) {
    std::cout << "Non-Empty template stamps: " << templateStamps.size()
              << std::endl;
    std::cout << "Non-Empty science stamps: " << scienceStamps.size()
              << std::endl;
  }

  // TODO: retry if not enough stamps have a substamp
  // this is not finished!!
  if(filledTemplate < 0.1 || filledScience < 0.1) {
    if(args.verbose)
      std::cout << "Not enough substamps found in images, "
                << "trying again with lower thresholds..." << std::endl;
    exit(-1);
    // args.threshLow *= 0.5;

    templateStamps.clear();
    scienceStamps.clear();

    computeStamps(w, h, templateImg, templateStamps, scienceImg, scienceStamps,
                  mask, args);
  }
}

void computeStamps(const int w, const int h, const Image &templateImg,
                   std::vector<Stamp> &templateStamps, const Image &scienceImg,
                   std::vector<Stamp> &scienceStamps, ImageMask &mask,
                   const Arguments &args) {
  templateStamps.resize(args.stampsx * args.stampsy, Stamp{});
  scienceStamps.resize(args.stampsx * args.stampsy, Stamp{});
#pragma omp parallel for collapse(2) default(none)                             \
    shared(w, h, templateImg, templateStamps, scienceImg, scienceStamps, args, \
               mask)
  for(int stampY = 0; stampY < args.stampsy; stampY++) {
    for(int stampX = 0; stampX < args.stampsx; stampX++) {
      size_t i = stampX + stampY * args.stampsx;
      Stamp &templateStamp = templateStamps[i];
      Stamp &scienceStamp = scienceStamps[i];

      createStampsMp(stampX, stampY, w, h, templateStamp, args);
      createStampsMp(stampX, stampY, w, h, scienceStamp, args);

      identifySStampsMp(templateStamp, templateImg, scienceStamp, scienceImg,
                        mask, args);
    }
  }
}

void cmv(const std::pair<cl_int, cl_int> &axis,
         std::vector<Stamp> &templateStamps, std::vector<Stamp> &sciStamps,
         ClData &clData, const Arguments &args) {
  std::cout << "\nCalculating matrix variables..." << std::endl;

  // Generate kernel stats
  std::vector<int> kernelGaussCpu{};
  std::vector<cl_int2> kernelXy{};

  for(size_t gauss = 0; gauss < args.dg.size(); gauss++) {
    for(int x = 0; x <= args.dg[gauss]; x++) {
      for(int y = 0; y <= args.dg[gauss] - x; y++) {
        kernelGaussCpu.push_back(gauss);
        kernelXy.push_back({x, y});
      }
    }
  }

  clData.gaussCount = kernelGaussCpu.size();

  // Upload kernel status to GPU
  cl::Buffer kernelGauss(clData.context, CL_MEM_READ_ONLY,
                         sizeof(cl_int) * kernelGaussCpu.size());
  clData.kernel.xy = cl::Buffer(clData.context, CL_MEM_READ_ONLY,
                                sizeof(cl_int2) * kernelXy.size());
  cl::Buffer kernelBg(clData.context, CL_MEM_READ_ONLY,
                      sizeof(cl_float) * args.bg.size());

  clData.queue.enqueueWriteBuffer(kernelGauss, CL_TRUE, 0,
                                  sizeof(cl_int) * kernelGaussCpu.size(),
                                  kernelGaussCpu.data());
  clData.queue.enqueueWriteBuffer(clData.kernel.xy, CL_TRUE, 0,
                                  sizeof(cl_int2) * kernelXy.size(),
                                  kernelXy.data());
  clData.queue.enqueueWriteBuffer(
      kernelBg, CL_TRUE, 0, sizeof(cl_float) * args.bg.size(), args.bg.data());

  // Generate background X/Y
  std::vector<cl_int2> bgXY;

  for(int x = 0; x <= args.backgroundOrder; x++) {
    for(int y = 0; y <= args.backgroundOrder - x; y++) {
      bgXY.push_back({x, y});
    }
  }

  clData.bg.xy = cl::Buffer(clData.context, CL_MEM_READ_ONLY,
                            sizeof(cl_int2) * bgXY.size());
  clData.queue.enqueueWriteBuffer(clData.bg.xy, CL_TRUE, 0,
                                  sizeof(cl_int2) * bgXY.size(), bgXY.data());

  // Create kernel filter
  clData.kernel.filterX =
      cl::Buffer(clData.context, CL_MEM_READ_WRITE,
                 sizeof(cl_double) * clData.gaussCount * args.fKernelWidth);
  clData.kernel.filterY =
      cl::Buffer(clData.context, CL_MEM_READ_WRITE,
                 sizeof(cl_double) * clData.gaussCount * args.fKernelWidth);

  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                    cl_int>
      filterFunc(clData.program, "createKernelFilter");
  cl::EnqueueArgs filterEargs(clData.queue, cl::NDRange(clData.gaussCount));
  cl::Event filterEvent = filterFunc(filterEargs, kernelGauss, clData.kernel.xy,
                                     kernelBg, clData.kernel.filterX,
                                     clData.kernel.filterY, args.fKernelWidth);
  filterEvent.wait();

  // Create kernel vector
  clData.kernel.vec = cl::Buffer(clData.context, CL_MEM_READ_WRITE,
                                 sizeof(cl_double) * clData.gaussCount *
                                     args.fKernelWidth * args.fKernelWidth);

  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl_int>
      vecFunc(clData.program, "createKernelVector");
  cl::EnqueueArgs vecEargs(
      clData.queue,
      cl::NDRange(args.fKernelWidth, args.fKernelWidth, clData.gaussCount));
  cl::Event vecEvent =
      vecFunc(vecEargs, clData.kernel.xy, clData.kernel.filterX,
              clData.kernel.filterY, clData.kernel.vec, args.fKernelWidth);

  vecEvent.wait();

  clData.cmv.yConvTmp = cl::Buffer(
      clData.context, CL_MEM_READ_WRITE,
      sizeof(cl_float) * std::max(templateStamps.size(), sciStamps.size()) *
          clData.gaussCount *
          (2 * (args.hSStampWidth + args.hKernelWidth) + 1) *
          (2 * args.hSStampWidth + 1));

  initFillStamps(templateStamps, axis, clData.tImgBuf, clData.sImgBuf, clData,
                 clData.tmpl, args);

  initFillStamps(sciStamps, axis, clData.sImgBuf, clData.tImgBuf, clData,
                 clData.sci, args);
}

bool cd(Image &templateImg, Image &scienceImg,
        std::vector<Stamp> &templateStamps, std::vector<Stamp> &sciStamps,
        ClData &clData, const Arguments &args) {
  std::cout << "\nChoosing convolution direction..." << std::endl;

  // Create kernel XY
  std::vector<cl_int2> kernelXy{};

  for(int i = 0; i <= args.kernelOrder; i++) {
    for(int j = 0; j <= args.kernelOrder - i; j++) {
      kernelXy.push_back({i, j});
    }
  }

  // Upload kernel XY
  clData.cd.kernelXy = cl::Buffer(clData.context, CL_MEM_READ_ONLY,
                                  sizeof(cl_int2) * kernelXy.size());
  clData.queue.enqueueWriteBuffer(clData.cd.kernelXy, CL_TRUE, 0,
                                  sizeof(cl_int2) * kernelXy.size(),
                                  kernelXy.data());

  const double templateMerit =
      testFit(templateStamps, templateImg.axis, clData.tImgBuf, clData.sImgBuf,
              clData, clData.tmpl, args);
  const double scienceMerit =
      testFit(sciStamps, scienceImg.axis, clData.sImgBuf, clData.tImgBuf,
              clData, clData.sci, args);

  std::cout << "template merit value = " << templateMerit
            << ", science merit value = " << scienceMerit << std::endl;

  bool convTemplate = scienceMerit > templateMerit;

  if(!convTemplate) {
    std::swap(scienceImg, templateImg);
    std::swap(sciStamps, templateStamps);
    std::swap(clData.sImgBuf, clData.tImgBuf);
    std::swap(clData.sci, clData.tmpl);
  }
  if(args.verbose)
    std::cout << templateImg.name << " chosen to be convolved." << std::endl;

  return convTemplate;
}

void ksc(std::vector<Stamp> &templateStamps, Kernel &convolutionKernel,
         const Image &sImg, const cl::Buffer &tImgBuf,
         const cl::Buffer &sImgBuf, ClData &clData,
         const ClStampsData &stampData, const Arguments &args) {
  std::cout << "\nFitting kernel..." << std::endl;

  fitKernel(convolutionKernel, templateStamps, sImg, tImgBuf, sImgBuf, clData,
            stampData, args);
}

double conv(const std::pair<cl_int, cl_int> &imgSize,
            const Image &templateImage, const Image &scienceImage,
            Image &convImg, Kernel &convolutionKernel, bool convTemplate,
            ClData &clData, const Arguments &args) {
  std::cout << "\nConvolving..." << std::endl;

  const auto [w, h] = imgSize;
  bool scaleConv = (args.normalizeTemplate && convTemplate) ||
                   (!args.normalizeTemplate && !convTemplate);

  // Convolution kernels generated beforehand since we only need on per
  // kernelsize.
  size_t kernelSize = args.fKernelWidth * args.fKernelWidth;
  int xSteps = std::ceil(imgSize.first / double(args.fKernelWidth));
  int ySteps = std::ceil(imgSize.second / double(args.fKernelWidth));
  std::vector<cl_double> convKernels(kernelSize * xSteps * ySteps);

#pragma omp parallel for default(none)                                   \
    shared(args, xSteps, ySteps, imgSize, convolutionKernel, kernelSize, \
               convKernels)
  for(int yStep = 0; yStep < ySteps; yStep++) {
    for(int xStep = 0; xStep < xSteps; xStep++) {
      std::vector<double> currKernel(args.fKernelWidth * args.fKernelWidth);
      makeKernel(convolutionKernel, currKernel, imgSize,
                 xStep * args.fKernelWidth + 2 * args.hKernelWidth,
                 yStep * args.fKernelWidth + 2 * args.hKernelWidth, args);

      size_t index = (xStep + yStep * xSteps) * kernelSize;
      // false sharing could occur here but i think it is quite rare
      std::copy(currKernel.begin(), currKernel.end(),
                convKernels.begin() + index);
    }
  }

  std::vector<double> currKernel(args.fKernelWidth * args.fKernelWidth);
  // Used to normalize the result since the kernel sum is not always 1.
  double kernSum = makeKernel(convolutionKernel, currKernel, imgSize,
                              imgSize.first / 2, imgSize.second / 2, args);
  double invKernSum = 1.0 / kernSum;

  if(args.verbose) {
    std::cout << "Sum of kernel at (" << imgSize.first / 2 << ","
              << imgSize.second / 2 << "): " << kernSum << std::endl;
  }

  // convCl(w, h, convKernels, xSteps, scaleConv, invKernSum, convImg, args,
  //  clData);
  // convMp(w, h, convKernels, xSteps, scaleConv, invKernSum, convImg,
  //  templateImage, scienceImage, convolutionKernel.solution, args, clData);
  convSplit(w, h, convKernels, xSteps, scaleConv, invKernSum, convImg,
            templateImage, convolutionKernel.solution, args, clData);

  return kernSum;
}

void sub(const std::pair<cl_int, cl_int> &imgSize, Image &diffImg,
         bool convTemplate, double kernSum, const ClData &clData,
         const Arguments &args) {
  std::cout << "\nSubtracting images..." << std::endl;

  const auto [w, h] = imgSize;
  bool scaleConv = (args.normalizeTemplate && convTemplate) ||
                   (!args.normalizeTemplate && !convTemplate);

  cl::Buffer diffImgBuf(clData.context, CL_MEM_WRITE_ONLY,
                        sizeof(cl_double) * w * h);

  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl_int,
                    cl_int, cl_int, cl_double, cl_double>
      subFunc(clData.program, "sub");
  cl::EnqueueArgs eargs(clData.queue, cl::NDRange(w * h));
  cl::Event subEvent =
      subFunc(eargs, clData.sImgBuf, clData.convImg, clData.maskBuf, diffImgBuf,
              args.fKernelWidth, w, h, scaleConv ? kernSum : 1.0,
              scaleConv ? -(1.0 / kernSum) : 1.0);
  subEvent.wait();

  // Read data from subtraction
  clData.queue.enqueueReadBuffer(diffImgBuf, CL_TRUE, 0,
                                 sizeof(cl_double) * w * h, &diffImg);
}

void fin(const Image &convImg, const Image &diffImg, const Arguments &args) {
  std::cout << "\nWriting output..." << std::endl;

  writeImage(convImg, args);
  writeImage(diffImg, args);
}
