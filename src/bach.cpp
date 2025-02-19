#include <CL/opencl.hpp>

#include <iterator>
#include <iostream>
#include <vector>

#include "fitsUtil.h"
#include "clUtil.h"
#include "argsUtil.h"
#include "bachUtil.h"
#include "mathUtil.h"

#include "bach.h"

void init(Image &templateImg, Image &scienceImg, ClData& clData, const Arguments& args) {
  // Read input images
  readImage(templateImg, args);
  readImage(scienceImg, args);

  if(templateImg.axis != scienceImg.axis) {
    std::cout << "Template image and science image must be the same size!"
              << std::endl;
    std::exit(1);
  }

  int pixelCount = templateImg.axis.first * templateImg.axis.second;

  // Upload buffers
  clData.tImgBuf = cl::Buffer(clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * pixelCount);
  clData.sImgBuf = cl::Buffer(clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * pixelCount);
  clData.maskBuf = cl::Buffer(clData.context, CL_MEM_READ_WRITE, sizeof(cl_ushort) * pixelCount);
  
  clData.queue.enqueueWriteBuffer(clData.tImgBuf, CL_TRUE, 0, sizeof(cl_double) * pixelCount, &templateImg);
  clData.queue.enqueueWriteBuffer(clData.sImgBuf, CL_TRUE, 0, sizeof(cl_double) * pixelCount, &scienceImg);

  maskInput(templateImg.axis, clData, args);
}

void sss(const std::pair<cl_int, cl_int> &axis, std::vector<Stamp> &templateStamps, std::vector<Stamp> &sciStamps, Arguments& args, ClData& clData) {
  std::cout << "\nCreating stamps..." << std::endl;
    
  const auto [w, h] = axis;
  args.fStampWidth = std::min(int(axis.first / args.stampsx),
                              int(axis.second / args.stampsy));
  args.fStampWidth -= args.fKernelWidth;
  args.fStampWidth -= args.fStampWidth % 2 == 0 ? 1 : 0;

  if(args.fStampWidth < args.fSStampWidth) {
    args.fStampWidth = args.fSStampWidth + args.fKernelWidth;
    args.fStampWidth -= args.fStampWidth % 2 == 0 ? 1 : 0;

    args.stampsx = int(axis.first / args.fStampWidth);
    args.stampsy = int(axis.second / args.fStampWidth);

    if(args.verbose)
        std::cout << "Too many stamps requested, using " << args.stampsx << "x"
                  << args.stampsy << " stamps instead." << std::endl;
  }

  templateStamps.reserve(args.stampsx * args.stampsy);
  sciStamps.reserve(args.stampsx * args.stampsy);
  
  int subStampMaxCount{2 * args.maxKSStamps};

  clData.tmpl.stampCoords    = cl::Buffer(clData.context, CL_MEM_READ_WRITE, sizeof(cl_int2) * args.stampsx * args.stampsy);
  clData.tmpl.stampSizes     = cl::Buffer(clData.context, CL_MEM_READ_WRITE, sizeof(cl_int2) * args.stampsx * args.stampsy);
  clData.tmpl.stats.skyEsts  = cl::Buffer(clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * args.stampsx * args.stampsy);
  clData.tmpl.stats.fwhms    = cl::Buffer(clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * args.stampsx * args.stampsy);
  clData.tmpl.subStampCoords = cl::Buffer(clData.context, CL_MEM_READ_WRITE, sizeof(cl_int2) * subStampMaxCount * args.stampsx * args.stampsy);
  clData.tmpl.subStampValues = cl::Buffer(clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * subStampMaxCount * args.stampsx * args.stampsy);
  clData.tmpl.subStampCounts = cl::Buffer(clData.context, CL_MEM_READ_WRITE, sizeof(cl_int) * args.stampsx * args.stampsy);

  clData.sci.stampCoords     = cl::Buffer(clData.context, CL_MEM_READ_WRITE, sizeof(cl_int2) * args.stampsx * args.stampsy);
  clData.sci.stampSizes      = cl::Buffer(clData.context, CL_MEM_READ_WRITE, sizeof(cl_int2) * args.stampsx * args.stampsy);
  clData.sci.stats.skyEsts   = cl::Buffer(clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * args.stampsx * args.stampsy);
  clData.sci.stats.fwhms     = cl::Buffer(clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * args.stampsx * args.stampsy);
  clData.sci.subStampCoords  = cl::Buffer(clData.context, CL_MEM_READ_WRITE, sizeof(cl_int2) * subStampMaxCount * args.stampsx * args.stampsy);
  clData.sci.subStampValues  = cl::Buffer(clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * subStampMaxCount * args.stampsx * args.stampsy);
  clData.sci.subStampCounts  = cl::Buffer(clData.context, CL_MEM_READ_WRITE, sizeof(cl_int) * args.stampsx * args.stampsy);

  createStamps(templateStamps, w, h, clData.tmpl, clData, args);
  createStamps(sciStamps, w, h, clData.sci, clData, args);
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
    std::cout << "Non-Empty template stamps: " << clData.tmpl.stampCount << std::endl;
    std::cout << "Non-Empty science stamps: " << clData.sci.stampCount << std::endl;
  }
  
  if(filledTempl < 0.1 || filledScience < 0.1) {
    if(args.verbose)
      std::cout << "Not enough substamps found in images, "
                << "trying again with lower thresholds..." << std::endl;
    args.threshLow *= 0.5;
    
    templateStamps.clear();
    sciStamps.clear();

    resetSStampSkipMask(w, h, clData);

    createStamps(templateStamps, w, h, clData.tmpl, clData, args);
    createStamps(sciStamps, w, h, clData.sci, clData, args);

    identifySStamps(axis, args, clData);

    removeEmptyStamps(args, clData.tmpl, clData);
    removeEmptyStamps(args, clData.sci, clData);
    args.threshLow /= 0.5;
  }

  readFinalStamps(templateStamps, clData.tmpl, clData, args);
  readFinalStamps(sciStamps, clData.sci, clData, args);

  if(templateStamps.size() == 0 && sciStamps.size() == 0) {
    std::cout << "No substamps found" << std::endl;
    std::exit(1);
  }
}

void cmv(const std::pair<cl_int, cl_int> &axis, std::vector<Stamp> &templateStamps, std::vector<Stamp> &sciStamps, const Kernel &convolutionKernel, ClData &clData, const Arguments& args) {
  std::cout << "\nCalculating matrix variables..." << std::endl;

  // Generate kernel stats
  std::vector<int> kernelGaussCpu{};
  std::vector<cl_int2> kernelXy{};

  for(int gauss = 0; gauss < args.dg.size(); gauss++) {
    for(int x = 0; x <= args.dg[gauss]; x++) {
      for(int y = 0; y <= args.dg[gauss] - x; y++) {
        kernelGaussCpu.push_back(gauss);
        kernelXy.push_back({ x, y });
      }
    }
  }

  clData.gaussCount = kernelGaussCpu.size();

  // Upload kernel status to GPU
  cl::Buffer kernelGauss(clData.context, CL_MEM_READ_ONLY, sizeof(cl_int) * kernelGaussCpu.size());
  clData.kernel.xy = cl::Buffer(clData.context, CL_MEM_READ_ONLY, sizeof(cl_int2) * kernelXy.size());
  cl::Buffer kernelBg(clData.context, CL_MEM_READ_ONLY, sizeof(cl_float) * args.bg.size());

  clData.queue.enqueueWriteBuffer(kernelGauss, CL_TRUE, 0, sizeof(cl_int) * kernelGaussCpu.size(), kernelGaussCpu.data());
  clData.queue.enqueueWriteBuffer(clData.kernel.xy, CL_TRUE, 0, sizeof(cl_int2) * kernelXy.size(), kernelXy.data());
  clData.queue.enqueueWriteBuffer(kernelBg, CL_TRUE, 0, sizeof(cl_float) * args.bg.size(), args.bg.data());

  // Generate background X/Y
  std::vector<cl_int2> bgXY;

  for(int x = 0; x <= args.backgroundOrder; x++) {
    for(int y = 0; y <= args.backgroundOrder - x; y++) {
      bgXY.push_back({ x, y });
    }
  }

  clData.bg.xy = cl::Buffer(clData.context, CL_MEM_READ_ONLY, sizeof(cl_int2) * bgXY.size());
  clData.queue.enqueueWriteBuffer(clData.bg.xy, CL_TRUE, 0, sizeof(cl_int2) * bgXY.size(), bgXY.data());

  // Create kernel filter
  clData.kernel.filterX = cl::Buffer(clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * clData.gaussCount * args.fKernelWidth);
  clData.kernel.filterY = cl::Buffer(clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * clData.gaussCount * args.fKernelWidth);

  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl_int>
      filterFunc(clData.program, "createKernelFilter");
  cl::EnqueueArgs filterEargs(clData.queue, cl::NDRange(clData.gaussCount));
  cl::Event filterEvent = filterFunc(filterEargs, kernelGauss, clData.kernel.xy,
                                     kernelBg, clData.kernel.filterX, clData.kernel.filterY,
                                     args.fKernelWidth);
  filterEvent.wait();

  // Create kernel vector
  clData.kernel.vec = cl::Buffer(clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * clData.gaussCount * args.fKernelWidth * args.fKernelWidth);

  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl_int>
      vecFunc(clData.program, "createKernelVector");
  cl::EnqueueArgs vecEargs(clData.queue, cl::NDRange(args.fKernelWidth, args.fKernelWidth, clData.gaussCount));
  cl::Event vecEvent = vecFunc(vecEargs, clData.kernel.xy,
                               clData.kernel.filterX, clData.kernel.filterY,
                               clData.kernel.vec, args.fKernelWidth);

  vecEvent.wait();
  
  clData.cmv.yConvTmp = cl::Buffer(clData.context, CL_MEM_READ_WRITE, sizeof(cl_float) * std::max(templateStamps.size(), sciStamps.size()) * clData.gaussCount * (2 * (args.hSStampWidth + args.hKernelWidth) + 1) * (2 * args.hSStampWidth + 1));
  
  initFillStamps(templateStamps, axis, clData.tImgBuf, clData.sImgBuf, convolutionKernel, clData, clData.tmpl, args);

  initFillStamps(sciStamps, axis, clData.sImgBuf, clData.tImgBuf, convolutionKernel, clData, clData.sci, args);
}

bool cd(Image &templateImg, Image &scienceImg, std::vector<Stamp> &templateStamps, std::vector<Stamp> &sciStamps, ClData &clData, const Arguments& args) {
  std::cout << "\nChoosing convolution direction..." << std::endl;

  // Create kernel XY
  std::vector<cl_int2> kernelXy{};

  for (int i = 0; i <= args.kernelOrder; i++) {
      for(int j = 0; j <= args.kernelOrder - i; j++) {
        kernelXy.push_back({ i, j });
      }
  }

  // Upload kernel XY
  clData.cd.kernelXy = cl::Buffer(clData.context, CL_MEM_READ_ONLY, sizeof(cl_int2) * kernelXy.size());
  clData.queue.enqueueWriteBuffer(clData.cd.kernelXy, CL_TRUE, 0, sizeof(cl_int2) * kernelXy.size(), kernelXy.data());

  const double templateMerit = testFit(templateStamps, templateImg.axis, clData.tImgBuf, clData.sImgBuf, clData, clData.tmpl, args);
  const double scienceMerit = testFit(sciStamps, scienceImg.axis, clData.sImgBuf, clData.tImgBuf, clData, clData.sci, args);
  
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

void ksc(std::vector<Stamp> &templateStamps, Kernel &convolutionKernel, const Image &sImg, const cl::Buffer &tImgBuf, const cl::Buffer &sImgBuf,
         ClData &clData, const ClStampsData &stampData, const Arguments& args) {
  std::cout << "\nFitting kernel..." << std::endl;

  fitKernel(convolutionKernel, templateStamps, sImg, tImgBuf, sImgBuf, clData, stampData, args);
}

double conv(const std::pair<cl_int, cl_int> &imgSize, Image &convImg, Kernel &convolutionKernel, bool convTemplate,
            ClData &clData, const Arguments& args) {
  std::cout << "\nConvolving..." << std::endl;
  
  const auto [w, h] = imgSize;
  bool scaleConv = args.normalizeTemplate && convTemplate ||
                   !args.normalizeTemplate && !convTemplate;

  // Convolution kernels generated beforehand since we only need on per
  // kernelsize.
  std::vector<cl_double> convKernels{};
  int xSteps = std::ceil(imgSize.first / double(args.fKernelWidth));
  int ySteps = std::ceil(imgSize.second / double(args.fKernelWidth));
  for(int yStep = 0; yStep < ySteps; yStep++) {
    for(int xStep = 0; xStep < xSteps; xStep++) {
      makeKernel(
          convolutionKernel, imgSize,
          xStep * args.fKernelWidth + args.hKernelWidth + args.hKernelWidth,
          yStep * args.fKernelWidth + args.hKernelWidth + args.hKernelWidth,
          args);
      convKernels.insert(convKernels.end(),
                         convolutionKernel.currKernel.begin(),
                         convolutionKernel.currKernel.end());
    }
  }

  // Used to normalize the result since the kernel sum is not always 1.
  double kernSum =
      makeKernel(convolutionKernel, imgSize,
                 imgSize.first / 2, imgSize.second / 2, args);
  double invKernSum = 1.0 / kernSum;

  if(args.verbose) {
    std::cout << "Sum of kernel at (" << imgSize.first / 2 << ","
              << imgSize.second / 2 << "): " << kernSum << std::endl;
  }

  // Declare all the buffers which will be need in opencl operations.  
  cl::Buffer convMaskBuf(clData.context, CL_MEM_READ_ONLY, sizeof(cl_ushort) * w * h);
  cl::Buffer kernBuf(clData.context, CL_MEM_READ_ONLY, sizeof(cl_double) * convKernels.size());
  clData.convImg = cl::Buffer(clData.context, CL_MEM_WRITE_ONLY, sizeof(cl_double) * w * h);

  // Write necessary data for convolution
  clData.queue.enqueueWriteBuffer(kernBuf, CL_TRUE, 0, sizeof(cl_double) * convKernels.size(), convKernels.data());
  
  // Create convolution mask
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl_int, cl_double, cl_double> createMaskFunc(clData.program, "createConvMask");
  cl::EnqueueArgs createMaskEargs(clData.queue, cl::NDRange(w, h));
  cl::Event createMaskEvent = createMaskFunc(createMaskEargs, clData.tImgBuf, convMaskBuf, w, args.threshHigh, args.threshLow);

  createMaskEvent.wait();

  // Convolve
  cl::KernelFunctor<cl::Buffer, cl_int, cl_int, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                    cl_int, cl_int, cl_int, cl_int, cl_double> convFunc(clData.program, "conv");
  cl::EnqueueArgs eargs(clData.queue, cl::NDRange(w * h));
  cl::Event convEvent = convFunc(eargs, kernBuf, args.fKernelWidth, xSteps, clData.tImgBuf, clData.convImg, convMaskBuf, clData.maskBuf, clData.kernel.solution,
                                 w, h, args.backgroundOrder, (args.nPSF - 1) * triNum(args.kernelOrder + 1) + 1, scaleConv ? invKernSum : 1.0);
  convEvent.wait();

  // Transfer convoluted image back to CPU
  clData.queue.enqueueReadBuffer(clData.convImg, CL_TRUE, 0, sizeof(cl_double) * w * h, &convImg);

  // Mask after convolve
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl_int, cl_double, cl_double> maskAfterFunc(clData.program, "maskAfterConv");
  cl::EnqueueArgs maskAfterEargs(clData.queue, cl::NDRange(w, h));
  cl::Event maskAfterEvent = maskAfterFunc(maskAfterEargs, clData.sImgBuf, clData.maskBuf, w, args.threshHigh, args.threshLow);

  maskAfterEvent.wait();

  return kernSum;
}

void sub(const std::pair<cl_int, cl_int> &imgSize, Image &diffImg, bool convTemplate, double kernSum,
         const ClData &clData, const Arguments& args) {
  std::cout << "\nSubtracting images..." << std::endl;

  const auto [w, h] = imgSize;
  bool scaleConv = args.normalizeTemplate && convTemplate ||
                   !args.normalizeTemplate && !convTemplate;

  cl::Buffer diffImgBuf(clData.context, CL_MEM_WRITE_ONLY, sizeof(cl_double) * w * h);
  
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl_int, cl_int, cl_int,
                    cl_double, cl_double> subFunc(clData.program, "sub");
  cl::EnqueueArgs eargs(clData.queue, cl::NDRange(w * h));
  cl::Event subEvent = subFunc(eargs, clData.sImgBuf, clData.convImg, clData.maskBuf, diffImgBuf, args.fKernelWidth, w, h,
                               scaleConv ? kernSum : 1.0, scaleConv ? -(1.0 / kernSum) : 1.0);
  subEvent.wait();

  // Read data from subtraction
  clData.queue.enqueueReadBuffer(diffImgBuf, CL_TRUE, 0, sizeof(cl_double) * w * h, &diffImg);
}

void fin(const Image &convImg, const Image &diffImg, const Arguments& args) {
  std::cout << "\nWriting output..." << std::endl;

  writeImage(convImg, args);  
  writeImage(diffImg, args);
}
