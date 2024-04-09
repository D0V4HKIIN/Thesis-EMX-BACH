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

void init(Image &templateImg, Image &scienceImg, ImageMask &mask, ClData& clData, const Arguments& args) {

  cl_int err{};

  // Read input images
  err = readImage(templateImg, args);
  checkError(err);
  err = readImage(scienceImg, args);
  checkError(err);

  if(templateImg.axis != scienceImg.axis) {
    std::cout << "Template image and science image must be the same size!"
              << std::endl;
    exit(1);
  }
  
  mask = ImageMask(templateImg.axis);

  int pixelCount = templateImg.axis.first * templateImg.axis.second;

  // Upload buffers
  clData.tImgBuf = cl::Buffer(clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * pixelCount);
  clData.sImgBuf = cl::Buffer(clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * pixelCount);
  clData.maskBuf = cl::Buffer(clData.context, CL_MEM_READ_WRITE, sizeof(cl_ushort) * pixelCount);

  err = clData.queue.enqueueWriteBuffer(clData.tImgBuf, CL_TRUE, 0, sizeof(cl_double) * pixelCount, &templateImg);
  checkError(err);
  err = clData.queue.enqueueWriteBuffer(clData.sImgBuf, CL_TRUE, 0, sizeof(cl_double) * pixelCount, &scienceImg);
  checkError(err);

  maskInput(mask, clData, args);
}

void sss(const Image &templateImg, const Image &scienceImg, ImageMask &mask, std::vector<Stamp> &templateStamps, std::vector<Stamp> &sciStamps, Arguments& args) {
  std::cout << "\nCreating stamps..." << std::endl;
    
  const auto [w, h] = templateImg.axis;
  args.fStampWidth = std::min(int(templateImg.axis.first / args.stampsx),
                              int(templateImg.axis.second / args.stampsy));
  args.fStampWidth -= args.fKernelWidth;
  args.fStampWidth -= args.fStampWidth % 2 == 0 ? 1 : 0;

  if(args.fStampWidth < args.fSStampWidth) {
    args.fStampWidth = args.fSStampWidth + args.fKernelWidth;
    args.fStampWidth -= args.fStampWidth % 2 == 0 ? 1 : 0;

    args.stampsx = int(templateImg.axis.first / args.fStampWidth);
    args.stampsy = int(templateImg.axis.second / args.fStampWidth);

  if(args.verbose)
      std::cout << "Too many stamps requested, using " << args.stampsx << "x"
                << args.stampsy << " stamps instead." << std::endl;
  }

  createStamps(templateImg, templateStamps, w, h, args);
  if(args.verbose) {
    std::cout << "Stamps created for " << templateImg.name << std::endl;
  }

  createStamps(scienceImg, sciStamps, w, h, args);
  if(args.verbose) {
    std::cout << "Stamps created for " << scienceImg.name << std::endl;
  }

  /* == Check Template Stamps  ==*/
  double filledTempl{};
  double filledScience{};
  identifySStamps(templateStamps, templateImg, sciStamps, scienceImg, mask, &filledTempl, &filledScience, args);
  if(filledTempl < 0.1 || filledScience < 0.1) {
    if(args.verbose)
      std::cout << "Not enough substamps found in " << templateImg.name
                << " trying again with lower thresholds..." << std::endl;
    args.threshLow *= 0.5;
    
    templateStamps.clear();
    sciStamps.clear();

    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        int index = y * w + x;
        mask.unmask(index, ImageMask::SKIP_S | ImageMask::SKIP_T);
      }
    }

    createStamps(templateImg, templateStamps, w, h, args);

    createStamps(scienceImg, sciStamps, w, h, args);

    identifySStamps(templateStamps, templateImg, sciStamps, scienceImg, mask, &filledTempl, &filledScience, args);
    args.threshLow /= 0.5;
  }

  if(templateStamps.size() == 0 && sciStamps.size() == 0) {
    std::cout << "No substamps found" << std::endl;
    exit(1);
  }
}

void cmv(const Image &templateImg, const Image &scienceImg, ImageMask &mask, std::vector<Stamp> &templateStamps, std::vector<Stamp> &sciStamps, const Kernel &convolutionKernel, ClData &clData, const Arguments& args) {
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
  
  initFillStamps(templateStamps, templateImg, scienceImg, clData.tImgBuf, clData.sImgBuf, mask, convolutionKernel, clData, clData.tmpl, args);

  initFillStamps(sciStamps, scienceImg, templateImg, clData.sImgBuf, clData.tImgBuf, mask, convolutionKernel, clData, clData.sci, args);
}

bool cd(Image &templateImg, Image &scienceImg, ImageMask &mask, std::vector<Stamp> &templateStamps, std::vector<Stamp> &sciStamps, ClData &clData, const Arguments& args) {
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

  // TEMP: return mask to CPU
  clData.queue.enqueueReadBuffer(clData.maskBuf, CL_TRUE, 0, sizeof(cl_ushort) * mask.axis.first * mask.axis.second, &mask);
  
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

void ksc(const Image &templateImg, const Image &scienceImg, ImageMask &mask, std::vector<Stamp> &templateStamps, Kernel &convolutionKernel,
         const cl::Buffer &tImgBuf, const cl::Buffer &sImgBuf, ClData &clData, const ClStampsData &stampData, const Arguments& args) {
  std::cout << "\nFitting kernel..." << std::endl;

  fitKernel(convolutionKernel, templateStamps, templateImg, scienceImg, mask, tImgBuf, sImgBuf, clData, stampData, args);
}

double conv(const std::pair<cl_long, cl_long> &imgSize, Image &convImg, Kernel &convolutionKernel, bool convTemplate,
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

void sub(const std::pair<cl_long, cl_long> &imgSize, Image &diffImg, bool convTemplate, double kernSum,
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

  cl_int err{};
  err = writeImage(convImg, args);
  checkError(err);
  
  err = writeImage(diffImg, args);
  checkError(err);
}
