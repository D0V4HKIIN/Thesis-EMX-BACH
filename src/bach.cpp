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

  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl_long>
      filterFunc(clData.program, "createKernelFilter");
  cl::EnqueueArgs filterEargs(clData.queue, cl::NullRange, cl::NDRange(clData.gaussCount), cl::NullRange);
  cl::Event filterEvent = filterFunc(filterEargs, kernelGauss, clData.kernel.xy,
                                     kernelBg, clData.kernel.filterX, clData.kernel.filterY,
                                     args.fKernelWidth);
  filterEvent.wait();

  // Create kernel vector
  clData.kernel.vec = cl::Buffer(clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * clData.gaussCount * args.fKernelWidth * args.fKernelWidth);

  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl_long>
      vecFunc(clData.program, "createKernelVector");
  cl::EnqueueArgs vecEargs(clData.queue, cl::NullRange, cl::NDRange(args.fKernelWidth, args.fKernelWidth, clData.gaussCount), cl::NullRange);
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
         const cl::Buffer &tImgBuf, const cl::Buffer &sImgBuf, const ClData &clData, const ClStampsData &stampData, const Arguments& args) {
  std::cout << "\nFitting kernel..." << std::endl;

  fitKernel(convolutionKernel, templateStamps, templateImg, scienceImg, mask, tImgBuf, sImgBuf, clData, stampData, args);
}

double conv(const Image &templateImg, const Image &scienceImg, ImageMask &mask, Image &convImg, Kernel &convolutionKernel, bool convTemplate,
          const ClData &clData, const Arguments& args) {
  std::cout << "\nConvolving..." << std::endl;
  
  const auto [w, h] = templateImg.axis;
  bool scaleConv = args.normalizeTemplate && convTemplate ||
                   !args.normalizeTemplate && !convTemplate;

  // Convolution kernels generated beforehand since we only need on per
  // kernelsize.
  std::vector<cl_double> convKernels{};
  int xSteps = std::ceil((templateImg.axis.first) / double(args.fKernelWidth));
  int ySteps = std::ceil((templateImg.axis.second) / double(args.fKernelWidth));
  for(int yStep = 0; yStep < ySteps; yStep++) {
    for(int xStep = 0; xStep < xSteps; xStep++) {
      makeKernel(
          convolutionKernel, templateImg.axis,
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
      makeKernel(convolutionKernel, templateImg.axis,
                 templateImg.axis.first / 2, templateImg.axis.second / 2, args);
  cl_double invKernSum = 1.0 / kernSum;

  if(args.verbose) {
    std::cout << "Sum of kernel at (" << templateImg.axis.first / 2 << ","
              << templateImg.axis.second / 2 << "): " << kernSum << std::endl;
  }

  mask.clear();

  // Declare all the buffers which will be need in opencl operations.  
  cl::Buffer convMaskBuf(clData.context, CL_MEM_READ_ONLY, sizeof(cl_ushort) * w * h);
  cl::Buffer kernBuf(clData.context, CL_MEM_READ_ONLY, sizeof(cl_double) * convKernels.size());
  cl::Buffer convImgBuf(clData.context, CL_MEM_WRITE_ONLY, sizeof(cl_double) * w * h);
  cl::Buffer outMaskBuf(clData.context, CL_MEM_WRITE_ONLY, sizeof(cl_ushort) * w * h);

  // Write necessary data for convolution
  clData.queue.enqueueWriteBuffer(kernBuf, CL_TRUE, 0, sizeof(cl_double) * convKernels.size(), convKernels.data());
  
  // Create convolution mask
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl_int, cl_double, cl_double> createMaskFunc(clData.program, "createConvMask");
  cl::EnqueueArgs createMaskEargs(clData.queue, cl::NDRange(w, h));
  cl::Event createMaskEvent = createMaskFunc(createMaskEargs, clData.tImgBuf, convMaskBuf, w, args.threshHigh, args.threshLow);

  createMaskEvent.wait();

  // Convolve
  cl::KernelFunctor<cl::Buffer, cl_long, cl_long, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl_long,
                    cl_long>
      convFunc(clData.program, "conv");
  cl::EnqueueArgs eargs(clData.queue, cl::NullRange, cl::NDRange(w * h), cl::NullRange);
  cl::Event convEvent = convFunc(eargs, kernBuf, args.fKernelWidth, xSteps, clData.tImgBuf, convImgBuf, convMaskBuf, outMaskBuf, w, h);
  convEvent.wait();

  clData.queue.enqueueReadBuffer(convImgBuf, CL_TRUE, 0, sizeof(cl_double) * w * h, &convImg);

  // Add background and scale by kernel sum for output of convoluted image.
  for(int y = args.hKernelWidth; y < h - args.hKernelWidth; y++) {
    for(int x = args.hKernelWidth; x < w - args.hKernelWidth; x++) {
      convImg.data[x + y * w] +=
          getBackground(x, y, convolutionKernel.solution, templateImg.axis, args);
    }
  }

  // Mask after convolve
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl_int, cl_double, cl_double> maskAfterFunc(clData.program, "maskAfterConv");
  cl::EnqueueArgs maskAfterEargs(clData.queue, cl::NDRange(w, h));
  cl::Event maskAfterEvent = maskAfterFunc(maskAfterEargs, clData.sImgBuf, outMaskBuf, w, args.threshHigh, args.threshLow);

  maskAfterEvent.wait();

  clData.queue.enqueueReadBuffer(outMaskBuf, CL_TRUE, 0, sizeof(cl_ushort) * w * h, &mask);

  if (scaleConv) {
    for(int y = args.hKernelWidth; y < h - args.hKernelWidth; y++) {
      for(int x = args.hKernelWidth; x < w - args.hKernelWidth; x++) {
        convImg.data[x + y * w] *= invKernSum;
      }
    }
  }

  return kernSum;
}

void sub(const Image &convImg, const Image &scienceImg, const ImageMask &mask, Image &diffImg, bool convTemplate, double kernSum,
         const cl::Context &context, const cl::Program &program, cl::CommandQueue &queue, const Arguments& args) {
  std::cout << "\nSubtracting images..." << std::endl;

  const auto [w, h] = scienceImg.axis;
  bool scaleConv = args.normalizeTemplate && convTemplate ||
                   !args.normalizeTemplate && !convTemplate;

  cl::Buffer convImgBuf(context, CL_MEM_READ_ONLY, sizeof(cl_double) * w * h);
  cl::Buffer diffImgBuf(context, CL_MEM_WRITE_ONLY, sizeof(cl_double) * w * h);
  cl::Buffer sImgBuf(context, CL_MEM_READ_ONLY, sizeof(cl_double) * w * h);

  cl_int err{};
  // Write necessary data for subtraction
  err = queue.enqueueWriteBuffer(convImgBuf, CL_TRUE, 0,
                                 sizeof(cl_double) * w * h, &convImg);
  err = queue.enqueueWriteBuffer(sImgBuf, CL_TRUE, 0, sizeof(cl_double) * w * h,
                                 &scienceImg);
  checkError(err);
  
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl_long, cl_long, cl_long, cl_double, cl_double> subFunc(program,
                                                                       "sub");
  cl::EnqueueArgs eargs{queue, cl::NullRange, cl::NDRange(w * h), cl::NullRange};
  cl::Event subEvent = subFunc(eargs, sImgBuf, convImgBuf, diffImgBuf, args.fKernelWidth, w, h,
                               scaleConv ? kernSum : 1.0, scaleConv ? -(1.0 / kernSum) : 1.0);
  subEvent.wait();

  // Read data from subtraction
  err = queue.enqueueReadBuffer(diffImgBuf, CL_TRUE, 0,
                                sizeof(cl_double) * w * h, &diffImg);
  checkError(err);

  for (int y = args.hKernelWidth; y < diffImg.axis.second - args.hKernelWidth; y++) {
    for (int x = args.hKernelWidth; x < diffImg.axis.first - args.hKernelWidth; x++) {
      int index = y * diffImg.axis.first + x;

      if (mask.isMasked(index, ImageMask::BAD_OUTPUT)) {
        diffImg.data[index] = 1e-30f;
      }
    }
  }
} 

void fin(const Image &convImg, const Image &diffImg, const Arguments& args) {
  std::cout << "\nWriting output..." << std::endl;

  cl_int err{};
  err = writeImage(convImg, args);
  checkError(err);
  
  err = writeImage(diffImg, args);
  checkError(err);
}
