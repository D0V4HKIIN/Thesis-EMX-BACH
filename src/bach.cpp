#include <CL/opencl.hpp>

#include <iterator>
#include <iostream>
#include <vector>

#include "fitsUtil.h"
#include "clUtil.h"
#include "argsUtil.h"
#include "bachUtil.h"

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

void sss(const Image &templateImg, const Image &scienceImg, ImageMask &mask, std::vector<Stamp> &templateStamps, std::vector<Stamp> &sciStamps, Arguments& args, ClData& clData) {
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

  templateStamps.reserve(args.stampsx * args.stampsy);
  sciStamps.reserve(args.stampsx * args.stampsy);
  
  constexpr int statsCount{5};
  int subStampMaxCount{2 * args.maxKSStamps};

  clData.tmpl.stampCoords    = cl::Buffer(clData.context, CL_MEM_READ_WRITE, sizeof(cl_long2) * args.stampsx * args.stampsy);
  clData.tmpl.stampSizes     = cl::Buffer(clData.context, CL_MEM_READ_WRITE, sizeof(cl_long2) * args.stampsx * args.stampsy);
  clData.tmpl.stampStats     = cl::Buffer(clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * statsCount * args.stampsx * args.stampsy);
  clData.tmpl.subStampCoords = cl::Buffer(clData.context, CL_MEM_READ_WRITE, sizeof(cl_int2) * subStampMaxCount * args.stampsx * args.stampsy);
  clData.tmpl.subStampValues = cl::Buffer(clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * subStampMaxCount * args.stampsx * args.stampsy);
  clData.tmpl.subStampCounts = cl::Buffer(clData.context, CL_MEM_READ_WRITE, sizeof(cl_int) * args.stampsx * args.stampsy);

  clData.sci.stampCoords     = cl::Buffer(clData.context, CL_MEM_READ_WRITE, sizeof(cl_long2) * args.stampsx * args.stampsy);
  clData.sci.stampSizes      = cl::Buffer(clData.context, CL_MEM_READ_WRITE, sizeof(cl_long2) * args.stampsx * args.stampsy);
  clData.sci.stampStats      = cl::Buffer(clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * statsCount * args.stampsx * args.stampsy);
  clData.sci.subStampCoords  = cl::Buffer(clData.context, CL_MEM_READ_WRITE, sizeof(cl_int2) * subStampMaxCount * args.stampsx * args.stampsy);
  clData.sci.subStampValues  = cl::Buffer(clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * subStampMaxCount * args.stampsx * args.stampsy);
  clData.sci.subStampCounts  = cl::Buffer(clData.context, CL_MEM_READ_WRITE, sizeof(cl_int) * args.stampsx * args.stampsy);


  createStamps(templateImg, scienceImg, templateStamps, sciStamps, w, h, args, clData);
  if(args.verbose) {
    std::cout << "Stamps created for " << templateImg.name << std::endl;
    std::cout << "Stamps created for " << scienceImg.name << std::endl;
  }

  /* == Check Template Stamps  ==*/
  double filledTempl{};
  double filledScience{};
  
  identifySStamps(templateStamps, templateImg, sciStamps, scienceImg, mask, &filledTempl, &filledScience, args, clData);
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
    clData.queue.enqueueWriteBuffer(clData.maskBuf, CL_TRUE, 0, sizeof(cl_ushort) * w * h, &mask);

    createStamps(templateImg, scienceImg, templateStamps, sciStamps, w, h, args, clData);

    identifySStamps(templateStamps, templateImg, sciStamps, scienceImg, mask, &filledTempl, &filledScience, args, clData);
    args.threshLow /= 0.5;
  }

  if(templateStamps.size() == 0 && sciStamps.size() == 0) {
    std::cout << "No substamps found" << std::endl;
    exit(1);
  }
}

void cmv(const Image &templateImg, const Image &scienceImg, ImageMask &mask, std::vector<Stamp> &templateStamps, std::vector<Stamp> &sciStamps, const Kernel &convolutionKernel, const Arguments& args) {
  std::cout << "\nCalculating matrix variables..." << std::endl;

  for(auto& s : templateStamps) {
    fillStamp(s, templateImg, scienceImg, mask, convolutionKernel, args);
  }
  for(auto& s : sciStamps) {
    fillStamp(s, scienceImg, templateImg, mask, convolutionKernel, args);
  }
}

bool cd(Image &templateImg, Image &scienceImg, ImageMask &mask, std::vector<Stamp> &templateStamps, std::vector<Stamp> &sciStamps, const Arguments& args) {
  std::cout << "\nChoosing convolution direction..." << std::endl;

  const double templateMerit = testFit(templateStamps, templateImg, scienceImg, mask, args);
  const double scienceMerit = testFit(sciStamps, scienceImg, templateImg, mask, args);
  if(args.verbose)
    std::cout << "template merit value = " << templateMerit
              << ", science merit value = " << scienceMerit << std::endl;

  bool convTemplate = scienceMerit > templateMerit;

  if(!convTemplate) {
    std::swap(scienceImg, templateImg);
    std::swap(sciStamps, templateStamps);
  }
  if(args.verbose)
    std::cout << templateImg.name << " chosen to be convolved." << std::endl;

  return convTemplate;
}

void ksc(const Image &templateImg, const Image &scienceImg, ImageMask &mask, std::vector<Stamp> &templateStamps, Kernel &convolutionKernel, const Arguments& args) {
  std::cout << "\nFitting kernel..." << std::endl;

  fitKernel(convolutionKernel, templateStamps, templateImg, scienceImg, mask, args);
}

double conv(const Image &templateImg, const Image &scienceImg, ImageMask &mask, Image &convImg, Kernel &convolutionKernel, bool convTemplate,
          const cl::Context &context, const cl::Program &program, cl::CommandQueue &queue, const Arguments& args) {
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
  ImageMask convMask(scienceImg.axis);
  
  for (int y = 0; y < convMask.axis.second; y++) {
    for (int x = 0; x < convMask.axis.first; x++) {
      int index = y * convMask.axis.first + x;

      if (templateImg[index] == 0.0) {
        convMask.maskPix(x, y, ImageMask::BAD_INPUT | ImageMask::BAD_PIX_VAL);
      }

      if (templateImg[index] >= args.threshHigh) {
        convMask.maskPix(x, y, ImageMask::BAD_INPUT | ImageMask::SAT_PIXEL);
      }

      if (templateImg[index] <= args.threshLow) {
        convMask.maskPix(x, y, ImageMask::BAD_INPUT | ImageMask::LOW_PIXEL);
      }
    }
  }

  // Declare all the buffers which will be need in opencl operations.
  cl::Buffer tImgBuf(context, CL_MEM_READ_ONLY, sizeof(cl_double) * w * h);
  
  cl::Buffer convMaskBuf(context, CL_MEM_READ_ONLY, sizeof(cl_ushort) * w * h);
  cl::Buffer kernBuf(context, CL_MEM_READ_ONLY,
                     sizeof(cl_double) * convKernels.size());
  cl::Buffer convImgBuf(context, CL_MEM_WRITE_ONLY, sizeof(cl_double) * w * h);
  cl::Buffer outMaskBuf(context, CL_MEM_WRITE_ONLY, sizeof(cl_ushort) * w * h);

  cl_int err{};
  // Write necessary data for convolution
  err = queue.enqueueWriteBuffer(kernBuf, CL_TRUE, 0,
                                 sizeof(cl_double) * convKernels.size(),
                                 &convKernels[0]);
  checkError(err);
  err = queue.enqueueWriteBuffer(tImgBuf, CL_TRUE, 0, sizeof(cl_double) * w * h,
                                 &templateImg);
  checkError(err);
  err = queue.enqueueWriteBuffer(convMaskBuf, CL_TRUE, 0, sizeof(cl_ushort) * w * h,
                                 &convMask);
  checkError(err);

  cl::KernelFunctor<cl::Buffer, cl_long, cl_long, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl_long,
                    cl_long>
      convFunc{program, "conv"};
  cl::EnqueueArgs eargs{queue, cl::NullRange, cl::NDRange(w * h), cl::NullRange};
  cl::Event convEvent = convFunc(eargs, kernBuf, args.fKernelWidth, xSteps, tImgBuf, convImgBuf, convMaskBuf, outMaskBuf, w, h);
  convEvent.wait();

  err = queue.enqueueReadBuffer(convImgBuf, CL_TRUE, 0,
                                sizeof(cl_double) * w * h, &convImg);
  checkError(err);
  err = queue.enqueueReadBuffer(outMaskBuf, CL_TRUE, 0, sizeof(cl_ushort) * w * h,
                                 &mask);
  checkError(err);

  // Add background and scale by kernel sum for output of convoluted image.
  for(int y = args.hKernelWidth; y < h - args.hKernelWidth; y++) {
    for(int x = args.hKernelWidth; x < w - args.hKernelWidth; x++) {
      convImg.data[x + y * w] +=
          getBackground(x, y, convolutionKernel.solution, templateImg.axis, args);
    }
  }

  for (int y = 0; y < convMask.axis.second; y++) {
    for (int x = 0; x < convMask.axis.first; x++) {
      int index = y * convMask.axis.first + x;

      if (scienceImg[index] == 0.0) {
        mask.maskPix(x, y, ImageMask::BAD_OUTPUT | ImageMask::BAD_INPUT | ImageMask::BAD_PIX_VAL);
      }

      if (scienceImg[index] >= args.threshHigh) {
        mask.maskPix(x, y, ImageMask::BAD_OUTPUT | ImageMask::BAD_INPUT | ImageMask::SAT_PIXEL);
      }

      if (scienceImg[index] <= args.threshLow) {
        mask.maskPix(x, y, ImageMask::BAD_OUTPUT | ImageMask::BAD_INPUT | ImageMask::LOW_PIXEL);
      }
    }
  }

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
