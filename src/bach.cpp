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
  std::vector<int> kernelGauss{};
  std::vector<int> kernelX{};
  std::vector<int> kernelY{};

  for(int gauss = 0; gauss < args.dg.size(); gauss++) {
    for(int x = 0; x <= args.dg[gauss]; x++) {
      for(int y = 0; y <= args.dg[gauss] - x; y++) {
        kernelGauss.push_back(gauss);
        kernelX.push_back(x);
        kernelY.push_back(y);
      }
    }
  }

  clData.gaussCount = kernelGauss.size();

  // Upload kernel status to GPU
  clData.kernel.gauss = cl::Buffer(clData.context, CL_MEM_READ_ONLY, sizeof(cl_int) * kernelGauss.size());
  clData.kernel.x = cl::Buffer(clData.context, CL_MEM_READ_ONLY, sizeof(cl_int) * kernelX.size());
  clData.kernel.y = cl::Buffer(clData.context, CL_MEM_READ_ONLY, sizeof(cl_int) * kernelY.size());
  clData.kernel.bg = cl::Buffer(clData.context, CL_MEM_READ_ONLY, sizeof(cl_float) * args.bg.size());

  cl_int err;
  err = clData.queue.enqueueWriteBuffer(clData.kernel.gauss, CL_TRUE, 0, sizeof(cl_int) * kernelGauss.size(), &kernelGauss[0]);
  checkError(err);
  err = clData.queue.enqueueWriteBuffer(clData.kernel.x, CL_TRUE, 0, sizeof(cl_int) * kernelX.size(), &kernelX[0]);
  checkError(err);
  err = clData.queue.enqueueWriteBuffer(clData.kernel.y, CL_TRUE, 0, sizeof(cl_int) * kernelY.size(), &kernelY[0]);
  checkError(err);
  err = clData.queue.enqueueWriteBuffer(clData.kernel.bg, CL_TRUE, 0, sizeof(cl_float) * args.bg.size(), &args.bg[0]);
  checkError(err);

  // Generate background X/Y
  std::vector<int> bgXY;

  for(int x = 0; x <= args.backgroundOrder; x++) {
    for(int y = 0; y <= args.backgroundOrder - x; y++) {
      bgXY.push_back(x);
      bgXY.push_back(y);
    }
  }

  clData.bg.xy = cl::Buffer(clData.context, CL_MEM_READ_ONLY, sizeof(cl_int) * bgXY.size());
  clData.bg.count = bgXY.size();

  err = clData.queue.enqueueWriteBuffer(clData.bg.xy, CL_TRUE, 0, sizeof(cl_int) * clData.bg.count, bgXY.data());
  checkError(err);

  // Create kernel filter
  clData.kernel.filterX = cl::Buffer(clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * clData.gaussCount * args.fKernelWidth);
  clData.kernel.filterY = cl::Buffer(clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * clData.gaussCount * args.fKernelWidth);

  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl_long>
      filterFunc(clData.program, "createKernelFilter");
  cl::EnqueueArgs filterEargs(clData.queue, cl::NullRange, cl::NDRange(clData.gaussCount), cl::NullRange);
  cl::Event filterEvent = filterFunc(filterEargs, clData.kernel.gauss, clData.kernel.x, clData.kernel.y,
                                     clData.kernel.bg, clData.kernel.filterX, clData.kernel.filterY,
                                     args.fKernelWidth);
  filterEvent.wait();

  // Create kernel vector
  clData.kernel.vec = cl::Buffer(clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * clData.gaussCount * args.fKernelWidth * args.fKernelWidth);

  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl_long>
      vecFunc(clData.program, "createKernelVector");
  cl::EnqueueArgs vecEargs(clData.queue, cl::NullRange, cl::NDRange(args.fKernelWidth, args.fKernelWidth, clData.gaussCount), cl::NullRange);
  cl::Event vecEvent = vecFunc(vecEargs, clData.kernel.x, clData.kernel.y,
                               clData.kernel.filterX, clData.kernel.filterY,
                               clData.kernel.vec, args.fKernelWidth);

  vecEvent.wait();

  // TEMP: return data to CPU
  std::vector<double> filterX(clData.gaussCount * args.fKernelWidth, 0.0);
  std::vector<double> filterY(clData.gaussCount * args.fKernelWidth, 0.0);
  std::vector<double> vec(clData.gaussCount * args.fKernelWidth * args.fKernelWidth, 0.0);
  
  err = clData.queue.enqueueReadBuffer(clData.kernel.filterX, CL_TRUE, 0, sizeof(cl_double) * filterX.size(), &filterX[0]);
  checkError(err);
  err = clData.queue.enqueueReadBuffer(clData.kernel.filterY, CL_TRUE, 0, sizeof(cl_double) * filterY.size(), &filterY[0]);
  checkError(err);
  err = clData.queue.enqueueReadBuffer(clData.kernel.vec, CL_TRUE, 0, sizeof(cl_double) * vec.size(), &vec[0]);
  checkError(err);

  for (int i = 0; i < filterX.size(); i++) {
    ((Kernel&)convolutionKernel).filterX[i / args.fKernelWidth][i % args.fKernelWidth] = filterX[i];
  }
  
  for (int i = 0; i < filterY.size(); i++) {
    ((Kernel&)convolutionKernel).filterY[i / args.fKernelWidth][i % args.fKernelWidth] = filterY[i];
  }

  for (int i = 0; i < vec.size(); i++) {
    ((Kernel&)convolutionKernel).kernVec[i / (args.fKernelWidth * args.fKernelWidth)][i % (args.fKernelWidth * args.fKernelWidth)] = vec[i];
  }

  fillStamps(templateStamps, templateImg, scienceImg, clData.tImgBuf, clData.sImgBuf, mask, convolutionKernel, clData, clData.tmpl, args);

  fillStamps(sciStamps, scienceImg, templateImg, clData.sImgBuf, clData.tImgBuf, mask, convolutionKernel, clData, clData.sci, args);
}

bool cd(Image &templateImg, Image &scienceImg, ImageMask &mask, std::vector<Stamp> &templateStamps, std::vector<Stamp> &sciStamps, ClData &clData, const Arguments& args) {
  std::cout << "\nChoosing convolution direction..." << std::endl;

  // Create kernel XY
  std::vector<cl_int> kernelXy{};

  for (int i = 0; i <= args.kernelOrder; i++) {
      for(int j = 0; j <= args.kernelOrder - i; j++) {
        kernelXy.push_back(i);
        kernelXy.push_back(j);
      }
  }

  // Upload kernel XY
  clData.cd.kernelXy = cl::Buffer(clData.context, CL_MEM_READ_ONLY, sizeof(cl_int) * kernelXy.size());
  clData.queue.enqueueWriteBuffer(clData.cd.kernelXy, CL_TRUE, 0, sizeof(cl_int) * kernelXy.size(), kernelXy.data());

  const double templateMerit = testFit(templateStamps, templateImg, scienceImg, clData.tImgBuf, clData.sImgBuf, mask, clData, clData.tmpl, args);
  const double scienceMerit = testFit(sciStamps, scienceImg, templateImg, clData.sImgBuf, clData.tImgBuf, mask, clData, clData.sci, args);
  if(args.verbose)
    std::cout << "template merit value = " << templateMerit
              << ", science merit value = " << scienceMerit << std::endl;

  bool convTemplate = scienceMerit > templateMerit;

  if(!convTemplate) {
    std::swap(scienceImg, templateImg);
    std::swap(sciStamps, templateStamps);
    std::swap(clData.sci, clData.tmpl);
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
