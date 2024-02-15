#include <time.h>

#include <CL/opencl.hpp>
#include <filesystem>
#include <iterator>
#include <vector>
#include <iostream>

#include "clUtil.h"
#include "fitsUtil.h"
#include "bachUtil.h"
#include "datatypeUtil.h"
#include "bach.h"

int main(int argc, const char* argv[]) {
  clock_t p1 = clock();

  CCfits::FITS::setVerboseMode(true);
  try {
    std::cout << "Reading in arguments..." << std::endl;
    getArguments(argc, argv);
  } catch(const std::invalid_argument& err) {
    std::cout << err.what() << '\n';
    return 1;
  }
  
  std::cout << "\nReading in images..." << std::endl;
  Image templateImg{args.templateName};
  Image scienceImg{args.scienceName};
  templateImg.path = scienceImg.path = args.inputPath + "/";
  
  
  if(args.verbose)
    std::cout << "template image name: " << args.templateName
              << ", science image name: " << args.scienceName << std::endl;

  ImageMask mask(std::make_pair(0, 0));

  std::cout << "\nSetting up openCL..." << std::endl;
  cl::Device defaultDevice{getDefaultDevice()};
  cl::Context context{defaultDevice};
  cl::Program program =
      loadBuildPrograms(context, defaultDevice, std::filesystem::path(argv[0]).parent_path(),
      "conv.cl", "sub.cl");

  init(templateImg, scienceImg, mask);
  auto [w, h] = templateImg.axis;

  clock_t p2 = clock();
  if(args.verboseTime) {
    printf("Initiation took %lds %ldms\n", (p2 - p1) / CLOCKS_PER_SEC,
           ((p2 - p1) * 1000 / CLOCKS_PER_SEC) % 1000);
  }

  /* ===== SSS ===== */

  clock_t p3 = clock();
  std::vector<Stamp> templateStamps{};
  std::vector<Stamp> sciStamps{};
  sss(templateImg, scienceImg, mask, templateStamps, sciStamps);

  clock_t p4 = clock();
  if(args.verboseTime) {
    printf("SSS took %lds %ldms\n", (p4 - p3) / CLOCKS_PER_SEC,
           ((p4 - p3) * 1000 / CLOCKS_PER_SEC) % 1000);
  }

  std::cout << std::endl;

  /* ===== CMV ===== */

  clock_t p5 = clock();

  Kernel convolutionKernel{};
  cmv(templateImg, scienceImg, mask, templateStamps, sciStamps, convolutionKernel);
  
  clock_t p6 = clock();
  if(args.verboseTime) {
    printf("CMV took %lds %ldms\n", (p6 - p5) / CLOCKS_PER_SEC,
           ((p6 - p5) * 1000 / CLOCKS_PER_SEC) % 1000);
  }

  /* ===== CD ===== */

  clock_t p7 = clock();

  cd(templateImg, scienceImg, mask, templateStamps, sciStamps);

  clock_t p8 = clock();
  if(args.verboseTime) {
    printf("CD took %lds %ldms\n", (p8 - p7) / CLOCKS_PER_SEC,
           ((p8 - p7) * 1000 / CLOCKS_PER_SEC) % 1000);
  }

  /* ===== KSC ===== */

  clock_t p9 = clock();

  ksc(templateImg, scienceImg, mask, templateStamps, convolutionKernel);

  clock_t p10 = clock();
  if(args.verboseTime) {
    printf("KSC took %lds %ldms\n", (p10 - p9) / CLOCKS_PER_SEC,
           ((p10 - p9) * 1000 / CLOCKS_PER_SEC) % 1000);
  }

  /* ===== Conv ===== */

  clock_t p11 = clock();

  std::cout << "\nConvolving..." << std::endl;

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
          yStep * args.fKernelWidth + args.hKernelWidth + args.hKernelWidth);
      convKernels.insert(convKernels.end(),
                         convolutionKernel.currKernel.begin(),
                         convolutionKernel.currKernel.end());
    }
  }

  // Used to normalize the result since the kernel sum is not always 1.
  double kernSum =
      makeKernel(convolutionKernel, templateImg.axis,
                 templateImg.axis.first / 2, templateImg.axis.second / 2);
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
  cl::Buffer sImgBuf(context, CL_MEM_READ_ONLY, sizeof(cl_double) * w * h);
  cl::Buffer convImgBuf(context, CL_MEM_READ_ONLY, sizeof(cl_double) * w * h);
  cl::Buffer convMaskBuf(context, CL_MEM_READ_ONLY, sizeof(cl_ushort) * w * h);
  cl::Buffer kernBuf(context, CL_MEM_READ_ONLY,
                     sizeof(cl_double) * convKernels.size());
  cl::Buffer outImgBuf(context, CL_MEM_WRITE_ONLY, sizeof(cl_double) * w * h);
  cl::Buffer diffImgBuf(context, CL_MEM_WRITE_ONLY, sizeof(cl_double) * w * h);
  cl::Buffer outMaskBuf(context, CL_MEM_WRITE_ONLY, sizeof(cl_ushort) * w * h);

  cl::CommandQueue queue(context, defaultDevice);

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
      conv{program, "conv"};
  cl::EnqueueArgs eargs{queue, cl::NullRange, cl::NDRange(w * h),
                        cl::NullRange};
  cl::Event convEvent = conv(eargs, kernBuf, args.fKernelWidth, xSteps, tImgBuf, outImgBuf, convMaskBuf, outMaskBuf, w, h);
  convEvent.wait();

  // Read data from convolution
  Image outImg{args.outName, templateImg.axis, args.outPath};

  err = queue.enqueueReadBuffer(outImgBuf, CL_TRUE, 0,
                                sizeof(cl_double) * w * h, &outImg);
  checkError(err);
  err = queue.enqueueReadBuffer(outMaskBuf, CL_TRUE, 0, sizeof(cl_ushort) * w * h,
                                 &mask);
  checkError(err);

  // Add background and scale by kernel sum for output of convoluted image.
  for(int y = args.hKernelWidth; y < h - args.hKernelWidth; y++) {
    for(int x = args.hKernelWidth; x < w - args.hKernelWidth; x++) {
      outImg.data[x + y * w] +=
          getBackground(x, y, convolutionKernel.solution, templateImg.axis);
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
        mask.maskPix(x, y,ImageMask::BAD_OUTPUT | ImageMask::BAD_INPUT | ImageMask::LOW_PIXEL);
      }
    }
  }

  clock_t p12 = clock();
  if(args.verboseTime) {
    printf("Conv took %lds %ldms\n", (p12 - p11) / CLOCKS_PER_SEC,
           ((p12 - p11) * 1000 / CLOCKS_PER_SEC) % 1000);
  }

  /* ===== Sub ===== */

  clock_t p13 = clock();

  std::cout << "\nSubtracting images..." << std::endl;

  // Write necessary data for subtraction
  err = queue.enqueueWriteBuffer(convImgBuf, CL_TRUE, 0,
                                 sizeof(cl_double) * w * h, &outImg);
  err = queue.enqueueWriteBuffer(sImgBuf, CL_TRUE, 0, sizeof(cl_double) * w * h,
                                 &scienceImg);
  checkError(err);

  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl_long, cl_long, cl_long> sub{program,
                                                                       "sub"};
  cl::Event subEvent = sub(eargs, sImgBuf, convImgBuf, diffImgBuf, args.fKernelWidth, w, h);
  subEvent.wait();

  // Read data from subtraction
  Image diffImg{"sub.fits", templateImg.axis, args.outPath};
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

  clock_t p14 = clock();
  if(args.verboseTime) {
    printf("Sub took %lds %ldms\n", (p14 - p13) / CLOCKS_PER_SEC,
           ((p14 - p13) * 1000 / CLOCKS_PER_SEC) % 1000);
  }

  /* ===== Fin ===== */

  clock_t p15 = clock();
  std::cout << "\nWriting output..." << std::endl;

  std::cout << "Inv kernel sum: " << invKernSum << std::endl;

  err = writeImage(outImg);
  checkError(err);
  
  err = writeImage(diffImg);
  checkError(err);

  clock_t p16 = clock();
  if(args.verboseTime) {
    printf("Fin took %lds %ldms\n", (p16 - p15) / CLOCKS_PER_SEC,
           ((p16 - p15) * 1000 / CLOCKS_PER_SEC) % 1000);
  }

  std::cout << "\nBACH finished." << std::endl;

  if(args.verboseTime) {
    printf("BACH took %lds %ldms\n", (p16 - p1) / CLOCKS_PER_SEC,
           ((p16 - p1) * 1000 / CLOCKS_PER_SEC) % 1000);
  }

  return 0;
}
