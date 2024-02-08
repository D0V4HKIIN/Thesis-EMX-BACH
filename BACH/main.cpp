#define CL_HPP_TARGET_OPENCL_VERSION 300

#include <time.h>

#include <CL/opencl.hpp>
#include <iostream>
#include <iterator>
#include <vector>

#include "utils/argsUtil.h"
#include "utils/bachUtil.h"
#include "utils/clUtil.h"
#include "utils/fitsUtil.h"

int main(int argc, char* argv[]) {
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

  cl_int err{};

  err = readImage(templateImg);
  checkError(err);
  err = readImage(scienceImg);
  checkError(err);

  maskInput(templateImg, scienceImg);

  auto [w, h] = templateImg.axis;
  if(w != scienceImg.axis.first || h != scienceImg.axis.second) {
    std::cout << "Template image and science image must be the same size!"
              << std::endl;

    exit(1);
  }

  std::cout << "\nSetting up openCL..." << std::endl;

  cl::Device default_device{get_default_device()};
  cl::Context context{default_device};

  cl::Program program =
      load_build_programs(context, default_device, "conv.cl", "sub.cl");

  clock_t p2 = clock();
  if(args.verboseTime) {
    printf("Initiation took %lds %ldms\n", (p2 - p1) / CLOCKS_PER_SEC,
           ((p2 - p1) * 1000 / CLOCKS_PER_SEC) % 1000);
  }

  /* ===== SSS ===== */

  clock_t p3 = clock();

  std::cout << "\nCreating stamps..." << std::endl;

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

  std::vector<Stamp> templateStamps{};
  createStamps(templateImg, templateStamps, w, h);
  if(args.verbose) {
    std::cout << "Stamps created for " << templateImg.name << std::endl;
  }

  std::vector<Stamp> sciStamps{};
  createStamps(scienceImg, sciStamps, w, h);
  if(args.verbose) {
    std::cout << "Stamps created for " << scienceImg.name << std::endl;
  }

  /* == Check Template Stamps  ==*/
  double filledTempl{};
  double filledScience{};
  identifySStamps(templateStamps, templateImg, sciStamps, scienceImg, &filledTempl, &filledScience);
  if(filledTempl < 0.1 || filledScience < 0.1) {
    std::cout << "ERROR" << std::endl;
    std::exit(1);

    /*if(args.verbose)
      std::cout << "Not enough substamps found in " << templateImg.name
                << " trying again with lower thresholds..." << std::endl;
    args.threshLow *= 0.5;
    numTemplSStamps = identifySStamps(templateStamps, templateImg);
    args.threshLow /= 0.5;*/
  }

  if(templateStamps.size() == 0 && sciStamps.size() == 0) {
    std::cout << "No substamps found" << std::endl;
    exit(1);
  }

  clock_t p4 = clock();
  if(args.verboseTime) {
    printf("SSS took %lds %ldms\n", (p4 - p3) / CLOCKS_PER_SEC,
           ((p4 - p3) * 1000 / CLOCKS_PER_SEC) % 1000);
  }

  std::cout << std::endl;

  /* ===== CMV ===== */

  clock_t p5 = clock();

  std::cout << "\nCalculating matrix variables..." << std::endl;

  Kernel convolutionKernel{};
  for(auto& s : templateStamps) {
    fillStamp(s, templateImg, scienceImg, convolutionKernel);
  }
  for(auto& s : sciStamps) {
    fillStamp(s, scienceImg, templateImg, convolutionKernel);
  }

  clock_t p6 = clock();
  if(args.verboseTime) {
    printf("CMV took %lds %ldms\n", (p6 - p5) / CLOCKS_PER_SEC,
           ((p6 - p5) * 1000 / CLOCKS_PER_SEC) % 1000);
  }

  /* ===== CD ===== */

  clock_t p7 = clock();

  std::cout << "\nChoosing convolution direction..." << std::endl;

  double templateMerit = testFit(templateStamps, templateImg, scienceImg);
  double scienceMerit = testFit(sciStamps, scienceImg, templateImg);
  if(args.verbose)
    std::cout << "template merit value = " << templateMerit
              << ", science merit value = " << scienceMerit << std::endl;
  if(scienceMerit <= templateMerit) {
    std::swap(scienceImg, templateImg);
    std::swap(sciStamps, templateStamps);
  }
  if(args.verbose)
    std::cout << templateImg.name << " chosen to be convolved." << std::endl;

  clock_t p8 = clock();
  if(args.verboseTime) {
    printf("CD took %lds %ldms\n", (p8 - p7) / CLOCKS_PER_SEC,
           ((p8 - p7) * 1000 / CLOCKS_PER_SEC) % 1000);
  }

  /* ===== KSC ===== */

  clock_t p9 = clock();

  std::cout << "\nFitting kernel..." << std::endl;

  fitKernel(convolutionKernel, templateStamps, templateImg, scienceImg);

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

  // Declare all the buffers which will be need in opencl operations.
  cl::Buffer tImgBuf(context, CL_MEM_READ_ONLY, sizeof(cl_double) * w * h);
  cl::Buffer sImgBuf(context, CL_MEM_READ_ONLY, sizeof(cl_double) * w * h);
  cl::Buffer convImgBuf(context, CL_MEM_READ_ONLY, sizeof(cl_double) * w * h);
  cl::Buffer kernBuf(context, CL_MEM_READ_ONLY,
                     sizeof(cl_double) * convKernels.size());
  cl::Buffer outImgBuf(context, CL_MEM_WRITE_ONLY, sizeof(cl_double) * w * h);
  cl::Buffer diffImgBuf(context, CL_MEM_WRITE_ONLY, sizeof(cl_double) * w * h);

  cl::CommandQueue queue(context, default_device);

  // Write necessary data for convolution
  err = queue.enqueueWriteBuffer(kernBuf, CL_TRUE, 0,
                                 sizeof(cl_double) * convKernels.size(),
                                 &convKernels[0]);
  checkError(err);
  err = queue.enqueueWriteBuffer(tImgBuf, CL_TRUE, 0, sizeof(cl_double) * w * h,
                                 &templateImg);
  checkError(err);

  cl::KernelFunctor<cl::Buffer, cl_long, cl_long, cl::Buffer, cl::Buffer, cl_long,
                    cl_long>
      conv{program, "conv"};
  cl::EnqueueArgs eargs{queue, cl::NullRange, cl::NDRange(w * h),
                        cl::NullRange};
  conv(eargs, kernBuf, args.fKernelWidth, xSteps, tImgBuf, outImgBuf, w, h);

  // Read data from convolution
  Image outImg{args.outName, templateImg.axis, args.outPath};

  err = queue.enqueueReadBuffer(outImgBuf, CL_TRUE, 0,
                                sizeof(cl_double) * w * h, &outImg);
  checkError(err);

  if (queue.finish() != CL_SUCCESS) {
    std::cout << "Could not finish executing the queue" << std::endl;
    std::exit(1);
  }

  // Add background and scale by kernel sum for output of convoluted image.
  for(int y = args.hKernelWidth; y < h - args.hKernelWidth; y++) {
    for(int x = args.hKernelWidth; x < w - args.hKernelWidth; x++) {
      outImg.data[x + y * w] +=
          getBackground(x, y, convolutionKernel.solution, templateImg.axis);
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
  sub(eargs, sImgBuf, convImgBuf, diffImgBuf, args.fKernelWidth, w, h);

  // Read data from subtraction
  Image diffImg{"sub.fits", templateImg.axis, args.outPath};
  err = queue.enqueueReadBuffer(diffImgBuf, CL_TRUE, 0,
                                sizeof(cl_double) * w * h, &diffImg);
  checkError(err);

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
