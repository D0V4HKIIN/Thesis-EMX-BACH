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
  
  Arguments args{};
  try {
    std::cout << "Reading in arguments..." << std::endl;
    getArguments(argc, argv, args);
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
  cl::Platform platform = getDefaultPlatform();
  cl::Device device = getDefaultDevice(platform);
  cl::Context context(device);
  cl::Program program =
      loadBuildPrograms(context, device, std::filesystem::path(argv[0]).parent_path(),
      "bash.cl", "cd.cl", "cmv.cl", "conv.cl", "ini.cl", "ksc.cl", "sub.cl");
  cl::CommandQueue queue(context, device);

  if (args.verbose) {
    printVerboseClInfo(platform, device);
  }

  ClData clData { device, context, program, queue };

  init(templateImg, scienceImg, mask, clData, args);
  const auto [w, h] = templateImg.axis;

  clock_t p2 = clock();
  if(args.verboseTime) {
    std::cout << "Ini took " << (p2 - p1) * 1000 / CLOCKS_PER_SEC << " ms" << std::endl;
  }

  /* ===== SSS ===== */

  clock_t p3 = clock();
  std::vector<Stamp> templateStamps{};
  std::vector<Stamp> sciStamps{};
  sss(templateImg, scienceImg, mask, templateStamps, sciStamps, args);

  clock_t p4 = clock();
  if(args.verboseTime) {
    std::cout << "SSS took " << (p4 - p3) * 1000 / CLOCKS_PER_SEC << " ms" << std::endl;
  }

  std::cout << std::endl;

  /* ===== CMV ===== */

  clock_t p5 = clock();

  Kernel convolutionKernel{args};
  cmv(templateImg, scienceImg, mask, templateStamps, sciStamps, convolutionKernel, clData, args);
  
  clock_t p6 = clock();
  if(args.verboseTime) {
    std::cout << "CMV took " << (p6 - p5) * 1000 / CLOCKS_PER_SEC << " ms" << std::endl;
  }

  /* ===== CD ===== */

  clock_t p7 = clock();

  bool convTemplate = cd(templateImg, scienceImg, mask, templateStamps, sciStamps, clData, args);

  clock_t p8 = clock();
  if(args.verboseTime) {
    std::cout << "CD took " << (p8 - p7) * 1000 / CLOCKS_PER_SEC << " ms" << std::endl;
  }

  /* ===== KSC ===== */

  clock_t p9 = clock();

  ksc(templateImg, scienceImg, mask, templateStamps, convolutionKernel, clData.tImgBuf, clData.sImgBuf, clData, clData.tmpl, args);

  clock_t p10 = clock();
  if(args.verboseTime) {
    std::cout << "KSC took " << (p10 - p9) * 1000 / CLOCKS_PER_SEC << " ms" << std::endl;
  }

  /* ===== Conv ===== */

  clock_t p11 = clock();

  Image convImg{args.outName, templateImg.axis, args.outPath};
  double kernSum = conv(templateImg, scienceImg, mask, convImg, convolutionKernel, convTemplate, context, program, queue, args);

  clock_t p12 = clock();
  if(args.verboseTime) {
    std::cout << "Conv took " << (p12 - p11) * 1000 / CLOCKS_PER_SEC << " ms" << std::endl;
  }

  /* ===== Sub ===== */

  clock_t p13 = clock();

  Image diffImg{"sub.fits", templateImg.axis, args.outPath};
  sub(convImg, scienceImg, mask, diffImg, convTemplate, kernSum, context, program, queue, args);

  clock_t p14 = clock();
  if(args.verboseTime) {
    std::cout << "Sub took " << (p14 - p13) * 1000 / CLOCKS_PER_SEC << " ms" << std::endl;
  }

  /* ===== Fin ===== */

  clock_t p15 = clock();

  fin(convImg, diffImg, args);

  clock_t p16 = clock();
  if(args.verboseTime) {
    std::cout << "Fin took " << (p16 - p15) * 1000 / CLOCKS_PER_SEC << " ms" << std::endl;
  }

  std::cout << "\nBACH finished." << std::endl;

  if(args.verboseTime) {
    std::cout << "BACH took " << (p16 - p1) * 1000 / CLOCKS_PER_SEC << " ms" << std::endl;
  }

  return 0;
}
