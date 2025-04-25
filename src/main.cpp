#include <time.h>

#include <CL/opencl.hpp>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <vector>

#include "bach.h"
#include "bachUtil.h"
#include "clUtil.h"
#include "datatypeUtil.h"
#include "fitsUtil.h"

int main(int argc, const char* argv[]) {
  auto p1 = std::chrono::steady_clock::now();

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
  cl::Device device{getDefaultDevice()};
  cl::Context context{device};
  cl::Program program = loadBuildPrograms(
      context, device, std::filesystem::path(argv[0]).parent_path(), "conv.cl",
      "sub.cl");
  cl::CommandQueue queue(context, device);

  ClData clData{device, context, program, queue};

  init(templateImg, scienceImg, mask, clData, args);
  const auto [w, h] = templateImg.axis;

  auto p2 = std::chrono::steady_clock::now();
  if(args.verboseTime) {
    std::cout << "Ini took " << timeDiff(p2, p1) << " ns" << std::endl;
  }

  /* ===== SSS ===== */

  auto p3 = std::chrono::steady_clock::now();
  std::vector<Stamp> templateStamps{};
  std::vector<Stamp> sciStamps{};
  sss(templateImg, scienceImg, mask, templateStamps, sciStamps, args);

  auto p4 = std::chrono::steady_clock::now();
  if(args.verboseTime) {
    std::cout << "SSS took " << timeDiff(p4, p3) << " ns" << std::endl;
  }

  std::cout << std::endl;

  /* ===== CMV ===== */

  auto p5 = std::chrono::steady_clock::now();

  Kernel convolutionKernel{args};
  cmv(templateImg, scienceImg, mask, templateStamps, sciStamps,
      convolutionKernel, args);

  auto p6 = std::chrono::steady_clock::now();
  if(args.verboseTime) {
    std::cout << "CMV took " << timeDiff(p6, p5) << " ns" << std::endl;
  }

  /* ===== CD ===== */

  auto p7 = std::chrono::steady_clock::now();

  bool convTemplate =
      cd(templateImg, scienceImg, mask, templateStamps, sciStamps, args);

  auto p8 = std::chrono::steady_clock::now();
  if(args.verboseTime) {
    std::cout << "CD took " << timeDiff(p8, p7) << " ns" << std::endl;
  }

  /* ===== KSC ===== */

  auto p9 = std::chrono::steady_clock::now();

  ksc(templateImg, scienceImg, mask, templateStamps, convolutionKernel, args);

  auto p10 = std::chrono::steady_clock::now();
  if(args.verboseTime) {
    std::cout << "KSC took " << timeDiff(p10, p9) << " ns" << std::endl;
  }

  /* ===== Conv ===== */

  auto p11 = std::chrono::steady_clock::now();

  Image convImg{args.outName, templateImg.axis, args.outPath};
  double kernSum =
      conv(templateImg, scienceImg, mask, convImg, convolutionKernel,
           convTemplate, context, program, queue, args);

  auto p12 = std::chrono::steady_clock::now();
  if(args.verboseTime) {
    std::cout << "Conv took " << timeDiff(p12, p11) << " ns" << std::endl;
  }

  /* ===== Sub ===== */

  auto p13 = std::chrono::steady_clock::now();

  Image diffImg{"sub.fits", templateImg.axis, args.outPath};
  sub(convImg, scienceImg, mask, diffImg, convTemplate, kernSum, context,
      program, queue, args);

  auto p14 = std::chrono::steady_clock::now();
  if(args.verboseTime) {
    std::cout << "Sub took " << timeDiff(p14, p13) << " ns" << std::endl;
  }

  /* ===== Fin ===== */

  auto p15 = std::chrono::steady_clock::now();

  fin(convImg, diffImg, args);

  auto p16 = std::chrono::steady_clock::now();
  if(args.verboseTime) {
    std::cout << "Fin took " << timeDiff(p16, p15) << " ns" << std::endl;
  }

  std::cout << "\nBACH finished." << std::endl;

  if(args.verboseTime) {
    std::cout << "BACH took " << timeDiff(p16, p1) << " ns" << std::endl;
  }

  return 0;
}
