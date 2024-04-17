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
  cl::Device device{getDefaultDevice()};
  cl::Context context{device};
  cl::Program program =
      loadBuildPrograms(context, device, std::filesystem::path(argv[0]).parent_path(),
      "conv.cl", "sub.cl");
  cl::CommandQueue queue(context, device);

  ClData clData { device, context, program, queue };

  init(templateImg, scienceImg, mask, clData, args);
  const auto [w, h] = templateImg.axis;

  clock_t p2 = clock();
  if(args.verboseTime) {
    printf("Initiation took %lds %ldms\n", (p2 - p1) / CLOCKS_PER_SEC,
           ((p2 - p1) * 1000 / CLOCKS_PER_SEC) % 1000);
  }

  /* ===== SSS ===== */

  clock_t p3 = clock();
  std::vector<Stamp> templateStamps{};
  std::vector<Stamp> sciStamps{};
  sss(templateImg, scienceImg, mask, templateStamps, sciStamps, args);

  clock_t p4 = clock();
  if(args.verboseTime) {
    printf("SSS took %lds %ldms\n", (p4 - p3) / CLOCKS_PER_SEC,
           ((p4 - p3) * 1000 / CLOCKS_PER_SEC) % 1000);
  }

  std::cout << std::endl;

  /* ===== CMV ===== */

  clock_t p5 = clock();

  Kernel convolutionKernel{args};
  cmv(templateImg, scienceImg, mask, templateStamps, sciStamps, convolutionKernel, args);
  
  clock_t p6 = clock();
  if(args.verboseTime) {
    printf("CMV took %lds %ldms\n", (p6 - p5) / CLOCKS_PER_SEC,
           ((p6 - p5) * 1000 / CLOCKS_PER_SEC) % 1000);
  }

  /* ===== CD ===== */

  clock_t p7 = clock();

  bool convTemplate = cd(templateImg, scienceImg, mask, templateStamps, sciStamps, args);

  clock_t p8 = clock();
  if(args.verboseTime) {
    printf("CD took %lds %ldms\n", (p8 - p7) / CLOCKS_PER_SEC,
           ((p8 - p7) * 1000 / CLOCKS_PER_SEC) % 1000);
  }

  /* ===== KSC ===== */

  clock_t p9 = clock();

  ksc(templateImg, scienceImg, mask, templateStamps, convolutionKernel, args);

  clock_t p10 = clock();
  if(args.verboseTime) {
    printf("KSC took %lds %ldms\n", (p10 - p9) / CLOCKS_PER_SEC,
           ((p10 - p9) * 1000 / CLOCKS_PER_SEC) % 1000);
  }

  /* ===== Conv ===== */

  clock_t p11 = clock();

  Image convImg{args.outName, templateImg.axis, args.outPath};
  double kernSum = conv(templateImg, scienceImg, mask, convImg, convolutionKernel, convTemplate, context, program, queue, args);

  clock_t p12 = clock();
  if(args.verboseTime) {
    printf("Conv took %lds %ldms\n", (p12 - p11) / CLOCKS_PER_SEC,
           ((p12 - p11) * 1000 / CLOCKS_PER_SEC) % 1000);
  }

  /* ===== Sub ===== */

  clock_t p13 = clock();

  Image diffImg{"sub.fits", templateImg.axis, args.outPath};
  sub(convImg, scienceImg, mask, diffImg, convTemplate, kernSum, context, program, queue, args);

  clock_t p14 = clock();
  if(args.verboseTime) {
    printf("Sub took %lds %ldms\n", (p14 - p13) / CLOCKS_PER_SEC,
           ((p14 - p13) * 1000 / CLOCKS_PER_SEC) % 1000);
  }

  /* ===== Fin ===== */

  clock_t p15 = clock();

  fin(convImg, diffImg, args);

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
