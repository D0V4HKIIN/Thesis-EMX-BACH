#include <omp.h>
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
  double start = omp_get_wtime();

  CCfits::FITS::setVerboseMode(true);

  Arguments args{};
  try {
    std::cout << "Reading in arguments..." << std::endl;
    getArguments(argc, argv, args);
  } catch(const std::invalid_argument& err) {
    std::cout << err.what() << '\n';
    return 1;
  }

  // such a lie!!! it's just creating the struct. The images are read in init()
  std::cout << "\nReading in images..." << std::endl;
  Image templateImg{args.templateName};
  Image scienceImg{args.scienceName};
  templateImg.path = scienceImg.path = args.inputPath + "/";

  if(args.verbose)
    std::cout << "template image name: " << args.templateName
              << ", science image name: " << args.scienceName << std::endl;

  std::cout << "\nSetting up openCL..." << std::endl;
  cl::Platform platform = getDefaultPlatform(args);
  cl::Device device = getDefaultDevice(platform, args);
  cl::Context context(device);
  cl::Program program = loadBuildPrograms(
      context, device, std::filesystem::path(argv[0]).parent_path(), "bach.cl",
      "ini.cl", "sss.cl", "cmv.cl", "cd.cl", "ksc.cl", "conv.cl", "sub.cl");
  cl::CommandQueue queue(context, device);

  if(args.verbose) {
    printVerboseClInfo(device);
  }

  ClData clData{device,
                context,
                program,
                queue,
                cl::Buffer(),
                cl::Buffer(),
                cl::Buffer(),
                cl::Buffer(),
                cl::Buffer(),
                cl::Buffer(),
                cl::Buffer(),
                cl::Buffer(),
                cl::Buffer(),
                cl::Buffer(),
                cl::Buffer(),
                cl::Buffer(),
                0,
                0,
                0,
                0,
                0,
                cl::Buffer(),
                cl::Buffer(),
                cl::Buffer(),
                cl::Buffer(),
                cl::Buffer(),
                cl::Buffer(),
                cl::Buffer(),
                cl::Buffer(),
                cl::Buffer(),
                cl::Buffer(),
                cl::Buffer(),
                0,
                cl::Buffer(),
                cl::Buffer(),
                cl::Buffer(),
                cl::Buffer(),
                cl::Buffer(),
                cl::Buffer(),
                cl::Buffer(),
                cl::Buffer(),
                cl::Buffer(),
                cl::Buffer(),
                cl::Buffer(),
                0};

  init(templateImg, scienceImg, clData, args);

  auto p2 = std::chrono::steady_clock::now();
  if(args.verboseTime) {
    std::cout << "Ini took " << timeDiff(p2, p1) << " ms" << std::endl;
  }

  /* ===== SSS ===== */

  auto p3 = std::chrono::steady_clock::now();
  std::vector<Stamp> templateStamps{};
  std::vector<Stamp> sciStamps{};
  if(args.sssMode == "cl") {
    sssCl(templateImg.axis, templateStamps, sciStamps, args, clData);
  } else if(args.sssMode == "mp") {
    sssMp(templateStamps, templateImg, sciStamps, scienceImg, args);
  } else if(args.sssMode == "compare") {
    std::vector<Stamp> templateStampsCl{};
    std::vector<Stamp> sciStampsCl{};
    std::vector<Stamp> templateStampsMp{};
    std::vector<Stamp> sciStampsMp{};
    sssCl(templateImg.axis, templateStampsCl, sciStampsCl, args, clData);
    sssMp(templateStampsMp, templateImg, sciStampsMp, scienceImg, args);

    std::cout << "sizes " << templateStampsCl.size()
              << " == " << templateStampsMp.size() << std::endl;

    std::vector<std::pair<int, int>> clCoords(templateStampsCl.size());
    clData.queue.enqueueReadBuffer(
        clData.tmpl.stampCoords, CL_TRUE, 0,
        sizeof(std::pair<int, int>) * clCoords.size(), &clCoords[0]);

    for(size_t i = 0; i < clCoords.size(); i++) {
      if(clCoords[i] != templateStampsMp[i].coords) {
        std::cout << clCoords[i].first << "-" << clCoords[i].second << " and "
                  << templateStampsMp[i].coords.first << "-"
                  << templateStampsMp[i].coords.second
                  << " are not the same!!!!!!!!!!!\n";
      }
    }

    // std::cout << "\nmp:" << std::endl;

    // for(size_t i = 0; i < templateStampsMp.size(); i++) {
    //   std::cout << templateStampsMp[i].coords.first << ",";
    // }

    exit(0);
  } else {
    std::cout << "expected sssMode to be cl or mp" << std::endl;
    exit(1);
  }

  auto p4 = std::chrono::steady_clock::now();
  if(args.verboseTime) {
    std::cout << "SSS took " << timeDiff(p4, p3) << " ms" << std::endl;
  }

  std::cout << std::endl;

  /* ===== CMV ===== */

  auto p5 = std::chrono::steady_clock::now();

  Kernel convolutionKernel{args};
  cmv(templateImg.axis, templateStamps, sciStamps, clData, args);

  auto p6 = std::chrono::steady_clock::now();
  if(args.verboseTime) {
    std::cout << "CMV took " << timeDiff(p6, p5) << " ms" << std::endl;
  }

  /* ===== CD ===== */

  auto p7 = std::chrono::steady_clock::now();

  bool convTemplate =
      cd(templateImg, scienceImg, templateStamps, sciStamps, clData, args);

  auto p8 = std::chrono::steady_clock::now();
  if(args.verboseTime) {
    std::cout << "CD took " << timeDiff(p8, p7) << " ms" << std::endl;
  }

  /* ===== KSC ===== */

  auto p9 = std::chrono::steady_clock::now();

  ksc(templateStamps, convolutionKernel, scienceImg, clData.tImgBuf,
      clData.sImgBuf, clData, clData.tmpl, args);

  auto p10 = std::chrono::steady_clock::now();
  if(args.verboseTime) {
    std::cout << "KSC took " << timeDiff(p10, p9) << " ms" << std::endl;
  }

  /* ===== Conv ===== */

  auto p11 = std::chrono::steady_clock::now();

  Image convImg{args.outName, templateImg.axis, args.outPath};
  double kernSum = conv(templateImg.axis, convImg, convolutionKernel,
                        convTemplate, clData, args);

  auto p12 = std::chrono::steady_clock::now();
  if(args.verboseTime) {
    std::cout << "Conv took " << timeDiff(p12, p11) << " ms" << std::endl;
  }

  /* ===== Sub ===== */

  auto p13 = std::chrono::steady_clock::now();

  Image diffImg{"sub.fits", templateImg.axis, args.outPath};
  sub(templateImg.axis, diffImg, convTemplate, kernSum, clData, args);

  auto p14 = std::chrono::steady_clock::now();
  if(args.verboseTime) {
    std::cout << "Sub took " << timeDiff(p14, p13) << " ms" << std::endl;
  }

  /* ===== Fin ===== */

  auto p15 = std::chrono::steady_clock::now();

  fin(convImg, diffImg, args);

  auto p16 = std::chrono::steady_clock::now();
  if(args.verboseTime) {
    std::cout << "Fin took " << timeDiff(p16, p15) << " ms" << std::endl;
  }

  double end = omp_get_wtime();
  std::cout << "omp time " << end - start << std::endl;

  std::cout << "\nBACH finished." << std::endl;

  if(args.verboseTime) {
    std::cout << "BACH took " << timeDiff(p16, p1) << " ms" << std::endl;
  }

  return 0;
}
