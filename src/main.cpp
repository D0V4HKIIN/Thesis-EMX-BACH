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

int main(int argc, const char *argv[]) {
  /* ===== INI ===== */
  auto p1 = std::chrono::steady_clock::now();
  double start = omp_get_wtime();

  CCfits::FITS::setVerboseMode(true);

  Arguments args{};
  try {
    std::cout << "Reading in arguments..." << std::endl;
    getArguments(argc, argv, args);
  } catch(const std::invalid_argument &err) {
    std::cout << err.what() << '\n';
    return 1;
  }

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

  // Read input images
  std::cout << "\nReading in images..." << std::endl;
  readImage(templateImg, args);
  readImage(scienceImg, args);

  int pixelCount = templateImg.axis.first * templateImg.axis.second;

  size_t subStampMaxCount{2 * args.maxKSStamps};

  ClData clData{
      device,
      context,
      program,
      queue,
      cl::Buffer(clData.context, CL_MEM_READ_WRITE,
                 sizeof(cl_double) * pixelCount),
      cl::Buffer(clData.context, CL_MEM_READ_WRITE,
                 sizeof(cl_double) * pixelCount),
      cl::Buffer(clData.context, CL_MEM_READ_WRITE,
                 sizeof(cl_ushort) * pixelCount),
      cl::Buffer(clData.context, CL_MEM_WRITE_ONLY,
                 sizeof(cl_double) * pixelCount),
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
      cl::Buffer(clData.context, CL_MEM_READ_WRITE,
                 sizeof(cl_int2) * args.stampsx * args.stampsy),
      cl::Buffer(clData.context, CL_MEM_READ_WRITE,
                 sizeof(cl_int2) * args.stampsx * args.stampsy),
      cl::Buffer(clData.context, CL_MEM_READ_WRITE,
                 sizeof(cl_double) * args.stampsx * args.stampsy),
      cl::Buffer(clData.context, CL_MEM_READ_WRITE,
                 sizeof(cl_double) * args.stampsx * args.stampsy),
      cl::Buffer(clData.context, CL_MEM_READ_WRITE,
                 sizeof(cl_double) * args.stampsx * args.stampsy),
      cl::Buffer(
          clData.context, CL_MEM_READ_WRITE,
          sizeof(cl_int2) * subStampMaxCount * args.stampsx * args.stampsy),
      cl::Buffer(
          clData.context, CL_MEM_READ_WRITE,
          sizeof(cl_double) * subStampMaxCount * args.stampsx * args.stampsy),
      cl::Buffer(
          clData.context, CL_MEM_READ_WRITE,
          sizeof(cl_int) * args.stampsx * args.stampsy * subStampMaxCount),
      cl::Buffer(),
      cl::Buffer(),
      cl::Buffer(),
      0,
      cl::Buffer(clData.context, CL_MEM_READ_WRITE,
                 sizeof(cl_int2) * args.stampsx * args.stampsy),
      cl::Buffer(clData.context, CL_MEM_READ_WRITE,
                 sizeof(cl_int2) * args.stampsx * args.stampsy),
      cl::Buffer(clData.context, CL_MEM_READ_WRITE,
                 sizeof(cl_double) * args.stampsx * args.stampsy),
      cl::Buffer(clData.context, CL_MEM_READ_WRITE,
                 sizeof(cl_double) * args.stampsx * args.stampsy),
      cl::Buffer(clData.context, CL_MEM_READ_WRITE,
                 sizeof(cl_double) * args.stampsx * args.stampsy),
      cl::Buffer(
          clData.context, CL_MEM_READ_WRITE,
          sizeof(cl_int2) * subStampMaxCount * args.stampsx * args.stampsy),
      cl::Buffer(
          clData.context, CL_MEM_READ_WRITE,
          sizeof(cl_double) * subStampMaxCount * args.stampsx * args.stampsy),
      cl::Buffer(
          clData.context, CL_MEM_READ_WRITE,
          sizeof(cl_int) * args.stampsx * args.stampsy * subStampMaxCount),
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

  const auto [w, h] = templateImg.axis;

  args.fStampWidth = std::min(int(w / args.stampsx), int(h / args.stampsy));
  args.fStampWidth -= args.fKernelWidth;
  args.fStampWidth -= args.fStampWidth % 2 == 0 ? 1 : 0;

  if(args.fStampWidth < args.fSStampWidth) {
    args.fStampWidth = args.fSStampWidth + args.fKernelWidth;
    args.fStampWidth -= args.fStampWidth % 2 == 0 ? 1 : 0;

    args.stampsx = int(w / args.fStampWidth);
    args.stampsy = int(h / args.fStampWidth);

    if(args.verbose)
      std::cout << "Too many stamps requested, using " << args.stampsx << "x"
                << args.stampsy << " stamps instead." << std::endl;
  }

  if(args.sssMode == "cl") {
    sssCl(templateImg.axis, templateStamps, sciStamps, args, clData);

  } else if(args.sssMode == "mp") {
    // read mask that ini created
    ImageMask mask{templateImg.axis};

    clData.queue.enqueueReadBuffer(
        clData.maskBuf, CL_TRUE, 0,
        sizeof(cl_ushort) * templateImg.axis.first * templateImg.axis.second,
        &mask.dataMask[0]);

    sssMp(templateStamps, templateImg, sciStamps, scienceImg, mask, args);

    moveSssToGpu(templateStamps, sciStamps, mask, clData, args);

  } else if(args.sssMode == "compare") {
    // used for debug
    std::vector<Stamp> templateStampsCl{};
    std::vector<Stamp> sciStampsCl{};

    std::vector<Stamp> templateStampsMp{};
    std::vector<Stamp> sciStampsMp{};

    // read mask before calling sssCl bc sssCl modifies the mask
    ImageMask mask{templateImg.axis};

    clData.queue.enqueueReadBuffer(
        clData.maskBuf, CL_TRUE, 0,
        sizeof(cl_ushort) * templateImg.axis.first * templateImg.axis.second,
        &mask.dataMask[0]);

    auto cl = std::chrono::steady_clock::now();
    sssCl(templateImg.axis, templateStampsCl, sciStampsCl, args, clData);

    auto mp = std::chrono::steady_clock::now();
    double start = omp_get_wtime();
    // omp_set_num_threads(1);
    sssMp(templateStampsMp, templateImg, sciStampsMp, scienceImg, mask, args);

    auto end_chrono = std::chrono::steady_clock::now();
    double end = omp_get_wtime();
    std::cout << end - start << " seconds for sssMp" << std::endl;
    std::cout << timeDiff(mp, cl) << "ms for sssCl" << std::endl;
    std::cout << timeDiff(end_chrono, mp) << "ms for sssMp" << std::endl;

    std::cout << "sizes " << templateStampsCl.size()
              << " == " << templateStampsMp.size() << std::endl;

    std::vector<std::pair<int, int>> clCoords(templateStampsCl.size());
    clData.queue.enqueueReadBuffer(
        clData.tmpl.stampCoords, CL_TRUE, 0,
        sizeof(std::pair<int, int>) * clCoords.size(), &clCoords[0]);

    std::vector<double> clSkyEst(templateStampsCl.size());
    clData.queue.enqueueReadBuffer(clData.tmpl.stats.skyEsts, CL_TRUE, 0,
                                   sizeof(cl_double) * clSkyEst.size(),
                                   &clSkyEst[0]);

    std::vector<double> clFwhm(templateStampsCl.size());
    clData.queue.enqueueReadBuffer(clData.tmpl.stats.fwhms, CL_TRUE, 0,
                                   sizeof(cl_double) * clFwhm.size(),
                                   &clFwhm[0]);

    std::vector<uint16_t> clMask(mask.dataMask.size());
    clData.queue.enqueueReadBuffer(
        clData.maskBuf, CL_TRUE, 0,
        sizeof(u_int16_t) * templateImg.axis.first * templateImg.axis.second,
        &clMask[0]);

    std::cout << "stampcoords" << std::endl;
    for(size_t i = 0; i < clCoords.size(); i++) {
      if(clCoords[i] != templateStampsMp[i].coords) {
        std::cout << clCoords[i].first << "-" << clCoords[i].second << " and "
                  << templateStampsMp[i].coords.first << "-"
                  << templateStampsMp[i].coords.second
                  << " are not the same!!!!!!!!!!!\n";
      }
    }

    clData.tmpl.stampCount = args.stampsx * args.stampsy;

    std::cout << "skyest" << std::endl;
    for(size_t i = 0; i < templateStampsMp.size(); i++) {
      if(clSkyEst[i] != templateStampsMp[i].stats.skyEst) {
        std::cout << clSkyEst[i] << "==" << templateStampsMp[i].stats.skyEst
                  << " are not the same\n";
      }
    }

    std::cout << "fwhm" << std::endl;
    for(size_t i = 0; i < clFwhm.size(); i++) {
      if(clFwhm[i] != templateStampsMp[i].stats.fwhm) {
        std::cout << clFwhm[i] << "==" << templateStampsMp[i].stats.fwhm
                  << " are not the same\n";
      }
    }

    std::cout << "mask" << std::endl;
    for(size_t i = 0; i < clMask.size(); i++) {
      if(clMask[i] != mask.dataMask[i]) {
        std::cout << "mask differs" << clMask[i] << " == " << mask.dataMask[i]
                  << std::endl;
      }
    }

    std::cout << "substamps " << std::endl;
    for(size_t i = 0; i < templateStampsCl.size(); i++) {
      if(templateStampsCl[i].subStamps.size() !=
         templateStampsMp[i].subStamps.size()) {
        std::cout << "substamp sizes are not the same for index " << i
                  << std::endl;
      } else {
        for(size_t j = 0; j < templateStampsCl[i].subStamps.size(); j++) {
          if(templateStampsCl[i].subStamps[j].val !=
             templateStampsMp[i].subStamps[j].val) {
            std::cout << "substamp value not matching"
                      << templateStampsCl[i].subStamps[j].val
                      << " == " << templateStampsMp[i].subStamps[j].val
                      << std::endl;
          }
          if(templateStampsCl[i].subStamps[j].imageCoords !=
             templateStampsMp[i].subStamps[j].imageCoords) {
            std::cout << "substamp coords not matching"
                      << templateStampsCl[i].subStamps[j].imageCoords.first
                      << ","
                      << templateStampsCl[i].subStamps[j].imageCoords.second
                      << " == "
                      << templateStampsMp[i].subStamps[j].imageCoords.first
                      << ","
                      << templateStampsMp[i].subStamps[j].imageCoords.second
                      << std::endl;
          }
        }
      }
    }

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
  double kernSum = conv(templateImg.axis, templateImg, convImg,
                        convolutionKernel, convTemplate, clData, args);

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
