#include <omp.h>

#include <cassert>
#include <iostream>

#include "bachUtil.h"
#include "mathUtil.h"

void identifySStamps(const std::pair<cl_int, cl_int>& axis,
                     const Arguments& args, const ClData& clData) {
  std::cout << "Identifying sub-stamps..." << std::endl;

  if(args.verbose) std::cout << "calcStats (template)" << std::endl;
  calcStats(axis, args, clData.tImgBuf, clData.tmpl, clData);
  if(args.verbose) std::cout << "calcStats (science)" << std::endl;
  calcStats(axis, args, clData.sImgBuf, clData.sci, clData);

  if(args.verbose) std::cout << "findSStamps (template)" << std::endl;
  findSStamps(axis, true, args, clData.tImgBuf, clData.tmpl, clData);
  if(args.verbose) std::cout << "findSStamps (science)" << std::endl;
  findSStamps(axis, false, args, clData.sImgBuf, clData.sci, clData);
}

void identifySStampsMp(Stamp& templStamp, const Image& templImage,
                       Stamp& scienceStamp, const Image& scienceImage,
                       ImageMask& mask, const Arguments& args) {
  calcStatsMp(templStamp, templImage, mask, args);
  calcStatsMp(scienceStamp, scienceImage, mask, args);

  findSStampsMp(templStamp, templImage, mask, true, args);
  findSStampsMp(scienceStamp, scienceImage, mask, false, args);
}

void createStamps(const int w, const int h, ClStampsData& stampsData,
                  const ClData& clData, const Arguments& args) {
  cl::EnqueueArgs eargsBounds{clData.queue,
                              cl::NDRange(args.stampsx * args.stampsy)};
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl_int, cl_int, cl_int, cl_int,
                    cl_int>
      boundsFunc(clData.program, "createStampBounds");

  cl::Event boundsEvent{boundsFunc(eargsBounds, stampsData.stampCoords,
                                   stampsData.stampSizes, args.stampsx,
                                   args.stampsy, args.fStampWidth, w, h)};
  stampsData.stampCount = args.stampsx * args.stampsy;
  boundsEvent.wait();
}

void createStampsMp(const int stampX, const int stampY, const int w,
                    const int h, Stamp& stamp, const Arguments& args) {
  int startX = stampX * w / args.stampsx;
  int startY = stampY * h / args.stampsy;

  int stopX = std::min(startX + args.fStampWidth, w);
  int stopY = std::min(startY + args.fStampWidth, h);

  int stampW = stopX - startX;
  int stampH = stopY - startY;

  stamp.coords = std::make_pair(startX, startY);
  stamp.size = std::make_pair(stampW, stampH);
}

// why is this not a void function? (This used to be a function that would
// return 1 on error)
cl_int findSStamps(const std::pair<cl_int, cl_int>& axis, const bool isTemplate,
                   const Arguments& args, const cl::Buffer& imgBuf,
                   const ClStampsData& stampsData, const ClData& clData) {
  auto [imgW, imgH] = axis;

  cl::size_type nStamps{static_cast<cl::size_type>(args.stampsx) *
                        static_cast<cl::size_type>(args.stampsy)};

  ImageMasks badMask = ImageMasks::ALL & ~ImageMasks::OK_CONV;
  ImageMasks badPixelMask, skipMask;

  if(isTemplate) {
    badMask &= ~(ImageMasks::BAD_PIXEL_S | ImageMasks::SKIP_S);
    badPixelMask = ImageMasks::BAD_PIXEL_T;
    skipMask = ImageMasks::SKIP_T;
  } else {
    badMask &= ~(ImageMasks::BAD_PIXEL_T | ImageMasks::SKIP_T);
    badPixelMask = ImageMasks::BAD_PIXEL_S;
    skipMask = ImageMasks::SKIP_S;
  }

  cl_uint maxSStamps{2 * args.maxKSStamps};

  constexpr int localSize{1};

  cl::EnqueueArgs eargsFindSStamps(
      clData.queue, cl::NDRange(roundUpToMultiple(nStamps, localSize)),
      cl::NDRange(localSize));
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                    cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl_double,
                    cl_double, cl_int, cl_int, cl_int, cl_int, cl_int,
                    cl_ushort, cl_ushort, cl_ushort, cl::LocalSpaceArg,
                    cl::LocalSpaceArg>
      findSStampsFunc{clData.program, "findSubStamps"};

  cl::Event findSStampsEvent{findSStampsFunc(
      eargsFindSStamps, imgBuf, clData.maskBuf, stampsData.stampCoords,
      stampsData.stampSizes, stampsData.stats.skyEsts, stampsData.stats.fwhms,
      stampsData.subStampCoords, stampsData.subStampValues,
      stampsData.subStampCounts, args.threshHigh, args.threshKernFit, imgW,
      args.fStampWidth, args.hSStampWidth, maxSStamps,
      static_cast<cl_int>(nStamps), static_cast<cl_ushort>(badMask),
      static_cast<cl_ushort>(badPixelMask), static_cast<cl_ushort>(skipMask),
      cl::Local(sizeof(cl_int2) * maxSStamps * localSize),
      cl::Local(sizeof(cl_double) * maxSStamps * localSize))};

  findSStampsEvent.wait();

  if(args.verbose) {
    std::vector<cl_int> sstampCounts(nStamps);
    clData.queue.enqueueReadBuffer(stampsData.subStampCounts, CL_TRUE, 0,
                                   sizeof(cl_int) * sstampCounts.size(),
                                   &sstampCounts[0]);

    for(size_t i{0}; i < nStamps; i++) {
      if(sstampCounts[i] == 0) {
        std::cout << "No suitable substamps found in stamp " << i << std::endl;
      } else {
        std::cout << "Added " << sstampCounts[i] << " substamps to stamp " << i
                  << std::endl;
      }
    }
  }
  return 0;
}

int findSStampsMp(Stamp& stamp, const Image& image, ImageMask& mask,
                  const bool isTemplate, const Arguments& args) {
  double floor = stamp.stats.skyEst + args.threshKernFit * stamp.stats.fwhm;

  double dfrac = 0.9;
  int maxSStamps = 2 * args.maxKSStamps;

  ImageMasks badMask = ImageMasks::ALL & ~ImageMasks::OK_CONV;
  ImageMasks badPixelMask, skipMask;

  if(isTemplate) {
    badMask &= ~(ImageMasks::BAD_PIXEL_S | ImageMasks::SKIP_S);
    badPixelMask = ImageMasks::BAD_PIXEL_T;
    skipMask = ImageMasks::SKIP_T;
  } else {
    badMask &= ~(ImageMasks::BAD_PIXEL_T | ImageMasks::SKIP_T);
    badPixelMask = ImageMasks::BAD_PIXEL_S;
    skipMask = ImageMasks::SKIP_S;
  }

  while(stamp.subStamps.size() < size_t(maxSStamps)) {
    double lowestPSFLim =
        std::max(floor, stamp.stats.skyEst +
                            (args.threshHigh - stamp.stats.skyEst) * dfrac);
    for(long y = 0; y < args.fStampWidth; y++) {
      long absy = y + stamp.coords.second;
      for(long x = 0; x < args.fStampWidth; x++) {
        long absx = x + stamp.coords.first;
        // long coords = x + (y * stamp.size.first);
        long absCoords = absx + (absy * image.axis.first);

        if(mask.isMasked(absCoords, badMask)) {
          continue;
        }

        if(image[absCoords] > args.threshHigh) {
          mask.maskPix(absx, absy, badPixelMask);
          continue;
        }

        if((image[absCoords] - stamp.stats.skyEst) * (1.0 / stamp.stats.fwhm) <
           args.threshKernFit) {
          continue;
        }

        if(image[absCoords] > lowestPSFLim) {  // good candidate found
          SubStamp s{std::make_pair(absx, absy), image[absCoords]};

          for(long ky = absy - args.hSStampWidth;
              ky <= absy + args.hSStampWidth; ky++) {
            if(ky < stamp.coords.second ||
               ky >= stamp.coords.second + args.fStampWidth)
              continue;
            for(long kx = absx - args.hSStampWidth;
                kx <= absx + args.hSStampWidth; kx++) {
              if(kx < stamp.coords.first ||
                 kx >= stamp.coords.first + args.fStampWidth)
                continue;
              long kCoords = kx + (ky * image.axis.first);

              if(mask.isMasked(kCoords, badMask)) {
                continue;
              }

              if(image[kCoords] >= args.threshHigh) {
                mask.maskPix(kx, ky, badPixelMask);
                continue;
              }

              if((image[kCoords] - stamp.stats.skyEst) *
                     (1.0 / stamp.stats.fwhm) <
                 args.threshKernFit) {
                continue;
              }

              if(image[kCoords] > s.val) {
                s.val = image[kCoords];
                s.imageCoords = std::make_pair(kx, ky);
                // s.stampCoords = std::make_pair(kx - stamp.coords.first,
                //  ky - stamp.coords.second);
              }
            }
          }
          s.val = checkSStampMp(s, image, mask, stamp, badMask, skipMask, args);
          if(s.val == 0.0) continue;
          stamp.subStamps.push_back(s);

          int startX2 = std::max(s.imageCoords.first - args.hSStampWidth,
                                 stamp.coords.first);
          int startY2 = std::max(s.imageCoords.second - args.hSStampWidth,
                                 stamp.coords.second);
          int endX2 = std::min(s.imageCoords.first + args.hSStampWidth,
                               stamp.coords.first + stamp.size.first - 1);
          int endY2 = std::min(s.imageCoords.second + args.hSStampWidth,
                               stamp.coords.second + stamp.size.second - 1);

          for(int y = startY2; y <= endY2; y++) {
            for(int x = startX2; x <= endX2; x++) {
              mask.maskPix(x, y, skipMask);
            }
          }
        }
        if(stamp.subStamps.size() >= size_t(maxSStamps)) break;
      }
      if(stamp.subStamps.size() >= size_t(maxSStamps)) break;
    }
    if(lowestPSFLim == floor) break;
    dfrac -= 0.2;
  }

  if(stamp.subStamps.size() == 0) {
    if(args.verbose)
      std::cout << "No suitable substamps found in stamp " << std::endl;
    return 1;
  }
  size_t keepSStampCount =
      std::min<size_t>(stamp.subStamps.size(), args.maxKSStamps);
  std::partial_sort(stamp.subStamps.begin(),
                    stamp.subStamps.begin() + keepSStampCount,
                    stamp.subStamps.end(), std::greater<SubStamp>());

  if(stamp.subStamps.size() > keepSStampCount) {
    stamp.subStamps.erase(stamp.subStamps.begin() + keepSStampCount,
                          stamp.subStamps.end());
  }

  if(args.verbose)
    std::cout << "Added " << stamp.subStamps.size() << " substamps to stamp "
              << std::endl;
  return 0;
}

double checkSStampMp(const SubStamp& sstamp, const Image& image,
                     ImageMask& mask, const Stamp& stamp,
                     const ImageMasks badMask, const ImageMasks skipMask,
                     const Arguments& args) {
  double retVal = 0.0;
  for(int y = sstamp.imageCoords.second - args.hSStampWidth;
      y <= sstamp.imageCoords.second + args.hSStampWidth; y++) {
    if(y < stamp.coords.second || y >= stamp.coords.second + stamp.size.second)
      continue;
    for(int x = sstamp.imageCoords.first - args.hSStampWidth;
        x <= sstamp.imageCoords.first + args.hSStampWidth; x++) {
      if(x < stamp.coords.first || x >= stamp.coords.first + stamp.size.first)
        continue;

      int absCoords = x + y * image.axis.first;
      if(mask.isMasked(absCoords, badMask)) return 0.0;

      if(image[absCoords] >= args.threshHigh) {
        mask.maskPix(x, y, skipMask);
        return 0.0;
      }
      if((image[absCoords] - stamp.stats.skyEst) / stamp.stats.fwhm >
         args.threshKernFit)
        retVal += image[absCoords];
    }
  }
  return retVal;
}

void removeEmptyStamps(const Arguments& args, ClStampsData& stampsData,
                       const ClData& clData) {
  size_t maxSStamps{2 * args.maxKSStamps};

  cl::size_type nStamps{
      static_cast<cl::size_type>(args.stampsx * args.stampsy)};
  cl::size_type paddedNStamps{static_cast<cl::size_type>(
      leastGreaterPow2(args.stampsx * args.stampsy))};

  cl::Buffer filteredStampCoords{clData.context, CL_MEM_READ_WRITE,
                                 sizeof(cl_int2) * nStamps};
  cl::Buffer filteredStampSizes{clData.context, CL_MEM_READ_WRITE,
                                sizeof(cl_int2) * nStamps};
  cl::Buffer filteredSkyEsts{clData.context, CL_MEM_READ_WRITE,
                             sizeof(cl_double) * nStamps};
  cl::Buffer filteredFwhms{clData.context, CL_MEM_READ_WRITE,
                           sizeof(cl_double) * nStamps};
  cl::Buffer filteredSubStampCoords{clData.context, CL_MEM_READ_WRITE,
                                    sizeof(cl_int2) * maxSStamps * nStamps};
  cl::Buffer filteredSubStampValues{clData.context, CL_MEM_READ_WRITE,
                                    sizeof(cl_double) * maxSStamps * nStamps};
  cl::Buffer filteredSubStampCounts{clData.context, CL_MEM_READ_WRITE,
                                    sizeof(cl_int) * maxSStamps * nStamps};

  cl::Buffer keepCounter{clData.context, CL_MEM_READ_WRITE, sizeof(cl_int)};
  cl::Buffer keepIndeces{clData.context, CL_MEM_READ_WRITE,
                         sizeof(cl_int) * paddedNStamps};

  cl::EnqueueArgs eargsMark{clData.queue, cl::NDRange{nStamps}};
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer> markFunc(
      clData.program, "markStampsToKeep");

  cl::EnqueueArgs eargsSort{clData.queue, cl::NDRange{paddedNStamps}};
  cl::KernelFunctor<cl::Buffer, cl::Buffer> padFunc(clData.program, "padMarks");

  cl::KernelFunctor<cl::Buffer, cl_int, cl_int> sortFunc(clData.program,
                                                         "sortMarks");

  cl::EnqueueArgs eargsRemove{clData.queue, cl::NDRange{nStamps}};
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                    cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                    cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                    cl::Buffer, cl::Buffer, cl_int>
      removeFunc(clData.program, "removeEmptyStamps");

  cl_int zero{0};
  clData.queue.enqueueWriteBuffer(keepCounter, CL_TRUE, 0, sizeof(cl_int),
                                  &zero);

  cl::Event markEvent{
      markFunc(eargsMark, stampsData.subStampCounts, keepIndeces, keepCounter)};
  markEvent.wait();

  cl::Event padEvent{padFunc(eargsSort, keepIndeces, keepCounter)};
  padEvent.wait();

  cl::Event sortEvent;
  // Outer loop, double size for each step
  for(size_t k = 2; k <= paddedNStamps; k = 2 * k) {
    // Inner loop, half size for each step
    for(int j = k >> 1; j > 0; j = j >> 1) {
      sortEvent = sortFunc(eargsSort, keepIndeces, j, k);
      sortEvent.wait();
    }
  }

  cl_int removedStampCount{};
  clData.queue.enqueueReadBuffer(keepCounter, CL_TRUE, 0, sizeof(cl_int),
                                 &removedStampCount);

  stampsData.stampCount = removedStampCount;
  stampsData.currentSubStamps = {clData.context, CL_MEM_READ_WRITE,
                                 sizeof(cl_int) * removedStampCount};

  cl::Event removeEvent = removeFunc(
      eargsRemove, stampsData.stampCoords, stampsData.stampSizes,
      stampsData.stats.skyEsts, stampsData.stats.fwhms,
      stampsData.subStampCounts, stampsData.subStampCoords,
      stampsData.subStampValues, filteredStampCoords, filteredStampSizes,
      filteredSkyEsts, filteredFwhms, filteredSubStampCounts,
      filteredSubStampCoords, filteredSubStampValues, keepIndeces, keepCounter,
      stampsData.currentSubStamps, maxSStamps);
  removeEvent.wait();

  stampsData.stampCoords = filteredStampCoords;
  stampsData.stampSizes = filteredStampSizes;
  stampsData.stats.skyEsts = filteredSkyEsts;
  stampsData.stats.fwhms = filteredFwhms;
  stampsData.subStampCoords = filteredSubStampCoords;
  stampsData.subStampValues = filteredSubStampValues;
  stampsData.subStampCounts = filteredSubStampCounts;
}

void resetSStampSkipMask(const int w, const int h, const ClData& clData) {
  cl::EnqueueArgs eargs{clData.queue, cl::NDRange(w * h)};
  cl::KernelFunctor<cl::Buffer> resetFunc(clData.program, "resetSkipMask");
  cl::Event unmaskEvent{resetFunc(eargs, clData.maskBuf)};
  unmaskEvent.wait();
}

void readFinalStamps(std::vector<Stamp>& stamps, const ClStampsData& stampsData,
                     const ClData& clData, const Arguments& args) {
  cl::size_type maxSStamps(2 * args.maxKSStamps);

  std::vector<cl_int2> subStampCoords(maxSStamps * stampsData.stampCount);
  std::vector<cl_double> subStampValues(maxSStamps * stampsData.stampCount);
  std::vector<cl_int> subStampCounts(maxSStamps * stampsData.stampCount);

  static constexpr int nStampBuffers{3};
  std::vector<cl::Event> readEvents(nStampBuffers);
  clData.queue.enqueueReadBuffer(
      stampsData.subStampCoords, CL_FALSE, 0,
      sizeof(cl_int2) * maxSStamps * stampsData.stampCount, &subStampCoords[0],
      nullptr, &readEvents[0]);
  clData.queue.enqueueReadBuffer(
      stampsData.subStampValues, CL_FALSE, 0,
      sizeof(cl_double) * maxSStamps * stampsData.stampCount,
      &subStampValues[0], nullptr, &readEvents[1]);
  clData.queue.enqueueReadBuffer(
      stampsData.subStampCounts, CL_FALSE, 0,
      sizeof(cl_int) * maxSStamps * stampsData.stampCount, &subStampCounts[0],
      nullptr, &readEvents[2]);
  cl::Event::waitForEvents(readEvents);

  stamps.clear();
  stamps.reserve(stampsData.stampCount);

  for(size_t i{0}; i < stampsData.stampCount; i++) {
    Stamp& stamp{stamps.emplace_back(std::vector<SubStamp>{})};

    std::vector<SubStamp>& sstamps{stamp.subStamps};

    for(int j{0}; j < subStampCounts[i]; j++) {
      size_t offset{i * maxSStamps + j};
      std::pair<cl_int, cl_int> imageCoords{subStampCoords[offset].s[0],
                                            subStampCoords[offset].s[1]};
      sstamps.emplace_back(SubStamp{imageCoords, subStampValues[offset]});
    }
  }

  assert(stampsData.stampCount == stamps.size());
}

void moveSssToGpu(const std::vector<Stamp>& templateStamps,
                  const std::vector<Stamp>& scienceStamps,
                  const ImageMask& mask, ClData& clData,
                  const std::pair<int, int>& axis, const Arguments& args) {
  // mask buf
  clData.queue.enqueueWriteBuffer(clData.maskBuf, CL_TRUE, 0,
                                  sizeof(u_int16_t) * mask.dataMask.size(),
                                  &mask.dataMask[0]);

  // std::vector<uint16_t> clMask(mask.dataMask.size());
  // clData.queue.enqueueReadBuffer(clData.maskBuf, CL_TRUE, 0,
  //                                sizeof(u_int16_t) * axis.first *
  //                                axis.second, &clMask[0]);
  // std::cout << "mask" << std::endl;
  // for(size_t i = 0; i < clMask.size(); i++) {
  //   if(clMask[i] != mask.dataMask[i]) {
  //     std::cout << "mask differs" << clMask[i] << " == " << mask.dataMask[i]
  //               << std::endl;
  //   }
  // }

  clData.tmpl.stampCount = templateStamps.size();
  clData.sci.stampCount = scienceStamps.size();

  std::cout << "copying template data" << std::endl;
  moveStamps(templateStamps, clData.tmpl, clData, args);
  std::cout << "copying science data" << std::endl;
  moveStamps(scienceStamps, clData.sci, clData, args);
}

void moveStamps(const std::vector<Stamp>& stamps, ClStampsData& stampsData,
                ClData& clData, const Arguments& args) {
  cl::size_type maxSStamps(2 * args.maxKSStamps);

  // stamp coords
  std::vector<int> coords(stamps.size() * maxSStamps * 2);
  for(size_t i = 0; i < stamps.size(); i++) {
    clData.queue.enqueueWriteBuffer(
        stampsData.stampCoords, CL_TRUE, sizeof(std::pair<int, int>) * i,
        sizeof(std::pair<int, int>), &stamps[i].coords.first);
  }

  // stamp sizes
  std::vector<int> sizes(stamps.size() * 2);
  for(size_t i = 0; i < stamps.size(); i++) {
    clData.queue.enqueueWriteBuffer(
        stampsData.stampSizes, CL_TRUE, sizeof(std::pair<int, int>) * i,
        sizeof(std::pair<int, int>), &stamps[i].size.first);
  }

  // skyest
  for(size_t i = 0; i < stamps.size(); i++) {
    clData.queue.enqueueWriteBuffer(stampsData.stats.skyEsts, CL_TRUE,
                                    sizeof(double) * i, sizeof(double),
                                    &stamps[i].stats.skyEst);
  }

  // fwhm
  for(size_t i = 0; i < stamps.size(); i++) {
    clData.queue.enqueueWriteBuffer(stampsData.stats.fwhms, CL_TRUE,
                                    sizeof(double) * i, sizeof(double),
                                    &stamps[i].stats.fwhm);
  }

  // substamp coords
  int index = 0;
  for(size_t i = 0; i < stamps.size(); i++) {
    for(size_t j = 0; j < stamps[i].subStamps.size(); j++) {
      size_t offset{i * maxSStamps + j};

      clData.queue.enqueueWriteBuffer(
          stampsData.subStampCoords, CL_TRUE,
          sizeof(std::pair<int, int>) * offset, sizeof(std::pair<int, int>),
          &stamps[i].subStamps[j].imageCoords.first);
      index++;
    }
  }

  size_t subStampMaxCount{2 * args.maxKSStamps};

  // substamp count
  std::vector<int> counts(stamps.size());
  for(size_t i = 0; i < stamps.size(); i++) {
    counts[i] = stamps[i].subStamps.size();
  }
  uploadBuffer(counts, stampsData.subStampCounts, clData.queue);

  // verify

  // std::vector<std::pair<int, int>> clCoords(stamps.size());
  // clData.queue.enqueueReadBuffer(stampsData.stampCoords, CL_TRUE, 0,
  //                                sizeof(std::pair<int, int>) *
  //                                clCoords.size(), &clCoords[0]);

  // std::vector<double> clSkyEst(stamps.size());
  // clData.queue.enqueueReadBuffer(stampsData.stats.skyEsts, CL_TRUE, 0,
  //                                sizeof(cl_double) * clSkyEst.size(),
  //                                &clSkyEst[0]);

  // std::vector<double> clFwhm(stamps.size());
  // clData.queue.enqueueReadBuffer(stampsData.stats.fwhms, CL_TRUE, 0,
  //                                sizeof(double) * stamps.size(), &clFwhm[0]);

  // std::vector<std::pair<int, int>> subStampCoords(maxSStamps *
  //                                                 stampsData.stampCount);
  // std::vector<double> subStampValues(maxSStamps * stampsData.stampCount);
  // std::vector<int> subStampCounts(maxSStamps * stampsData.stampCount);

  // static constexpr int nStampBuffers{3};
  // std::vector<cl::Event> readEvents(nStampBuffers);
  // clData.queue.enqueueReadBuffer(
  //     stampsData.subStampCoords, CL_TRUE, 0,
  //     sizeof(std::pair<int, int>) * maxSStamps * stampsData.stampCount,
  //     &subStampCoords[0]);
  // clData.queue.enqueueReadBuffer(
  //     stampsData.subStampValues, CL_TRUE, 0,
  //     sizeof(double) * maxSStamps * stampsData.stampCount,
  //     &subStampValues[0]);
  // clData.queue.enqueueReadBuffer(
  //     stampsData.subStampCounts, CL_TRUE, 0,
  //     sizeof(int) * maxSStamps * stampsData.stampCount, &subStampCounts[0]);

  // std::cout << "stampcoords" << std::endl;
  // for(size_t i = 0; i < clCoords.size(); i++) {
  //   if(clCoords[i] != stamps[i].coords) {
  //     std::cout << clCoords[i].first << "-" << clCoords[i].second << " and "
  //               << stamps[i].coords.first << "-" << stamps[i].coords.second
  //               << " are not the same!!!!!!!!!!!" << i << "\n";
  //   }
  // }

  // std::cout << "skyest" << std::endl;
  // for(size_t i = 0; i < clSkyEst.size(); i++) {
  //   if(clSkyEst[i] != stamps[i].stats.skyEst) {
  //     std::cout << clSkyEst[i] << "==" << stamps[i].stats.skyEst
  //               << " are not the same " << i << "\n";
  //   }
  // }

  // std::cout << "fwhm" << std::endl;
  // for(size_t i = 0; i < clFwhm.size(); i++) {
  //   if(clFwhm[i] != stamps[i].stats.fwhm) {
  //     std::cout << clFwhm[i] << "==" << stamps[i].stats.fwhm
  //               << " are not the same " << i << "\n";
  //   }
  // }

  // index = 0;
  // std::cout << "substamps " << std::endl;
  // for(size_t i = 0; i < stamps.size(); i++) {
  //   if(subStampCounts[i] != stamps[i].subStamps.size()) {
  //     std::cout << subStampCounts[i] << " == " << stamps[i].subStamps.size()
  //               << " substamp sizes are not the same for index " << i
  //               << std::endl;
  //   } else {
  //     for(size_t j = 0; j < stamps[i].subStamps.size(); j++) {
  //       // if(subStampValues[index] != stamps[i].subStamps[j].val) {
  //       //   std::cout << "substamp value not matching" <<
  //       subStampValues[index]
  //       //             << " == " << stamps[i].subStamps[j].val << std::endl;
  //       // }
  //       if(subStampCoords[index] != stamps[i].subStamps[j].imageCoords) {
  //         std::cout << "substamp coords not matching"
  //                   << subStampCoords[index].first << ","
  //                   << subStampCoords[index].second
  //                   << " == " << stamps[i].subStamps[j].imageCoords.first <<
  //                   ","
  //                   << stamps[i].subStamps[j].imageCoords.second <<
  //                   std::endl;
  //       }
  //       index++;
  //     }
  //   }
  // }
}