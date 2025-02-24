#include <cassert>
#include <iostream>

#include "bachUtil.h"
#include "mathUtil.h"

void identifySStampsCl(const std::pair<cl_int, cl_int>& axis,
                       const Arguments& args, ClData& clData) {
  std::cout << "Identifying sub-stamps..." << std::endl;

  if(args.verbose) std::cout << "calcStats (template)" << std::endl;
  calcStatsCl(axis, args, clData.tImgBuf, clData.tmpl, clData);
  if(args.verbose) std::cout << "calcStats (science)" << std::endl;
  calcStatsCl(axis, args, clData.sImgBuf, clData.sci, clData);

  if(args.verbose) std::cout << "findSStampsCl (template)" << std::endl;
  findSStampsCl(axis, true, args, clData.tImgBuf, clData.tmpl, clData);
  if(args.verbose) std::cout << "findSStampsCl (science)" << std::endl;
  findSStampsCl(axis, false, args, clData.sImgBuf, clData.sci, clData);
}

// feels like this function should fill the stamps vector but it just calls an
// opencl kernel my guess is that it was before a cpu only implementation
void createStampsCl(const int w, const int h, ClStampsData& stampsData,
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

// why is this not a void function?
cl_int findSStampsCl(const std::pair<cl_int, cl_int>& axis,
                     const bool isTemplate, const Arguments& args,
                     const cl::Buffer& imgBuf, const ClStampsData& stampsData,
                     const ClData& clData) {
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

  cl_int maxSStamps{2 * args.maxKSStamps};

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

void removeEmptyStampsCl(const Arguments& args, ClStampsData& stampsData,
                         const ClData& clData) {
  int maxSStamps{2 * args.maxKSStamps};

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

void resetSStampSkipMaskCl(const int w, const int h, const ClData& clData) {
  cl::EnqueueArgs eargs{clData.queue, cl::NDRange(w * h)};
  cl::KernelFunctor<cl::Buffer> resetFunc(clData.program, "resetSkipMask");
  cl::Event unmaskEvent{resetFunc(eargs, clData.maskBuf)};
  unmaskEvent.wait();
}

void readFinalStampsCl(std::vector<Stamp>& stamps,
                       const ClStampsData& stampsData, const ClData& clData,
                       const Arguments& args) {
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
    auto& stamp{stamps.emplace_back(std::vector<SubStamp>{})};

    auto& sstamps{stamp.subStamps};

    for(int j{0}; j < subStampCounts[i]; j++) {
      size_t offset{i * maxSStamps + j};
      std::pair<cl_int, cl_int> imageCoords{subStampCoords[offset].s[0],
                                            subStampCoords[offset].s[1]};
      sstamps.emplace_back(SubStamp{imageCoords, subStampValues[offset]});
    }
  }

  assert(stampsData.stampCount == stamps.size());
}

void sigmaClipCl(const cl::Buffer& data, int dataOffset, int dataCount,
                 double* mean, double* stdDev, int maxIter,
                 const ClData& clData, const Arguments& args) {
  if(dataCount == 0) {
    std::cout << "Cannot send in empty vector to Sigma Clip" << std::endl;
    *mean = 0.0;
    *stdDev = 1e10;
    return;
  }

  constexpr int localSize = 32;
  int reduceCount = (dataCount + localSize - 1) / localSize;

  std::vector<cl_double> sumVec(reduceCount);
  std::vector<cl_double> sum2Vec(reduceCount);

  cl::Buffer intMask(clData.context, CL_MEM_READ_WRITE,
                     sizeof(cl_uchar) * dataCount);
  cl::Buffer clipCountBuf(clData.context, CL_MEM_READ_WRITE, sizeof(cl_int));
  cl::Buffer sumBuf(clData.context, CL_MEM_READ_WRITE,
                    sizeof(cl_double) * reduceCount);
  cl::Buffer sum2Buf(clData.context, CL_MEM_READ_WRITE,
                     sizeof(cl_double) * reduceCount);

  cl::KernelFunctor<cl::Buffer> initMaskFunc(clData.program,
                                             "sigmaClipInitMask");
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl_int>
      calcFunc(clData.program, "sigmaClipCalc");
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl_double, cl_double,
                    cl_double>
      maskFunc(clData.program, "sigmaClipMask");

  cl::EnqueueArgs calcEargs(clData.queue, cl::NDRange(dataOffset),
                            cl::NDRange(reduceCount * localSize),
                            cl::NDRange(localSize));
  cl::EnqueueArgs maskEargs(clData.queue, cl::NDRange(dataOffset),
                            cl::NDRange(dataCount), cl::NullRange);

  // Zero mask
  cl::Event initMaskEvent = initMaskFunc(maskEargs, intMask);
  initMaskEvent.wait();

  size_t currNPoints = 0;
  size_t prevNPoints = dataCount;

  // Do three times or a stable solution has been found.
  for(int i = 0; (i < maxIter) && (currNPoints != prevNPoints); i++) {
    if(prevNPoints <= 1) {
      std::cout << "prevNPoints is: " << prevNPoints
                << "Needs to be greater than 1" << std::endl;
      *mean = 0.0;
      *stdDev = 1e10;
      return;
    }

    currNPoints = prevNPoints;

    // Calculate mean and standard deviation
    cl::Event calcEvent =
        calcFunc(calcEargs, sumBuf, sum2Buf, data, intMask, dataCount);
    calcEvent.wait();

    // Can be optimized to use a tree structure instead of reducing on CPU
    clData.queue.enqueueReadBuffer(
        sumBuf, CL_TRUE, 0, sizeof(cl_double) * sumVec.size(), sumVec.data());
    clData.queue.enqueueReadBuffer(sum2Buf, CL_TRUE, 0,
                                   sizeof(cl_double) * sum2Vec.size(),
                                   sum2Vec.data());

    double sum = std::accumulate(sumVec.begin(), sumVec.end(), 0.0);
    double sum2 = std::accumulate(sum2Vec.begin(), sum2Vec.end(), 0.0);

    double tempMean = sum / prevNPoints;
    double tempStdDev = std::sqrt((sum2 - prevNPoints * tempMean * tempMean) /
                                  (prevNPoints - 1));

    double invStdDev = 1.0 / tempStdDev;

    cl_int clipCount = 0;
    clData.queue.enqueueWriteBuffer(clipCountBuf, CL_TRUE, 0, sizeof(cl_int),
                                    &clipCount);

    // Mask bad values
    cl::Event maskEvent = maskFunc(maskEargs, intMask, clipCountBuf, data,
                                   invStdDev, tempMean, args.sigClipAlpha);
    maskEvent.wait();

    clData.queue.enqueueReadBuffer(clipCountBuf, CL_TRUE, 0, sizeof(cl_int),
                                   &clipCount);

    prevNPoints = currNPoints - clipCount;
    *mean = tempMean;
    *stdDev = tempStdDev;
  }
}

void calcStatsCl(const std::pair<cl_int, cl_int>& axis, const Arguments& args,
                 const cl::Buffer& imgBuf, const ClStampsData& stampsData,
                 const ClData& clData) {
  /* Heavily taken from HOTPANTS which itself copied it from Gary Bernstein
   * Calculates important values of stamps for futher calculations.
   */
  auto&& [imgW, imgH] = axis;

  cl::size_type nStamps{
      static_cast<cl::size_type>(args.stampsx * args.stampsy)};

  static constexpr cl_int nSamples{100};
  static constexpr cl_int paddedNSamples{leastGreaterPow2(nSamples)};

  {
    std::vector<cl_int2> stampSizes(nStamps);
    clData.queue.enqueueReadBuffer(stampsData.stampSizes, CL_TRUE, 0,
                                   sizeof(cl_int2) * stampsData.stampCount,
                                   &stampSizes[0]);
    for(size_t i{0}; i < stampsData.stampCount; i++) {
      cl_int stampNumPix = stampSizes[i].s[0] * stampSizes[i].s[1];
      if(stampNumPix < nSamples) {
        std::cout << "Not enough pixels in a stamp" << std::endl;
        std::exit(1);
      }
    }
  }

  cl_int nPix{args.fStampWidth * args.fStampWidth};
  cl::Buffer samples{clData.context, CL_MEM_READ_WRITE,
                     sizeof(cl_double) * nSamples * nStamps};
  cl::Buffer paddedSamples{clData.context, CL_MEM_READ_WRITE,
                           sizeof(cl_double) * paddedNSamples * nStamps};
  cl::Buffer sampleCounts{clData.context, CL_MEM_READ_WRITE,
                          sizeof(cl_int) * nStamps};

  cl::Buffer goodPixels{clData.context, CL_MEM_READ_WRITE,
                        sizeof(cl_double) * nPix * nStamps};
  cl::Buffer goodPixelCounts{clData.context, CL_MEM_READ_WRITE,
                             sizeof(cl_int) * nStamps};

  cl::Buffer bins{clData.context, CL_MEM_READ_WRITE,
                  sizeof(cl_int) * 256 * nStamps};
  cl::Buffer means{clData.context, CL_MEM_READ_ONLY,
                   sizeof(cl_double) * nStamps};
  cl::Buffer invStdDevs{clData.context, CL_MEM_READ_ONLY,
                        sizeof(cl_double) * nStamps};
  cl::Buffer binSizes{clData.context, CL_MEM_READ_ONLY,
                      sizeof(cl_double) * nStamps};
  cl::Buffer lowerBinVals{clData.context, CL_MEM_READ_ONLY,
                          sizeof(cl_double) * nStamps};

  cl::EnqueueArgs eargsSample{clData.queue, cl::NDRange{nStamps}};
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                    cl::Buffer, cl_int, cl_int>
      sampleStampFunc(clData.program, "sampleStamp");

  cl::EnqueueArgs eargsPadSamples{clData.queue,
                                  cl::NDRange{paddedNSamples, nStamps}};
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl_int, cl_int> padFunc(
      clData.program, "pad");

  cl::EnqueueArgs eargsSortSamples{clData.queue,
                                   cl::NDRange(paddedNSamples * nStamps)};
  cl::KernelFunctor<cl::Buffer, cl_int, cl_int, cl_int> sortSamplesFunc(
      clData.program, "sortSamples");

  cl::EnqueueArgs eargsResetGoodPixelCounts{clData.queue, cl::NDRange{nStamps}};
  cl::KernelFunctor<cl::Buffer> resetGoodPixelCountsFunc(
      clData.program, "resetGoodPixelCounts");

  cl::EnqueueArgs eargsMask{clData.queue, cl::NDRange(nPix, nStamps)};
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                    cl_int, cl_int, cl_int>
      maskFunc(clData.program, "maskStamp");

  static constexpr int histogramLocalSize = 4;
  cl::EnqueueArgs eargsHistogram(
      clData.queue, cl::NDRange(roundUpToMultiple(nStamps, histogramLocalSize)),
      cl::NDRange(histogramLocalSize));
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                    cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                    cl::Buffer, cl_int, cl_int, cl_int, cl_int, cl_double,
                    cl_double>
      histogramFunc(clData.program, "createHistogram");

  cl::Event sampleEvent = sampleStampFunc(
      eargsSample, imgBuf, clData.maskBuf, stampsData.stampCoords,
      stampsData.stampSizes, samples, sampleCounts, imgW, nSamples);
  sampleEvent.wait();

  cl::Event resetEvent =
      resetGoodPixelCountsFunc(eargsResetGoodPixelCounts, goodPixelCounts);
  resetEvent.wait();

  cl::Event padEvent = padFunc(eargsPadSamples, samples, paddedSamples,
                               nSamples, paddedNSamples);
  padEvent.wait();

  cl::Event sortEvent;
  for(cl_int k = 2; k <= paddedNSamples;
      k = 2 * k)  // Outer loop, double size for each step
  {
    for(cl_int j = k >> 1; j > 0;
        j = j >> 1)  // Inner loop, half size for each step
    {
      sortEvent = sortSamplesFunc(eargsSortSamples, paddedSamples,
                                  paddedNSamples, j, k);
      sortEvent.wait();
    }
  }

  cl::Event maskEvent =
      maskFunc(eargsMask, imgBuf, clData.maskBuf, stampsData.stampCoords,
               goodPixels, goodPixelCounts, args.fStampWidth, imgW, imgH);
  maskEvent.wait();

  std::vector<cl_int> cpuGoodPixelCounts(nStamps);
  std::vector<cl_double> cpuMeans(nStamps);
  std::vector<cl_double> cpuInvStdDevs(nStamps);

  clData.queue.enqueueReadBuffer(goodPixelCounts, CL_TRUE, 0,
                                 sizeof(cl_int) * cpuGoodPixelCounts.size(),
                                 &cpuGoodPixelCounts[0]);

  for(size_t stampIdx{0}; stampIdx < nStamps; stampIdx++) {
    int goodPixelCount{cpuGoodPixelCounts[stampIdx]};

    // sigma clip of maskedStamp to get mean and sd.
    double mean, stdDev, invStdDev;
    sigmaClipCl(goodPixels, stampIdx * nPix, goodPixelCount, &mean, &stdDev, 3,
                clData, args);
    invStdDev = 1.0 / stdDev;

    cpuMeans[stampIdx] = mean;
    cpuInvStdDevs[stampIdx] = invStdDev;
  }

  clData.queue.enqueueWriteBuffer(means, CL_TRUE, 0,
                                  sizeof(cl_double) * nStamps, &cpuMeans[0]);
  clData.queue.enqueueWriteBuffer(
      invStdDevs, CL_TRUE, 0, sizeof(cl_double) * nStamps, &cpuInvStdDevs[0]);

  cl::Event histogramEvent = histogramFunc(
      eargsHistogram, imgBuf, clData.maskBuf, stampsData.stampCoords,
      stampsData.stampSizes, means, invStdDevs, paddedSamples, sampleCounts,
      bins, stampsData.stats.fwhms, stampsData.stats.skyEsts, axis.first,
      nStamps, nSamples, paddedNSamples, args.iqRange, args.sigClipAlpha);

  histogramEvent.wait();
}

void maskInputCl(const std::pair<cl_int, cl_int>& axis, const ClData& clData,
                 const Arguments& args) {
  // Create mask from input data
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl_int, cl_int, cl_int,
                    cl_double, cl_double>
      maskFunc(clData.program, "maskInput");
  cl::EnqueueArgs maskEargs(clData.queue,
                            cl::NDRange(axis.first * axis.second));
  cl::Event maskEvent =
      maskFunc(maskEargs, clData.tImgBuf, clData.sImgBuf, clData.maskBuf,
               axis.first, axis.second, args.hSStampWidth + args.hKernelWidth,
               args.threshHigh, args.threshLow);
  maskEvent.wait();

  // Spread mask
  int spreadWidth =
      static_cast<int>(args.hKernelWidth * args.inSpreadMaskFactor);
  cl::KernelFunctor<cl::Buffer, cl_int, cl_int, cl_int> spreadFunc(
      clData.program, "spreadMask");
  cl::EnqueueArgs spreadEargs(clData.queue,
                              cl::NDRange(axis.first, axis.second));
  cl::Event spreadEvent = spreadFunc(spreadEargs, clData.maskBuf, axis.first,
                                     axis.second, spreadWidth);
  spreadEvent.wait();
}