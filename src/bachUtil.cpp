#include "bachUtil.h"

#include <omp.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <numeric>

#include "mathUtil.h"

void maskInput(const std::pair<cl_int, cl_int>& axis, const ClData& clData,
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

void sigmaClip(const cl::Buffer& data, int dataOffset, int dataCount,
               double* mean, double* stdDev, int maxIter, const ClData& clData,
               const Arguments& args) {
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

void sigmaClipMp(const std::vector<double>& data, double& mean, double& stdDev,
                 const int iter, const Arguments& args) {
  /* Does sigma clipping on data to provide the mean and stdDev of said
   * data
   */
  if(data.empty()) {
    std::cout << "Cannot send in empty vector to Sigma Clip" << std::endl;
    mean = 0.0;
    stdDev = 1e10;
    return;
  }

  size_t currNPoints = 0;
  size_t prevNPoints = data.size();
  std::vector<bool> intMask(data.size(), false);

  // Do three times or a stable solution has been found.
  for(int i = 0; (i < iter) && (currNPoints != prevNPoints); i++) {
    currNPoints = prevNPoints;
    mean = 0;
    stdDev = 0;

    for(size_t i = 0; i < data.size(); i++) {
      if(!intMask[i]) {
        mean += data[i];
        stdDev += data[i] * data[i];
      }
    }

    if(prevNPoints > 1) {
      mean = mean / prevNPoints;
      stdDev = stdDev - prevNPoints * mean * mean;
      stdDev = std::sqrt(stdDev / double(prevNPoints - 1));
    } else {
      std::cout << "prevNPoints is: " << prevNPoints
                << "Needs to be greater than 1" << std::endl;
      mean = 0.0;
      stdDev = 1e10;
      return;
    }

    prevNPoints = 0;
    double invStdDev = 1.0 / stdDev;
    for(size_t i = 0; i < data.size(); i++) {
      if(!intMask[i]) {
        // Doing the sigmaClip
        if(std::abs(data[i] - mean) * invStdDev > args.sigClipAlpha) {
          intMask[i] = true;
        } else {
          prevNPoints++;
        }
      }
    }
  }
}

void calcStats(const std::pair<cl_int, cl_int>& axis, const Arguments& args,
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

  // check that stamps are big enough
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
    sigmaClip(goodPixels, stampIdx * nPix, goodPixelCount, &mean, &stdDev, 3,
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

#define M1 259200
#define IA1 7141
#define IC1 54773
#define RM1 (1.0 / M1)
#define M2 134456
#define IA2 8121
#define IC2 28411
#define RM2 (1.0 / M2)
#define M3 243000
#define IA3 4561
#define IC3 51349
double ran1(int* idum, long* ix1, long* ix2, long* ix3, double* r, int* iff) {
  double temp;
  int j;
  /* void nrerror(char *error_text); */

  if(*idum < 0 || *iff == 0) {
    *iff = 1;
    *ix1 = (IC1 - (*idum)) % M1;
    *ix1 = (IA1 * (*ix1) + IC1) % M1;
    *ix2 = *ix1 % M2;
    *ix1 = (IA1 * (*ix1) + IC1) % M1;
    *ix3 = (*ix1) % M3;
    for(j = 1; j <= 97; j++) {
      *ix1 = (IA1 * (*ix1) + IC1) % M1;
      *ix2 = (IA2 * (*ix2) + IC2) % M2;
      r[j] = ((*ix1) + (*ix2) * RM2) * RM1;
    }
    *idum = 1;
  }
  *ix1 = (IA1 * (*ix1) + IC1) % M1;
  *ix2 = (IA2 * (*ix2) + IC2) % M2;
  *ix3 = (IA3 * (*ix3) + IC3) % M3;
  j = 1 + ((97 * (*ix3)) / M3);
  /* if (j > 97 || j < 1) nrerror("RAN1: This cannot happen."); */
  temp = r[j];
  r[j] = (*ix1 + (*ix2) * RM2) * RM1;
  return temp;
}
#undef M1
#undef IA1
#undef IC1
#undef RM1
#undef M2
#undef IA2
#undef IC2
#undef RM2
#undef M3
#undef IA3
#undef IC3

// computes stamp.stats.skyEst and stamp.stats.fwhm
void calcStatsMp(Stamp& stamp, const Image& image, ImageMask& mask,
                 const Arguments& args) {
  constexpr int nSamples = 100;

  // check that stamps are big enough
  int numPix = stamp.size.first * stamp.size.second;
  if(numPix < nSamples) {
    std::cout << "Not enough pixels in a stamp" << std::endl;
    exit(1);
  }

  // seed for random number generator (?)
  int idum = -666;
  long ix1, ix2, ix3;
  double r[98];
  int iff = 0;

  // sample randomly
  std::array<double, nSamples> samples{};
  int samplesCount = 0;

  // Stop after randomly having selected a pixel numPix times.
  for(int iter = 0; samplesCount < nSamples && iter < numPix; iter++) {
    int randX =
        std::floor(ran1(&idum, &ix1, &ix2, &ix3, r, &iff) * stamp.size.first);
    int randY =
        std::floor(ran1(&idum, &ix1, &ix2, &ix3, r, &iff) * stamp.size.second);

    // Random pixel in stamp in Image coords.
    int xI = randX + stamp.coords.first;
    int yI = randY + stamp.coords.second;
    int indexI = xI + yI * image.axis.first;

    if(mask.isMaskedAny(indexI) || std::abs(image[indexI]) <= 1e-10) {
      continue;
    }

    samples[samplesCount++] = image[indexI];
  }

  std::sort(samples.begin(), samples.end());

  double upProc = 0.9;
  double midProc = 0.5;
  // Width of a histogram bin.
  double binSize = (samples[(int)(upProc * samplesCount)] -
                    samples[(int)(midProc * samplesCount)]) /
                   (double)nSamples;

  // Value of lowest bin.
  double lowerBinVal =
      samples[(int)(midProc * samplesCount)] - (128.0 * binSize);

  // Contains all good Pixels in the stamp, aka not masked.
  std::vector<double> maskedStamp{};
  for(int y = 0; y < stamp.size.second; y++) {
    for(int x = 0; x < stamp.size.first; x++) {
      // Pixel in stamp in Image coords.
      int xI = x + stamp.coords.first;
      int yI = y + stamp.coords.second;
      int indexI = xI + yI * image.axis.first;

      if(mask.isMaskedAny(indexI) || image[indexI] <= 1e-10) {
        continue;
      }

      if(std::isnan(image[indexI])) {
        // I believe this should never happen
        std::cout << "non-masked NaN pixel in image" << std::endl;
        mask.maskPix(xI, yI, ImageMasks::NAN_PIXEL | ImageMasks::BAD_INPUT);
        continue;
      }

      maskedStamp.push_back(image[indexI]);
    }
  }

  // sigma clip of maskedStamp to get mean and sd.
  double mean, stdDev, invStdDev;
  sigmaClipMp(maskedStamp, mean, stdDev, 3, args);
  invStdDev = 1.0 / stdDev;

  double lower;
  double upper;
  int attempts = 0;
  std::vector<int> bins(256, 0);
  while(true) {
    if(attempts >= 5) {
      std::cout << "Creation of histogram unsuccessful after 5 attempts"
                << std::endl;
      return;
    }

    std::fill(bins.begin(), bins.end(), 0);

    int okCount = 0;

    for(int y = 0; y < stamp.size.second; y++) {
      for(int x = 0; x < stamp.size.first; x++) {
        // Pixel in stamp in Image coords.
        int xI = x + stamp.coords.first;
        int yI = y + stamp.coords.second;
        int indexI = xI + yI * image.axis.first;

        if(mask.isMaskedAny(indexI) || image[indexI] <= 1e-10) {
          continue;
        }

        if((std::abs(image[indexI] - mean) * invStdDev) > args.sigClipAlpha) {
          continue;
        }
        int index = std::clamp(
            (int)std::floor((image[indexI] - lowerBinVal) / binSize) + 1, 0,
            255);

        bins[index]++;
        okCount++;
      }
    }

    if(okCount == 0 || binSize == 0.0) {
      std::cout << "No good pixels or variation in pixels" << std::endl;
      return;
    }

    double sumBins = 0.0;
    double maxDens = 0.0;
    int lowerIndex = 1;
    int upperIndex = 1;
    int maxIndex = -1;
    while(upperIndex < 255) {
      while(sumBins < okCount / 10.0 && upperIndex < 255) {
        sumBins += bins[upperIndex++];
      }
      if(sumBins / (upperIndex - lowerIndex) > maxDens) {
        maxDens = sumBins / (upperIndex - lowerIndex);
        maxIndex = lowerIndex;
      }
      sumBins -= bins[lowerIndex++];
    }
    if(maxIndex < 0 || maxIndex > 255) maxIndex = 0;

    sumBins = 0.0;
    double sumExpect = 0.0;
    for(int i = maxIndex; sumBins < okCount / 10.0 && i < 255; i++) {
      sumBins += bins[i];
      sumExpect += i * bins[i];
    }

    double modeBin = sumExpect / sumBins + 0.5;
    stamp.stats.skyEst = lowerBinVal + binSize * (modeBin - 1.0);

    lower = okCount * 0.25;
    upper = okCount * 0.75;
    sumBins = 0.0;

    int i = 0;
    while(sumBins < lower) {
      sumBins += bins[i++];
    }
    lower = i - (sumBins - lower) / bins[i - 1];
    while(sumBins < upper) {
      sumBins += bins[i++];
    }
    upper = i - (sumBins - upper) / bins[i - 1];

    if(lower < 1.0 || upper > 255.0) {
      lowerBinVal -= 128.0 * binSize;
      binSize *= 2;
    } else if(upper - lower < 40.0) {
      binSize /= 3.0;
      lowerBinVal = stamp.stats.skyEst - 128.0 * binSize;
    } else {
      break;
    }

    attempts++;
  }
  stamp.stats.fwhm = binSize * (upper - lower) / args.iqRange;
}

long long timeDiff(std::chrono::time_point<std::chrono::steady_clock> end,
                   std::chrono::time_point<std::chrono::steady_clock> start) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
      .count();
}

void ludcmp(const cl::Buffer& matrix, int matrixSize, int stampCount,
            const cl::Buffer& index, const cl::Buffer& vv,
            const ClData& clData) {
  // Find big values
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl_int> bigFunc(clData.program,
                                                            "ludcmpBig");
  cl::EnqueueArgs bigEargs(clData.queue, cl::NDRange(matrixSize, stampCount));
  cl::Event bigEvent = bigFunc(bigEargs, matrix, vv, matrixSize);

  bigEvent.wait();

  // Rest of LU-decomposition
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl_int> restFunc(
      clData.program, "ludcmpRest");
  cl::EnqueueArgs restEargs(clData.queue, cl::NDRange(stampCount));
  cl::Event restEvent = restFunc(restEargs, vv, matrix, index, matrixSize);

  restEvent.wait();
}

void lubksb(const cl::Buffer& matrix, int matrixSize, int stampCount,
            const cl::Buffer& index, const cl::Buffer& result,
            const ClData& clData) {
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl_int> func(
      clData.program, "lubksb");
  cl::EnqueueArgs eargs(clData.queue, cl::NDRange(stampCount));
  cl::Event event = func(eargs, matrix, index, result, matrixSize);

  event.wait();
}

int ludcmp(std::vector<std::vector<double>>& matrix, int matrixSize,
           std::vector<int>& index, double& d, const Arguments& args) {
  std::vector<double> vv(matrixSize + 1, 0.0);
  int maxI{};
  double temp2{};

  d = 1.0;

  // Calculate vv
  for(int i = 1; i <= matrixSize; i++) {
    double big = 0.0;
    for(int j = 1; j <= matrixSize; j++) {
      temp2 = fabs(matrix[i][j]);
      if(temp2 > big) big = temp2;
    }
    if(big == 0.0) {
      if(args.verbose)
        std::cout << " Numerical Recipies run error" << std::endl;
      return 1;
    }
    vv[i] = 1.0 / big;
  }

  // Do the rest
  for(int j = 1; j <= matrixSize; j++) {
    for(int i = 1; i < j; i++) {
      double sum = matrix[i][j];
      for(int k = 1; k < i; k++) {
        sum -= matrix[i][k] * matrix[k][j];
      }
      matrix[i][j] = sum;
    }
    double big = 0.0;
    for(int i = j; i <= matrixSize; i++) {
      double sum = matrix[i][j];
      for(int k = 1; k < j; k++) {
        sum -= matrix[i][k] * matrix[k][j];
      }
      matrix[i][j] = sum;
      double dum = vv[i] * fabs(sum);
      if(dum >= big) {
        big = dum;
        maxI = i;
      }
    }
    if(j != maxI) {
      for(int k = 1; k <= matrixSize; k++) {
        double dum = matrix[maxI][k];
        matrix[maxI][k] = matrix[j][k];
        matrix[j][k] = dum;
      }
      d = -d;
      vv[maxI] = vv[j];
    }
    index[j] = maxI;
    matrix[j][j] = matrix[j][j] == 0.0 ? 1.0e-20 : matrix[j][j];
    if(j != matrixSize) {
      double dum = 1.0 / matrix[j][j];
      for(int i = j + 1; i <= matrixSize; i++) {
        matrix[i][j] *= dum;
      }
    }
  }

  return 0;
}

void lubksb(std::vector<std::vector<double>>& matrix, const int matrixSize,
            const std::vector<int>& index, std::vector<double>& result) {
  int ii{};

  for(int i = 1; i <= matrixSize; i++) {
    int ip = index[i];
    double sum = result[ip];
    result[ip] = result[i];
    if(ii) {
      for(int j = ii; j <= i - 1; j++) {
        sum -= matrix[i][j] * result[j];
      }
    } else if(sum) {
      ii = i;
    }
    result[i] = sum;
  }

  for(int i = matrixSize; i >= 1; i--) {
    double sum = result[i];
    for(int j = i + 1; j <= matrixSize; j++) {
      sum -= matrix[i][j] * result[j];
    }
    result[i] = sum / matrix[i][i];
  }
}

double makeKernel(const cl::Buffer& kernel, const cl::Buffer& kernSolution,
                  const std::pair<cl_int, cl_int>& imgSize, const int x,
                  const int y, const Arguments& args, const ClData& clData) {
  double hWidth = 0.5 * imgSize.first;
  double hHeight = 0.5 * imgSize.second;

  double xf = (x - hWidth) / hWidth;
  double yf = (y - hHeight) / hHeight;

  static constexpr int localCount = 32;

  // Create buffers
  cl::Buffer kernCoeffs(clData.context, CL_MEM_READ_WRITE,
                        sizeof(cl_double) * args.nPSF);
  cl::Buffer kernelSum(
      clData.context, CL_MEM_READ_WRITE,
      sizeof(cl_double) * args.fKernelWidth * args.fKernelWidth);
  cl::Buffer kernelSum2(
      clData.context, CL_MEM_READ_WRITE,
      sizeof(cl_double) *
          ((args.fKernelWidth * args.fKernelWidth + localCount - 1) /
           localCount));

  // Create coefficients
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl_int, cl_int, cl_double,
                    cl_double>
      coeffFunc(clData.program, "makeKernelCoeffs");
  cl::EnqueueArgs coeffEargs(clData.queue, cl::NDRange(args.nPSF));
  cl::Event coeffEvent =
      coeffFunc(coeffEargs, kernSolution, kernCoeffs, args.kernelOrder,
                triNum(args.kernelOrder + 1), xf, yf);

  coeffEvent.wait();

  // Create kernel
  static constexpr int kernelLocalSize = 16;
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::LocalSpaceArg,
                    cl_int, cl_int>
      kernelFunc(clData.program, "makeKernel");
  cl::EnqueueArgs kernelEargs(
      clData.queue,
      cl::NDRange(roundUpToMultiple(args.fKernelWidth * args.fKernelWidth,
                                    kernelLocalSize)),
      cl::NDRange(kernelLocalSize));
  cl::Event kernelEvent =
      kernelFunc(kernelEargs, kernCoeffs, clData.kernel.vec, kernel,
                 cl::Local(kernelLocalSize * sizeof(cl_double)), args.nPSF,
                 args.fKernelWidth);

  kernelEvent.wait();

  // Sum kernel
  cl::Event copyEvent{};
  clData.queue.enqueueCopyBuffer(
      kernel, kernelSum, 0, 0,
      sizeof(cl_double) * args.fKernelWidth * args.fKernelWidth, nullptr,
      &copyEvent);

  copyEvent.wait();

  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::LocalSpaceArg, cl_int> sumFunc(
      clData.program, "sumKernel");
  int sumCount = args.fKernelWidth * args.fKernelWidth;

  cl::Buffer* src = &kernelSum;
  cl::Buffer* dst = &kernelSum2;

  while(sumCount > 1) {
    cl::EnqueueArgs sumEargs(
        clData.queue, cl::NDRange(roundUpToMultiple(sumCount, localCount)),
        cl::NDRange(localCount));
    cl::Event sumEvent =
        sumFunc(sumEargs, *src, *dst, cl::Local(localCount * sizeof(cl_double)),
                sumCount);

    sumEvent.wait();

    sumCount = (sumCount + localCount - 1) / localCount;
    std::swap(src, dst);
  }

  // Transfer sum to CPU
  cl_double sumKernel = 0.0;
  clData.queue.enqueueReadBuffer(*src, CL_TRUE, 0, sizeof(cl_double),
                                 &sumKernel);

  return sumKernel;
}

double makeKernel(const Kernel& kern, std::vector<double>& currKernel,
                  const std::pair<cl_int, cl_int>& imgSize, const int x,
                  const int y, const Arguments& args) {
  /*
   * Calculates the kernel for a certain pixel, need finished kernelSol.
   */

  int k = 2;
  std::vector<double> kernCoeffs(args.nPSF, 0.0);
  std::pair<double, double> hImgAxis =
      std::make_pair(0.5 * imgSize.first, 0.5 * imgSize.second);
  double xf = (x - hImgAxis.first) / hImgAxis.first;
  double yf = (y - hImgAxis.second) / hImgAxis.second;

  for(int i = 1; i < args.nPSF; i++) {
    double aX = 1.0;
    for(int iX = 0; iX <= args.kernelOrder; iX++) {
      double aY = 1.0;
      for(int iY = 0; iY <= args.kernelOrder - iX; iY++) {
        kernCoeffs[i] += kern.solution[k++] * aX * aY;
        aY *= yf;
      }
      aX *= xf;
    }
  }
  kernCoeffs[0] = kern.solution[1];

  double sumKernel = 0.0;
  for(int i = 0; i < args.fKernelWidth * args.fKernelWidth; i++) {
    for(int psf = 0; psf < args.nPSF; psf++) {
      currKernel[i] += kernCoeffs[psf] * kern.kernVec[psf][i];
    }
    sumKernel += currKernel[i];
  }

  return sumKernel;
}

void convCl(const int w, const int h, const std::vector<cl_double> convKernels,
            const int xSteps, const bool scaleConv, const double invKernSum,
            Image& convImg, const Arguments& args, ClData& clData) {
  // Declare all the buffers which will be need in opencl operations.
  cl::Buffer convMaskBuf(clData.context, CL_MEM_READ_ONLY,
                         sizeof(cl_ushort) * w * h);
  cl::Buffer kernBuf(clData.context, CL_MEM_READ_ONLY,
                     sizeof(cl_double) * convKernels.size());

  // Write necessary data for convolution
  clData.queue.enqueueWriteBuffer(kernBuf, CL_TRUE, 0,
                                  sizeof(cl_double) * convKernels.size(),
                                  convKernels.data());

  // Create convolution mask
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl_int, cl_double, cl_double>
      createMaskFunc(clData.program, "createConvMask");
  cl::EnqueueArgs createMaskEargs(clData.queue, cl::NDRange(w, h));
  cl::Event createMaskEvent =
      createMaskFunc(createMaskEargs, clData.tImgBuf, convMaskBuf, w,
                     args.threshHigh, args.threshLow);

  createMaskEvent.wait();

  int nBgComp = (args.nPSF - 1) * triNum(args.kernelOrder + 1) + 1;
  // Convolve
  cl::KernelFunctor<cl::Buffer, cl_int, cl_int, cl::Buffer, cl::Buffer,
                    cl::Buffer, cl::Buffer, cl::Buffer, cl_int, cl_int, cl_int,
                    cl_int, cl_double>
      convFunc(clData.program, "conv");
  cl::EnqueueArgs eargs(clData.queue, cl::NDRange(w * h));
  cl::Event convEvent = convFunc(
      eargs, kernBuf, args.fKernelWidth, xSteps, clData.tImgBuf, clData.convImg,
      convMaskBuf, clData.maskBuf, clData.kernel.solution, w, h,
      args.backgroundOrder, nBgComp, scaleConv ? invKernSum : 1.0);
  convEvent.wait();

  // Transfer convoluted image back to CPU
  clData.queue.enqueueReadBuffer(clData.convImg, CL_TRUE, 0,
                                 sizeof(cl_double) * w * h, &convImg);

  // Mask after convolve
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl_int, cl_double, cl_double>
      maskAfterFunc(clData.program, "maskAfterConv");
  cl::EnqueueArgs maskAfterEargs(clData.queue, cl::NDRange(w, h));
  cl::Event maskAfterEvent =
      maskAfterFunc(maskAfterEargs, clData.sImgBuf, clData.maskBuf, w,
                    args.threshHigh, args.threshLow);

  maskAfterEvent.wait();
}

double getBackground(const int x, const int y, const std::vector<double>& sol,
                     const int width, const int height, const int bgOrder,
                     const int nBgComp) {
  double xf = (x - 0.5 * width) / (0.5 * width);
  double yf = (y - 0.5 * height) / (0.5 * height);

  double bg = 0.0;
  int k = nBgComp + 1;

  double ax = 1.0;

  for(int i = 0; i <= bgOrder; i++) {
    double ay = 1.0;

    for(int j = 0; j <= bgOrder - i; j++) {
      bg += sol[k++] * ax * ay;

      ay *= yf;
    }

    ax *= xf;
  }

  return bg;
}

void convMp(const int w, const int h, const std::vector<cl_double>& convKernels,
            const int xSteps, const bool scaleConv, const double invKernSum,
            Image& convImg, const Image& templateImage,
            const Image& scienceImage, const std::vector<double>& kernSolution,
            const Arguments& args, ClData& clData) {
  // create conv mask
  ImageMask convMask(w, h);

#pragma omp parallel for collapse(2) schedule(static) default(none) \
    shared(w, h, convMask, templateImage, args)
  for(int y = 0; y < h; y++) {
    for(int x = 0; x < w; x++) {
      int id = x + y * w;
      double t = templateImage[id];

      if(t == 0) {
        convMask.maskPix(x, y, ImageMasks::BAD_INPUT | ImageMasks::BAD_PIX_VAL);
      }

      if(t >= args.threshHigh) {
        convMask.maskPix(x, y, ImageMasks::BAD_INPUT | ImageMasks::SAT_PIXEL);
      }

      if(t <= args.threshLow) {
        convMask.maskPix(x, y, ImageMasks::BAD_INPUT | ImageMasks::LOW_PIXEL);
      }
    }
  }

  ImageMask mask{w, h};
  clData.queue.enqueueReadBuffer(clData.maskBuf, CL_TRUE, 0,
                                 sizeof(uint16_t) * w * h, &mask.dataMask[0]);

  // convolution
  int halfConvWidth = args.fKernelWidth / 2;
  int nBgComp = (args.nPSF - 1) * triNum(args.kernelOrder + 1) + 1;
  double invKernMult = scaleConv ? invKernSum : 1.0;

#pragma omp parallel for collapse(2) schedule(static) default(none)        \
    shared(w, h, halfConvWidth, convImg, kernSolution, mask, args, xSteps, \
               convKernels, templateImage, convMask, nBgComp, invKernMult)
  for(int y = 0; y < h; y++) {
    for(int x = 0; x < w; x++) {
      double acc = 0.0;
      int id = x + y * w;

      if(x >= halfConvWidth && x < w - halfConvWidth && y >= halfConvWidth &&
         y < h - halfConvWidth) {
        // convolve
        int xS = (x - halfConvWidth) / args.fKernelWidth;
        int yS = (y - halfConvWidth) / args.fKernelWidth;

        int convOffset =
            (xS + yS * xSteps) * args.fKernelWidth * args.fKernelWidth;

        int maskAcc = 0;
        double aks = 0.0;
        double uks = 0.0;

        for(int j = y - halfConvWidth; j <= y + halfConvWidth; j++) {
          int jk = y - j + halfConvWidth;
          for(int i = x - halfConvWidth; i <= x + halfConvWidth; i++) {
            int ik = x - i + halfConvWidth;
            int convIndex = ik + jk * args.fKernelWidth;
            convIndex += convOffset;
            int imgIndex = i + w * j;

            double kk = convKernels[convIndex];
            acc += kk * templateImage[imgIndex];
            maskAcc |= convMask.dataMask[imgIndex];
            aks += fabs(kk);

            if((convMask.dataMask[imgIndex] & ImageMasks::BAD_INPUT) == 0) {
              uks += fabs(kk);
            }
          }
        }

        acc += getBackground(x, y, kernSolution, w, h, args.backgroundOrder,
                             nBgComp);
        acc *= invKernMult;

        convImg.data[id] = acc;

        ushort newMask = convMask.dataMask[id];

        if((convMask.dataMask[id] & ImageMasks::BAD_INPUT) != 0) {
          newMask |= ImageMasks::BAD_OUTPUT;
        }

        if(maskAcc != 0) {
          if((uks / aks) < 0.99f) {
            newMask |= ImageMasks::BAD_OUTPUT | ImageMasks::BAD_CONV;
          } else {
            newMask |= ImageMasks::OK_CONV;
          }
        }

        mask.dataMask[id] = newMask;
      } else {
        // default value
        convImg.data[id] = 1e-30;
      }
    }
  }

  clData.queue.enqueueWriteBuffer(clData.convImg, CL_TRUE, 0,
                                  sizeof(cl_double) * w * h, &convImg.data[0]);

// mask after conv
#pragma omp parallel for collapse(2) schedule(static) default(none) \
    shared(w, h, mask, scienceImage, args)
  for(int y = 0; y < h; y++) {
    for(int x = 0; x < w; x++) {
      int id = x + y * w;

      double t = scienceImage[id];

      if(t == 0) {
        mask.maskPix(x, y,
                     ImageMasks::BAD_OUTPUT | ImageMasks::BAD_INPUT |
                         ImageMasks::BAD_PIX_VAL);
      }

      if(t >= args.threshHigh) {
        mask.maskPix(x, y,
                     ImageMasks::BAD_OUTPUT | ImageMasks::BAD_INPUT |
                         ImageMasks::SAT_PIXEL);
      }

      if(t <= args.threshLow) {
        mask.maskPix(x, y,
                     ImageMasks::BAD_OUTPUT | ImageMasks::BAD_INPUT |
                         ImageMasks::LOW_PIXEL);
      }
    }
  }

  clData.queue.enqueueWriteBuffer(clData.maskBuf, CL_TRUE, 0,
                                  sizeof(cl_ushort) * w * h, &mask.dataMask[0]);
}

// doesn't split masking bc they are already very fast (sorry for this
// monstrosity)
void convSplit(const int w, const int h,
               const std::vector<cl_double>& convKernels, const int xSteps,
               const bool scaleConv, const double invKernSum, Image& convImg,
               const Image& templateImage,
               const std::vector<double>& kernSolution, const Arguments& args,
               ClData& clData) {
  // Declare all the buffers which will be need in opencl operations.
  cl::Buffer convMaskBuf(clData.context, CL_MEM_READ_ONLY,
                         sizeof(uint16_t) * w * h);
  cl::Buffer kernBuf(clData.context, CL_MEM_READ_ONLY,
                     sizeof(cl_double) * convKernels.size());

  // Write necessary data for convolution
  clData.queue.enqueueWriteBuffer(kernBuf, CL_TRUE, 0,
                                  sizeof(cl_double) * convKernels.size(),
                                  convKernels.data());

  // Create convolution mask
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl_int, cl_double, cl_double>
      createMaskFunc(clData.program, "createConvMask");
  cl::EnqueueArgs createMaskEargs(clData.queue, cl::NDRange(w, h));
  cl::Event createMaskEvent =
      createMaskFunc(createMaskEargs, clData.tImgBuf, convMaskBuf, w,
                     args.threshHigh, args.threshLow);

  createMaskEvent.wait();

  // Convolve
  auto p1 = std::chrono::steady_clock::now();

  // gpu convolves the first gpuH lines of the image while the cpu convolves the
  // last cpuH lines
  int cpuH = h * args.cpuPart;
  int accSum = 0;
  std::vector<int> accH{};
  for(size_t i = 0; i < clData.accelerators.size(); i++) {
    int lines = clData.accelerators[i].workSplit * h;
    accH.emplace_back(lines);
    accSum += lines;
  }

  int gpuH = h - cpuH - accSum;

  if(args.verbose) {
    std::cout << "lines to be convolved on cpu:" << cpuH << " on gpu: " << gpuH
              << " total lines: " << h << std::endl
              << "width: " << w << std::endl;
    for(size_t i = 0; i < accH.size(); i++) {
      std::cout << "accelerator " << i << " will conv " << accH[i] << " lines"
                << std::endl;
    }
  }

  assert(gpuH + cpuH + accSum == h);

  // read convMask
  ImageMask convMask(w, h);
  clData.queue.enqueueReadBuffer(convMaskBuf, CL_TRUE, 0,
                                 sizeof(cl_ushort) * w * h,
                                 &convMask.dataMask[0]);

  ImageMask mask{w, h};
  clData.queue.enqueueReadBuffer(clData.maskBuf, CL_TRUE, 0,
                                 sizeof(cl_ushort) * w * h, &mask.dataMask[0]);

  // main gpu convolution
  if(args.verbose) {
    std::cout << "starting convolution on main gpu" << std::endl;
  }

  int nBgComp = (args.nPSF - 1) * triNum(args.kernelOrder + 1) + 1;
  cl::KernelFunctor<cl::Buffer, cl_int, cl_int, cl_int, cl::Buffer, cl::Buffer,
                    cl::Buffer, cl::Buffer, cl::Buffer, cl_int, cl_int, cl_int,
                    cl_int, cl_double>
      convFunc(clData.program, "conv");
  cl::EnqueueArgs eargs(clData.queue, cl::NDRange(w * gpuH));
  cl::Event convEvent = convFunc(
      eargs, kernBuf, args.fKernelWidth, xSteps, 0, clData.tImgBuf,
      clData.convImg, convMaskBuf, clData.maskBuf, clData.kernel.solution, w, h,
      args.backgroundOrder, nBgComp, scaleConv ? invKernSum : 1.0);

  // accelerator convolution
  auto bacc = std::chrono::steady_clock::now();
  if(args.verbose) {
    std::cout << "starting convolution on " << clData.accelerators.size()
              << " accelerators" << std::endl;
  }

  const int nComp2 = triNum(args.kernelOrder + 1);
  const int nBGComp = triNum(args.backgroundOrder + 1);
  const int nKernSolComp = args.nPSF * nComp2 + nBGComp + 1;

  std::vector<cl::Event> acceleratorEvents{};
  // conv, mask
  std::vector<std::pair<cl::Buffer, cl::Buffer>> outBuffers{};
  // in lines
  int offset = gpuH;
  for(size_t i = 0; i < clData.accelerators.size(); i++) {
    // create and write buffers to accelerator
    cl::Buffer accConvMaskBuf(clData.accelerators[i].context, CL_MEM_READ_WRITE,
                              sizeof(uint16_t) * w * h);
    clData.accelerators[i].queue.enqueueWriteBuffer(accConvMaskBuf, CL_TRUE, 0,
                                                    sizeof(uint16_t) * w * h,
                                                    &convMask.dataMask[0]);

    cl::Buffer maskBuf(clData.accelerators[i].context, CL_MEM_READ_WRITE,
                       sizeof(uint16_t) * w * h);
    clData.accelerators[i].queue.enqueueWriteBuffer(
        maskBuf, CL_TRUE, 0, sizeof(uint16_t) * w * h, &mask.dataMask[0]);

    cl::Buffer tImgBuf(clData.accelerators[i].context, CL_MEM_READ_WRITE,
                       sizeof(cl_double) * w * h);
    clData.accelerators[i].queue.enqueueWriteBuffer(
        tImgBuf, CL_TRUE, 0, sizeof(cl_double) * w * h, &templateImage.data[0]);

    cl::Buffer kernelSolutionBuf(clData.accelerators[i].context,
                                 CL_MEM_READ_WRITE,
                                 sizeof(cl_double) * nKernSolComp);
    clData.accelerators[i].queue.enqueueWriteBuffer(
        kernelSolutionBuf, CL_TRUE, 0, sizeof(cl_double) * nKernSolComp,
        &kernSolution[0]);

    cl::Buffer accKernBuf(clData.accelerators[i].context, CL_MEM_READ_WRITE,
                          sizeof(cl_double) * convKernels.size());
    clData.accelerators[i].queue.enqueueWriteBuffer(
        accKernBuf, CL_TRUE, 0, sizeof(cl_double) * convKernels.size(),
        convKernels.data());

    cl::Buffer accConvImg(clData.accelerators[i].context, CL_MEM_READ_WRITE,
                          sizeof(cl_double) * w * h);

    outBuffers.push_back(std::make_pair(accConvImg, maskBuf));

    // start kernel
    cl::KernelFunctor<cl::Buffer, cl_int, cl_int, cl_int, cl::Buffer,
                      cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl_int,
                      cl_int, cl_int, cl_int, cl_double>
        accConvFunc(clData.accelerators[i].program, "conv");
    cl::EnqueueArgs acceargs(clData.accelerators[i].queue,
                             cl::NDRange(w * accH[i]));
    cl::Event accEvent = accConvFunc(
        acceargs, accKernBuf, args.fKernelWidth, xSteps, offset, tImgBuf,
        accConvImg, accConvMaskBuf, maskBuf, kernelSolutionBuf, w, h,
        args.backgroundOrder, nBgComp, scaleConv ? invKernSum : 1.0);
    acceleratorEvents.push_back(accEvent);

    offset += accH[i];
  }
  auto pacc = std::chrono::steady_clock::now();
  std::cout << timeDiff(pacc, bacc) << " ns for starting acc" << std::endl;

  // cpu convolution
  if(args.verbose) {
    std::cout << "starting convolution on cpu" << std::endl;
  }
  int halfConvWidth = args.fKernelWidth / 2;
  double invKernMult = scaleConv ? invKernSum : 1.0;
  int cpuStartH = gpuH + accSum;

#pragma omp parallel for collapse(2) schedule(static) default(none)           \
    shared(w, h, cpuH, cpuStartH, halfConvWidth, convImg, kernSolution, mask, \
               args, xSteps, convKernels, templateImage, convMask, nBgComp,   \
               invKernMult)
  for(int offy = 0; offy < cpuH; offy++) {
    for(int x = 0; x < w; x++) {
      int y = offy + cpuStartH;
      double acc = 0.0;
      int id = x + y * w;

      if(x >= halfConvWidth && x < w - halfConvWidth && y >= halfConvWidth &&
         y < h - halfConvWidth) {
        // convolve
        int xS = (x - halfConvWidth) / args.fKernelWidth;
        int yS = (y - halfConvWidth) / args.fKernelWidth;

        int convOffset =
            (xS + yS * xSteps) * args.fKernelWidth * args.fKernelWidth;

        int maskAcc = 0;
        double aks = 0.0;
        double uks = 0.0;

        for(int j = y - halfConvWidth; j <= y + halfConvWidth; j++) {
          int jk = y - j + halfConvWidth;
          for(int i = x - halfConvWidth; i <= x + halfConvWidth; i++) {
            int ik = x - i + halfConvWidth;
            int convIndex = ik + jk * args.fKernelWidth;
            convIndex += convOffset;
            int imgIndex = i + w * j;

            double kk = convKernels[convIndex];
            acc += kk * templateImage[imgIndex];
            maskAcc |= convMask.dataMask[imgIndex];
            aks += fabs(kk);

            if((convMask.dataMask[imgIndex] & ImageMasks::BAD_INPUT) == 0) {
              uks += fabs(kk);
            }
          }
        }

        acc += getBackground(x, y, kernSolution, w, h, args.backgroundOrder,
                             nBgComp);
        acc *= invKernMult;

        convImg.data[id] = acc;

        uint16_t newMask = convMask.dataMask[id];

        if((convMask.dataMask[id] & ImageMasks::BAD_INPUT) != 0) {
          newMask |= ImageMasks::BAD_OUTPUT;
        }

        if(maskAcc != 0) {
          if((uks / aks) < 0.99f) {
            newMask |= ImageMasks::BAD_OUTPUT | ImageMasks::BAD_CONV;
          } else {
            newMask |= ImageMasks::OK_CONV;
          }
        }

        mask.dataMask[id] = newMask;
      } else {
        // default value
        convImg.data[id] = 1e-30;
      }
    }
  }

  auto bwait = std::chrono::steady_clock::now();
  convEvent.wait();
  cl::Event::waitForEvents(acceleratorEvents);
  auto pwait = std::chrono::steady_clock::now();
  std::cout << "waited for " << timeDiff(pwait, bwait) << " ns" << std::endl;

  // Transfer convoluted image back to CPU, needed for saving to file and copy
  // to main gpu
  clData.queue.enqueueReadBuffer(clData.convImg, CL_TRUE, 0,
                                 sizeof(cl_double) * gpuH * w,
                                 &convImg.data[0]);

  // in lines
  offset = gpuH;
  for(size_t i = 0; i < outBuffers.size(); i++) {
    cl::Buffer conv = outBuffers[i].first;
    cl::Buffer maskBuf = outBuffers[i].second;

    clData.accelerators[i].queue.enqueueReadBuffer(
        conv, CL_TRUE, sizeof(cl_double) * offset * w,
        sizeof(cl_double) * accH[i] * w, &convImg.data[offset * w]);

    clData.accelerators[i].queue.enqueueReadBuffer(
        maskBuf, CL_TRUE, sizeof(uint16_t) * offset * w,
        sizeof(uint16_t) * accH[i] * w, &mask.dataMask[offset * w]);

    offset += accH[i];
  }

  // #lines computed by both cpu and acc
  int accCpuH = cpuH + accSum;

  // put mask together on gpu
  clData.queue.enqueueWriteBuffer(
      clData.maskBuf, CL_TRUE, sizeof(cl_ushort) * gpuH * w,
      sizeof(cl_ushort) * accCpuH * w, &mask.dataMask[gpuH * w]);
  // make the convoluted image full on GPU, needed for subtraction
  clData.queue.enqueueWriteBuffer(
      clData.convImg, CL_TRUE, sizeof(cl_double) * gpuH * w,
      sizeof(cl_double) * accCpuH * w, &convImg.data[gpuH * w]);

  auto p2 = std::chrono::steady_clock::now();

  if(args.verboseTime) {
    std::cout << "Convolution (without masking) took " << timeDiff(p2, p1)
              << " ns " << std::endl;
  }
  // Mask after convolve
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl_int, cl_double, cl_double>
      maskAfterFunc(clData.program, "maskAfterConv");
  cl::EnqueueArgs maskAfterEargs(clData.queue, cl::NDRange(w, h));
  cl::Event maskAfterEvent =
      maskAfterFunc(maskAfterEargs, clData.sImgBuf, clData.maskBuf, w,
                    args.threshHigh, args.threshLow);

  maskAfterEvent.wait();
}