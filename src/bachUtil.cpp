#include "bachUtil.h"
#include "mathUtil.h"
#include <numeric>
#include <fstream>
#include <iomanip>
#include <algorithm>

void checkError(const cl_int err) {
  if(err != 0) {
    std::cout << "Error encountered with error code: " << err << std::endl;
    exit(err);
  }
}

void maskInput(ImageMask& mask, const ClData& clData, const Arguments& args) {
  cl::EnqueueArgs eargs{clData.queue, cl::NullRange, cl::NDRange(mask.axis.first * mask.axis.second), cl::NullRange};

  // Create mask from input data
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl_long, cl_long, cl_long, cl_double, cl_double>
      maskFunc(clData.program, "maskInput");
  cl::Event maskEvent = maskFunc(eargs, clData.tImgBuf, clData.sImgBuf, clData.maskBuf,
    mask.axis.first, mask.axis.second, args.hSStampWidth + args.hKernelWidth,
    args.threshHigh, args.threshLow);
  maskEvent.wait();

  // Spread mask
  cl::KernelFunctor<cl::Buffer, cl_long, cl_long, cl_long>
      spreadFunc(clData.program, "spreadMask");
  cl::Event spreadEvent = spreadFunc(eargs, clData.maskBuf,
    mask.axis.first, mask.axis.second,
    static_cast<int>(args.hKernelWidth * args.inSpreadMaskFactor));
  spreadEvent.wait();
  
  // For now, return the image mask to the CPU
  cl_int err = clData.queue.enqueueReadBuffer(clData.maskBuf, CL_TRUE, 0,
    sizeof(cl_ushort) * mask.axis.first * mask.axis.second, &mask);
  checkError(err);
}

void sigmaClip(const cl::Buffer &data, int dataCount, double *mean, double *stdDev, int maxIter, const ClData &clData, const Arguments& args) {
  if(dataCount == 0) {
    std::cout << "Cannot send in empty vector to Sigma Clip" << std::endl;
    *mean = 0.0;
    *stdDev = 1e10;
    return;
  }

  constexpr int localSize = 32;
  int reduceCount = (dataCount + localSize - 1) / localSize;

  std::vector<double> sumVec(reduceCount);
  std::vector<double> sum2Vec(reduceCount);

  cl::Buffer intMask(clData.context, CL_MEM_READ_WRITE, sizeof(cl_uchar) * dataCount);
  cl::Buffer clipCountBuf(clData.context, CL_MEM_READ_WRITE, sizeof(cl_int));
  cl::Buffer sumBuf(clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * reduceCount);
  cl::Buffer sum2Buf(clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * reduceCount);

  cl::KernelFunctor<cl::Buffer> initMaskFunc(clData.program, "sigmaClipInitMask");
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl_long> calcFunc(clData.program, "sigmaClipCalc");
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl_double, cl_double, cl_double> maskFunc(clData.program, "sigmaClipMask");
  
  cl::EnqueueArgs calcEargs(clData.queue, cl::NullRange, cl::NDRange(reduceCount * localSize), cl::NDRange(localSize));
  cl::EnqueueArgs eargs(clData.queue, cl::NullRange, cl::NDRange(dataCount), cl::NullRange);

  // Zero mask
  cl::Event initMaskEvent = initMaskFunc(eargs, intMask);
  initMaskEvent.wait();

  size_t currNPoints = 0;
  size_t prevNPoints = dataCount;

  // Do three times or a stable solution has been found.
  for (int i = 0; (i < maxIter) && (currNPoints != prevNPoints); i++) {
    if (prevNPoints <= 1) {
      std::cout << "prevNPoints is: " << prevNPoints << "Needs to be greater than 1" << std::endl;
      *mean = 0.0;
      *stdDev = 1e10;
      return;
    }

    currNPoints = prevNPoints;
        
    // Calculate mean and standard deviation    
    cl::Event calcEvent = calcFunc(calcEargs, sumBuf, sum2Buf, data, intMask, dataCount);
    calcEvent.wait();
    
    // Can be optimized to use a tree structure instead of reducing on CPU
    clData.queue.enqueueReadBuffer(sumBuf, CL_TRUE, 0, sizeof(cl_double) * sumVec.size(), sumVec.data());    
    clData.queue.enqueueReadBuffer(sum2Buf, CL_TRUE, 0, sizeof(cl_double) * sum2Vec.size(), sum2Vec.data());

    double sum = std::accumulate(sumVec.begin(), sumVec.end(), 0.0);
    double sum2 = std::accumulate(sum2Vec.begin(), sum2Vec.end(), 0.0);

    double tempMean = sum / prevNPoints;
    double tempStdDev = std::sqrt((sum2 - prevNPoints * tempMean * tempMean) / (prevNPoints - 1));
    
    double invStdDev = 1.0 / tempStdDev;

    int clipCount = 0;
    clData.queue.enqueueWriteBuffer(clipCountBuf, CL_TRUE, 0, sizeof(cl_int), &clipCount);

    // Mask bad values
    cl::Event maskEvent = maskFunc(eargs, intMask, clipCountBuf, data, invStdDev, tempMean, args.sigClipAlpha);
    maskEvent.wait();

    clData.queue.enqueueReadBuffer(clipCountBuf, CL_TRUE, 0, sizeof(cl_int), &clipCount);

    prevNPoints = currNPoints - clipCount;
    *mean = tempMean;
    *stdDev = tempStdDev;
  }
}

void sigmaClip(const std::vector<double>& data, double& mean, double& stdDev,
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

constexpr cl_int leastGreaterPow2(cl_int n) {
  if (n < 0) return 0;
  --n;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  return n+1;
}

void calcStats(std::vector<Stamp>& stamps, const Image& image, ImageMask& mask, const Arguments& args, const cl::Buffer& imgBuf, const ClStampsData& stampsData, const ClData& clData) {
  /* Heavily taken from HOTPANTS which itself copied it from Gary Bernstein
   * Calculates important values of stamps for futher calculations.
   */
  auto&& [imgW, imgH] = image.axis;
  clData.queue.enqueueWriteBuffer(clData.maskBuf, CL_TRUE, 0, sizeof(cl_ushort) * imgW * imgH, &mask);
  
  size_t nStamps{stamps.size()};

  static constexpr cl_int nSamples{100};
  static constexpr cl_int paddedNSamples{leastGreaterPow2(nSamples)};

  static constexpr double upProc = 0.9;
  static constexpr double midProc = 0.5;

  for (Stamp& stamp : stamps) {
    cl_int stampNumPix = stamp.size.first * stamp.size.second;

    if(stampNumPix < nSamples) {
      std::cout << "Not enough pixels in a stamp" << std::endl;
      exit(1);
    }
  }

  cl_int nPix{args.fStampWidth * args.fStampWidth};
  cl::Buffer samples{clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * nSamples * nStamps};
  cl::Buffer paddedSamples{clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * paddedNSamples * nStamps};
  cl::Buffer sampleCounts{clData.context, CL_MEM_READ_WRITE, sizeof(cl_int) * nStamps};
  
  cl::Buffer goodPixels{clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * nPix * nStamps};
  cl::Buffer goodPixelCounts{clData.context, CL_MEM_READ_WRITE, sizeof(cl_int) * nStamps};

  cl::Buffer bins{clData.context, CL_MEM_READ_WRITE, sizeof(cl_int) * 256 * nStamps};
  cl::Buffer means{clData.context, CL_MEM_READ_ONLY, sizeof(cl_double) * nStamps};
  cl::Buffer invStdDevs{clData.context, CL_MEM_READ_ONLY, sizeof(cl_double) * nStamps};
  cl::Buffer binSizes{clData.context, CL_MEM_READ_ONLY, sizeof(cl_double) * nStamps};
  cl::Buffer lowerBinVals{clData.context, CL_MEM_READ_ONLY, sizeof(cl_double) * nStamps};

  cl::EnqueueArgs eargsSample{clData.queue, cl::NDRange{nStamps}};
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl_long, cl_int>
  sampleStampFunc(clData.program, "sampleStamp");
  
  cl::EnqueueArgs eargsPadSamples{clData.queue, cl::NDRange{paddedNSamples, nStamps}};
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl_int, cl_int>
  padFunc(clData.program, "pad");

  cl::EnqueueArgs eargsSortSamples{clData.queue, cl::NDRange(paddedNSamples * nStamps)};
  cl::KernelFunctor<cl::Buffer, cl_int, cl_int, cl_int>
  sortSamplesFunc(clData.program, "sortSamples");

  cl::EnqueueArgs eargsMask{clData.queue, cl::NDRange(nPix, nStamps)};
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl_int, cl_int, cl_int>
  maskFunc(clData.program, "maskStamp");

  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                    cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                    cl::Buffer, cl::Buffer, cl::Buffer,
                    cl_int, cl_int, cl_double, cl_double>
  histogramFunc(clData.program, "createHistogram");
  
  
  std::vector<cl_double> cpuSamples(paddedNSamples * nStamps);
  std::vector<cl_int>    cpuSampleCounts(nStamps);

  std::vector<cl_double> cpuGoodPixels(nPix * nStamps);
  std::vector<cl_int>    cpuGoodPixelCounts(nStamps, 0);
  
  std::vector<cl_double> cpuMeans{};
  std::vector<cl_double> cpuInvStdDevs{};
  std::vector<cl_double> cpuBinSizes{};
  std::vector<cl_double> cpuLowerBinVals{};

  clData.queue.enqueueWriteBuffer(sampleCounts, CL_TRUE, 0, sizeof(cl_int) * cpuSampleCounts.size(), &cpuSampleCounts[0]);
  clData.queue.enqueueWriteBuffer(goodPixelCounts, CL_TRUE, 0, sizeof(cl_int) * cpuGoodPixelCounts.size(), &cpuGoodPixelCounts[0]);

  cl::Event sampleEvent =
    sampleStampFunc(eargsSample, imgBuf, clData.maskBuf,
                    stampsData.stampCoords, stampsData.stampSizes,
                    samples, sampleCounts, imgW, nSamples);
  sampleEvent.wait();

  cl::Event padEvent =
    padFunc(eargsPadSamples, samples, paddedSamples, 
            nSamples, paddedNSamples);
  padEvent.wait();

  cl::Event sortEvent;
  for (cl_int k=2;k<=paddedNSamples;k=2*k) // Outer loop, double size for each step
  {
    for (cl_int j=k>>1;j>0;j=j>>1)  // Inner loop, half size for each step
    {
      sortEvent = sortSamplesFunc(eargsSortSamples, paddedSamples, paddedNSamples, j, k);
      cl_int err = sortEvent.wait();
      checkError(err);
    }
  }

  cl::Event maskEvent =
    maskFunc(eargsMask, imgBuf, clData.maskBuf,
             stampsData.stampCoords,
             goodPixels, goodPixelCounts,
             args.fStampWidth, imgW, imgH);
  maskEvent.wait();
  

  clData.queue.enqueueReadBuffer(sampleCounts, CL_TRUE, 0, sizeof(cl_int) * cpuSampleCounts.size(), &cpuSampleCounts[0]);
  clData.queue.enqueueReadBuffer(paddedSamples, CL_TRUE, 0, sizeof(cl_double) * cpuSamples.size(), &cpuSamples[0]);
  
  clData.queue.enqueueReadBuffer(goodPixelCounts, CL_TRUE, 0, sizeof(cl_int) * cpuGoodPixelCounts.size(), &cpuGoodPixelCounts[0]);
  clData.queue.enqueueReadBuffer(goodPixels, CL_TRUE, 0, sizeof(cl_double) * cpuGoodPixels.size(), &cpuGoodPixels[0]);
  
  clData.queue.enqueueReadBuffer(clData.maskBuf, CL_TRUE, 0, sizeof(cl_ushort) * imgW * imgH, &mask);

  for (size_t stampIdx{0}; stampIdx < nStamps; stampIdx++)
  {
    auto &&stamp{stamps[stampIdx]};
    cl_int stampNumPix = stamp.size.first * stamp.size.second;
    int sampleCount{cpuSampleCounts[stampIdx]};
    int goodPixelCount{cpuGoodPixelCounts[stampIdx]};

    // Width of a histogram bin.
    double binSize = (cpuSamples[stampIdx * paddedNSamples + (int)(upProc * sampleCount)] -
                      cpuSamples[stampIdx * paddedNSamples + (int)(midProc * sampleCount)]) /
                    (double)nSamples;

    // Value of lowest bin.
    double lowerBinVal =
        cpuSamples[stampIdx * paddedNSamples + (int)(midProc * sampleCount)] - (128.0 * binSize);

    // Contains all good Pixels in the stamp, aka not masked.
    std::vector<double> maskedStamp(goodPixelCount);
    std::copy(std::begin(cpuGoodPixels)+stampIdx*nPix,
              std::begin(cpuGoodPixels)+(stampIdx*nPix+goodPixelCount),
              std::begin(maskedStamp));

    // sigma clip of maskedStamp to get mean and sd.  
    double mean, stdDev, invStdDev;
    sigmaClip(maskedStamp, mean, stdDev, 3, args); //TODO: Use parallel version later
    invStdDev = 1.0 / stdDev;
    
    cpuMeans.push_back(mean);
    cpuInvStdDevs.push_back(invStdDev);
    cpuBinSizes.push_back(binSize);
    cpuLowerBinVals.push_back(lowerBinVal);
  }
  
  clData.queue.enqueueWriteBuffer(means, CL_TRUE, 0, sizeof(cl_double) * nStamps, &cpuMeans[0]);
  clData.queue.enqueueWriteBuffer(invStdDevs, CL_TRUE, 0, sizeof(cl_double) * nStamps, &cpuInvStdDevs[0]);
  clData.queue.enqueueWriteBuffer(binSizes, CL_TRUE, 0, sizeof(cl_double) * nStamps, &cpuBinSizes[0]);
  clData.queue.enqueueWriteBuffer(lowerBinVals, CL_TRUE, 0, sizeof(cl_double) * nStamps, &cpuLowerBinVals[0]);

  static constexpr int histogramLocalSize = 4;
  cl::EnqueueArgs histogramEargs(clData.queue, cl::NDRange(roundUpToMultiple(nStamps, histogramLocalSize)), cl::NDRange(histogramLocalSize));
  
  cl::Event histogramEvent = histogramFunc(histogramEargs, imgBuf, clData.maskBuf, stampsData.stampCoords, stampsData.stampSizes,
                                            means, invStdDevs, binSizes, lowerBinVals,
                                            bins, stampsData.stats.fwhms, stampsData.stats.skyEsts,
                                            image.axis.first, nStamps, args.iqRange, args.sigClipAlpha);
  
  histogramEvent.wait();

  std::vector<double> skyEsts(nStamps);
  std::vector<double> fwhms(nStamps);

  clData.queue.enqueueReadBuffer(stampsData.stats.skyEsts, CL_TRUE, 0, sizeof(cl_double) * nStamps, &skyEsts[0]);
  clData.queue.enqueueReadBuffer(stampsData.stats.fwhms, CL_TRUE, 0,sizeof(cl_double) * nStamps, &fwhms[0]);
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

double makeKernel(Kernel& kern, const std::pair<cl_long, cl_long> imgSize, const int x,
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

  for(int i = 0; i < args.fKernelWidth * args.fKernelWidth; i++) {
    kern.currKernel[i] = 0.0;
  }

  double sumKernel = 0.0;
  for(int i = 0; i < args.fKernelWidth * args.fKernelWidth; i++) {
    for(int psf = 0; psf < args.nPSF; psf++) {
      kern.currKernel[i] += kernCoeffs[psf] * kern.kernVec[psf][i];
    }
    sumKernel += kern.currKernel[i];
  }

  return sumKernel;
}
