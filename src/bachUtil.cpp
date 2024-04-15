#include "bachUtil.h"
#include "mathUtil.h"
#include <numeric>
#include <algorithm>

void maskInput(const std::pair<cl_long, cl_long> &axis, const ClData& clData, const Arguments& args) {
  // Create mask from input data
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl_int, cl_int, cl_int, cl_double, cl_double> maskFunc(clData.program, "maskInput");
  cl::EnqueueArgs maskEargs(clData.queue, cl::NDRange(axis.first * axis.second));
  cl::Event maskEvent = maskFunc(maskEargs, clData.tImgBuf, clData.sImgBuf, clData.maskBuf,
                                 axis.first, axis.second, args.hSStampWidth + args.hKernelWidth,
                                 args.threshHigh, args.threshLow);
  maskEvent.wait();

  // Spread mask
  int spreadWidth = static_cast<int>(args.hKernelWidth * args.inSpreadMaskFactor);
  cl::KernelFunctor<cl::Buffer, cl_int, cl_int, cl_int> spreadFunc(clData.program, "spreadMask");
  cl::EnqueueArgs spreadEargs(clData.queue, cl::NDRange(axis.first, axis.second));
  cl::Event spreadEvent = spreadFunc(spreadEargs, clData.maskBuf,
                                     axis.first, axis.second,
                                     spreadWidth);
  spreadEvent.wait();
}

void sigmaClip(const cl::Buffer &data, int dataOffset, int dataCount, double *mean, double *stdDev, int maxIter, const ClData &clData, const Arguments& args) {
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
  
  cl::EnqueueArgs calcEargs(clData.queue, cl::NDRange(dataOffset), cl::NDRange(reduceCount * localSize), cl::NDRange(localSize));
  cl::EnqueueArgs maskEargs(clData.queue, cl::NDRange(dataOffset), cl::NDRange(dataCount), cl::NullRange);

  // Zero mask
  cl::Event initMaskEvent = initMaskFunc(maskEargs, intMask);
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
    cl::Event maskEvent = maskFunc(maskEargs, intMask, clipCountBuf, data, invStdDev, tempMean, args.sigClipAlpha);
    maskEvent.wait();

    clData.queue.enqueueReadBuffer(clipCountBuf, CL_TRUE, 0, sizeof(cl_int), &clipCount);

    prevNPoints = currNPoints - clipCount;
    *mean = tempMean;
    *stdDev = tempStdDev;
  }
}

void calcStats(const std::pair<cl_long, cl_long> &axis, const Arguments& args, const cl::Buffer& imgBuf, const ClStampsData& stampsData, const ClData& clData) {
  /* Heavily taken from HOTPANTS which itself copied it from Gary Bernstein
   * Calculates important values of stamps for futher calculations.
   */
  auto&& [imgW, imgH] = axis;
  
  cl::size_type nStamps{static_cast<cl::size_type>(args.stampsx * args.stampsy)};

  static constexpr cl_int nSamples{100};
  static constexpr cl_int paddedNSamples{leastGreaterPow2(nSamples)};

  {
    std::vector<cl_long2> stampSizes(nStamps);
    clData.queue.enqueueReadBuffer(stampsData.stampSizes, CL_TRUE, 0, sizeof(cl_long2) * stampsData.stampCount, &stampSizes[0]);
    for (size_t i{0}; i < stampsData.stampCount; i++) {
      cl_int stampNumPix = stampSizes[i].x * stampSizes[i].y;
      if(stampNumPix < nSamples) {
        std::cout << "Not enough pixels in a stamp" << std::endl;
        std::exit(1);
      }
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

  cl::EnqueueArgs eargsResetGoodPixelCounts{clData.queue, cl::NDRange{nStamps}};
  cl::KernelFunctor<cl::Buffer>
  resetGoodPixelCountsFunc(clData.program, "resetGoodPixelCounts");

  cl::EnqueueArgs eargsMask{clData.queue, cl::NDRange(nPix, nStamps)};
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl_int, cl_int, cl_int>
  maskFunc(clData.program, "maskStamp");

  static constexpr int histogramLocalSize = 4;
  cl::EnqueueArgs eargsHistogram(clData.queue, cl::NDRange(roundUpToMultiple(nStamps, histogramLocalSize)), cl::NDRange(histogramLocalSize));
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                    cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                    cl::Buffer, cl::Buffer, cl::Buffer,
                    cl_int, cl_int, cl_int, cl_int,cl_double, cl_double>
  histogramFunc(clData.program, "createHistogram");
  
  cl::Event sampleEvent =
    sampleStampFunc(eargsSample, imgBuf, clData.maskBuf,
                    stampsData.stampCoords, stampsData.stampSizes,
                    samples, sampleCounts, imgW, nSamples);
  sampleEvent.wait();

  cl::Event resetEvent = 
    resetGoodPixelCountsFunc(eargsResetGoodPixelCounts, goodPixelCounts);
  resetEvent.wait();

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
      sortEvent.wait();
    }
  }

  cl::Event maskEvent =
    maskFunc(eargsMask, imgBuf, clData.maskBuf,
             stampsData.stampCoords,
             goodPixels, goodPixelCounts,
             args.fStampWidth, imgW, imgH);
  maskEvent.wait();
  
  std::vector<cl_int>    cpuGoodPixelCounts(nStamps);  
  std::vector<cl_double> cpuMeans(nStamps);
  std::vector<cl_double> cpuInvStdDevs(nStamps);

  clData.queue.enqueueReadBuffer(goodPixelCounts, CL_TRUE, 0, sizeof(cl_int) * cpuGoodPixelCounts.size(), &cpuGoodPixelCounts[0]);
  
  for (size_t stampIdx{0}; stampIdx < nStamps; stampIdx++)
  {
    int goodPixelCount{cpuGoodPixelCounts[stampIdx]};

    // sigma clip of maskedStamp to get mean and sd.  
    double mean, stdDev, invStdDev;
    sigmaClip(goodPixels, stampIdx*nPix, goodPixelCount, &mean, &stdDev, 3, clData, args);
    invStdDev = 1.0 / stdDev;
    
    cpuMeans[stampIdx] = mean;
    cpuInvStdDevs[stampIdx] = invStdDev;
  }
  
  clData.queue.enqueueWriteBuffer(means, CL_TRUE, 0, sizeof(cl_double) * nStamps, &cpuMeans[0]);
  clData.queue.enqueueWriteBuffer(invStdDevs, CL_TRUE, 0, sizeof(cl_double) * nStamps, &cpuInvStdDevs[0]);

  cl::Event histogramEvent =
    histogramFunc(eargsHistogram, imgBuf, clData.maskBuf,
                  stampsData.stampCoords, stampsData.stampSizes,
                  means, invStdDevs, paddedSamples, sampleCounts,
                  bins, stampsData.stats.fwhms, stampsData.stats.skyEsts,
                  axis.first, nStamps, nSamples, paddedNSamples,
                  args.iqRange, args.sigClipAlpha);

  histogramEvent.wait();
}

void ludcmp(const cl::Buffer &matrix, int matrixSize, int stampCount, const cl::Buffer &index, const cl::Buffer &vv, const ClData &clData) {
  // Find big values
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl_int> bigFunc(clData.program, "ludcmpBig");
  cl::EnqueueArgs bigEargs(clData.queue, cl::NDRange(matrixSize, stampCount));
  cl::Event bigEvent = bigFunc(bigEargs, matrix, vv, matrixSize);

  bigEvent.wait();

  // Rest of LU-decomposition
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl_int> restFunc(clData.program, "ludcmpRest");
  cl::EnqueueArgs restEargs(clData.queue, cl::NDRange(stampCount));
  cl::Event restEvent = restFunc(restEargs, vv, matrix, index, matrixSize);

  restEvent.wait();
}

void lubksb(const cl::Buffer &matrix, int matrixSize, int stampCount, const cl::Buffer &index, const cl::Buffer &result, const ClData &clData) {
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl_int> func(clData.program, "lubksb");
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

double makeKernel(const cl::Buffer &kernel, const cl::Buffer &kernSolution, const std::pair<cl_long, cl_long> &imgSize, const int x, const int y, const Arguments& args, const ClData &clData) {
  double hWidth = 0.5 * imgSize.first;
  double hHeight = 0.5 * imgSize.second;

  double xf = (x - hWidth) / hWidth;
  double yf = (y - hHeight) / hHeight;

  static constexpr int localCount = 32;

  // Create buffers
  cl::Buffer kernCoeffs(clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * args.nPSF);
  cl::Buffer kernelSum(clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * args.fKernelWidth * args.fKernelWidth);
  cl::Buffer kernelSum2(clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * ((args.fKernelWidth * args.fKernelWidth + localCount - 1) / localCount));

  // Create coefficients
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl_int, cl_int,
                    cl_double, cl_double> coeffFunc(clData.program, "makeKernelCoeffs");
  cl::EnqueueArgs coeffEargs(clData.queue, cl::NDRange(args.nPSF));
  cl::Event coeffEvent = coeffFunc(coeffEargs, kernSolution, kernCoeffs, args.kernelOrder,
                                   triNum(args.kernelOrder + 1), xf, yf);

  coeffEvent.wait();

  // Create kernel
  static constexpr int kernelLocalSize = 16;
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::LocalSpaceArg, cl_int, cl_int> kernelFunc(clData.program, "makeKernel");
  cl::EnqueueArgs kernelEargs(clData.queue, cl::NDRange(roundUpToMultiple(args.fKernelWidth * args.fKernelWidth, kernelLocalSize)), cl::NDRange(kernelLocalSize));
  cl::Event kernelEvent = kernelFunc(kernelEargs, kernCoeffs, clData.kernel.vec, kernel, cl::Local(kernelLocalSize * sizeof(cl_double)), args.nPSF, args.fKernelWidth);
  
  kernelEvent.wait();

  // Sum kernel
  cl::Event copyEvent{};
  clData.queue.enqueueCopyBuffer(kernel, kernelSum, 0, 0, sizeof(cl_double) * args.fKernelWidth * args.fKernelWidth, nullptr, &copyEvent);

  copyEvent.wait();

  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::LocalSpaceArg, cl_int> sumFunc(clData.program, "sumKernel");
  int sumCount = args.fKernelWidth * args.fKernelWidth;

  cl::Buffer* src = &kernelSum;
  cl::Buffer* dst = &kernelSum2;

  while (sumCount > 1) {
    cl::EnqueueArgs sumEargs(clData.queue, cl::NDRange(roundUpToMultiple(sumCount, localCount)), cl::NDRange(localCount));
    cl::Event sumEvent = sumFunc(sumEargs, *src, *dst, cl::Local(localCount * sizeof(cl_double)), sumCount);
    
    sumEvent.wait();

    sumCount = (sumCount + localCount - 1) / localCount;
    std::swap(src, dst);
  }

  // Transfer sum to CPU
  cl_double sumKernel = 0.0;
  clData.queue.enqueueReadBuffer(*src, CL_TRUE, 0, sizeof(cl_double), &sumKernel);

  return sumKernel;
}

double makeKernel(Kernel& kern, const std::pair<cl_long, cl_long> &imgSize, const int x,
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
