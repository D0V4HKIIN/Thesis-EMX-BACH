#include "bachUtil.h"
#include <fstream>
#include <iomanip>
#include <algorithm>

#define M1 259200
#define IA1 7141
#define IC1 54773
#define RM1 (1.0/M1)
#define M2 134456
#define IA2 8121
#define IC2 28411
#define RM2 (1.0/M2)
#define M3 243000
#define IA3 4561
#define IC3 51349
double ran1(int *idum) {
    static long ix1,ix2,ix3;
    static double r[98];
    double temp;
    static int iff=0;
    int j;
    /* void nrerror(char *error_text); */
    
    if (*idum < 0 || iff == 0) {
        iff=1;
        ix1=(IC1-(*idum)) % M1;
        ix1=(IA1*ix1+IC1) % M1;
        ix2=ix1 % M2;
        ix1=(IA1*ix1+IC1) % M1;
        ix3=ix1 % M3;
        for (j=1;j<=97;j++) {
            ix1=(IA1*ix1+IC1) % M1;
            ix2=(IA2*ix2+IC2) % M2;
            r[j]=(ix1+ix2*RM2)*RM1;
        }
        *idum=1;
    }
    ix1=(IA1*ix1+IC1) % M1;
    ix2=(IA2*ix2+IC2) % M2;
    ix3=(IA3*ix3+IC3) % M3;
    j=1 + ((97*ix3)/M3);
    /* if (j > 97 || j < 1) nrerror("RAN1: This cannot happen."); */
    temp=r[j];
    r[j]=(ix1+ix2*RM2)*RM1;
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
  double median, sum;

  std::vector<cl_int> bins(256, 0);

  constexpr cl_int nValues = 100;
  double upProc = 0.9;
  double midProc = 0.5;
  
  static constinit int statsSize{5};
  size_t statsCount{statsSize * stamps.size()};
  std::vector<cl_double> stampStats(statsCount);

  size_t statIdx{0};
  for (auto &&stamp : stamps)
  {
    cl_int numPix = stamp.size.first * stamp.size.second;

    if(numPix < nValues) {
      std::cout << "Not enough pixels in a stamp" << std::endl;
      exit(1);
    }
    int idum = -666;

    std::array<double, nValues> values{};
    int valuesCount = 0;

    // Stop after randomly having selected a pixel numPix times.
    for(int iter = 0; valuesCount < nValues && iter < numPix; iter++) {
      int randX = std::floor(ran1(&idum) * stamp.size.first);
      int randY = std::floor(ran1(&idum) * stamp.size.second);
      
      // Random pixel in stamp in stamp coords.
      cl_int indexS = randX + randY * stamp.size.first;

      // Random pixel in stamp in Image coords.
      cl_int xI = randX + stamp.coords.first;
      cl_int yI = randY + stamp.coords.second;
      int indexI = xI + yI * image.axis.first;

      if(mask.isMaskedAny(indexI) || std::abs(image[indexI]) <= 1e-10) {
        continue;
      }

      values[valuesCount++] = stamp[indexS];
    }

    std::sort(std::begin(values), std::end(values));

    // Width of a histogram bin.
    double binSize = (values[(int)(upProc * valuesCount)] -
                      values[(int)(midProc * valuesCount)]) /
                    (double)nValues;

    // Value of lowest bin.
    double lowerBinVal =
        values[(int)(midProc * valuesCount)] - (128.0 * binSize);

    // Contains all good Pixels in the stamp, aka not masked.
    std::vector<double> maskedStamp{};
    for(int y = 0; y < stamp.size.second; y++) {
      for(int x = 0; x < stamp.size.first; x++) {
        // Pixel in stamp in stamp coords.
        cl_int indexS = x + y * stamp.size.first;

        // Pixel in stamp in Image coords.
        cl_int xI = x + stamp.coords.first;
        cl_int yI = y + stamp.coords.second;
        int indexI = xI + yI * image.axis.first;

        if(mask.isMaskedAny(indexI) || image[indexI] <= 1e-10) {
          continue;
        }

        if (std::isnan(image[indexI])) {
          mask.maskPix(xI, yI, ImageMask::NAN_PIXEL | ImageMask::BAD_INPUT);
          continue;
        }

        maskedStamp.push_back(stamp[indexS]);
      }
    }

    

    // sigma clip of maskedStamp to get mean and sd.
    double mean, stdDev, invStdDev;
    sigmaClip(maskedStamp, mean, stdDev, 3, args); //TODO: Use parallel version later
    invStdDev = 1.0 / stdDev;
    
    double median;
    double sum;
    cl_int okCount = 0;
    int attempts = 0;
    double sumBins = 0.0;
    double sumExpect = 0.0;
    double lower, upper;
    while(true) {
      if(attempts >= 5) {
        std::cout <<"Stamp " << statIdx << ": Creation of histogram unsuccessful after 5 attempts" << std::endl;
        break;
      }

      std::fill(bins.begin(), bins.end(), 0);
      okCount = 0;
      sum = 0.0;
      sumBins = 0.0;
      sumExpect = 0.0;
      for(int y = 0; y < stamp.size.second; y++) {
        for(int x = 0; x < stamp.size.first; x++) {
          // Pixel in stamp in stamp coords.
          cl_int indexS = x + y * stamp.size.first;

          // Pixel in stamp in Image coords.
          cl_int xI = x + stamp.coords.first;
          cl_int yI = y + stamp.coords.second;
          int indexI = xI + yI * image.axis.first;

          if(mask.isMaskedAny(indexI) || image[indexI] <= 1e-10) {
            continue;
          }

          if((std::abs(stamp[indexS] - mean) * invStdDev) > args.sigClipAlpha) {
            continue;
          }
          
          int index = std::clamp(
              (int)std::floor((stamp[indexS] - lowerBinVal) / binSize) + 1, 0, 255);

          bins[index]++;
          sum += abs(stamp[indexS]);
          okCount++;
        }
      }

      if(okCount == 0 || binSize == 0.0) {
        std::cout << "No good pixels or variation in pixels" << std::endl;
        break;
      }

      double maxDens = 0.0;
      int lowerIndex, upperIndex, maxIndex = -1;
      for(lowerIndex = upperIndex = 1; upperIndex < 255;
          sumBins -= bins[lowerIndex++]) {
        while(sumBins < okCount / 10.0 && upperIndex < 255) {
          sumBins += bins[upperIndex++];
        }
        if(sumBins / (upperIndex - lowerIndex) > maxDens) {
          maxDens = sumBins / (upperIndex - lowerIndex);
          maxIndex = lowerIndex;
        }
      }
      if(maxIndex < 0 || maxIndex > 255) maxIndex = 0;

      sumBins = 0.0;
      for(int i = maxIndex; sumBins < okCount / 10.0 && i < 255; i++) {
        sumBins += bins[i];
        sumExpect += i * bins[i];
      }

      double modeBin = sumExpect / sumBins + 0.5;
      stamp.stats.skyEst = lowerBinVal + binSize * (modeBin - 1.0);
      stampStats[statsSize*statIdx + StatsIndeces::SKY_EST] = stamp.stats.skyEst;

      lower = okCount * 0.25;
      upper = okCount * 0.75;
      sumBins = 0.0;
      int i = 0;
      for(; sumBins < lower; sumBins += bins[i++])
        ;
      lower = i - (sumBins - lower) / bins[i - 1];
      for(; sumBins < upper; sumBins += bins[i++])
        ;
      upper = i - (sumBins - upper) / bins[i - 1];

      if(lower < 1.0 || upper > 255.0) {
        if(args.verbose) {
          std::cout << "Expanding bin size..." << std::endl;
        }
        lowerBinVal -= 128.0 * binSize;
        binSize *= 2;
        attempts++;
      } else if(upper - lower < 40.0) {
        if(args.verbose) {
          std::cout << "Shrinking bin size..." << std::endl;
        }
        binSize /= 3.0;
        lowerBinVal = stamp.stats.skyEst - 128.0 * binSize;
        attempts++;
      } else
        break;
    }
    
    if (okCount == 0 || binSize == 0.) {
      ++statIdx;
      continue;
    }
    stamp.stats.fwhm = binSize * (upper - lower) / args.iqRange;
    stampStats[statsSize*statIdx + StatsIndeces::FWHM] = stamp.stats.fwhm;
    int i = 0;
    for(i = 0, sumBins = 0; sumBins < okCount / 2.0; sumBins += bins[i++])
      ;
    median = i - (sumBins - okCount / 2.0) / bins[i - 1];
    median = lowerBinVal + binSize * (median - 1.0);
    
    ++statIdx;
  }
  cl_int err = clData.queue.enqueueWriteBuffer(stampsData.stampStats, CL_TRUE, 0, sizeof(cl_double) * statsCount, &stampStats[0]);
  checkError(err);
  err = clData.queue.enqueueWriteBuffer(clData.maskBuf, CL_TRUE, 0, sizeof(cl_ushort) * image.axis.first * image.axis.second, &mask);
  checkError(err);
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
