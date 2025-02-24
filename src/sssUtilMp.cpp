#include <cassert>

#include "bachUtil.h"

void identifySStampsMp(std::vector<StampMp>& templStamps,
                       const Image& templImage,
                       std::vector<StampMp>& scienceStamps,
                       const Image& scienceImage, ImageMask& mask,
                       double* filledTempl, double* filledScience,
                       const Arguments& args) {
  std::cout << "Identifying sub-stamps in " << templImage.name << " and "
            << scienceImage.name << "..." << std::endl;

  assert(templStamps.size() == scienceStamps.size());

  for(int i = 0; i < templStamps.size(); i++) {
    calcStatsMp(templStamps[i], templImage, mask, args);
    calcStatsMp(scienceStamps[i], scienceImage, mask, args);

    findSStampsMp(templStamps[i], templImage, mask, i, true, args);
    findSStampsMp(scienceStamps[i], scienceImage, mask, i, false, args);
  }

  int oldCount = templStamps.size();

  templStamps.erase(
      std::remove_if(templStamps.begin(), templStamps.end(),
                     [](StampMp& s) { return s.subStamps.empty(); }),
      templStamps.end());
  scienceStamps.erase(
      std::remove_if(scienceStamps.begin(), scienceStamps.end(),
                     [](StampMp& s) { return s.subStamps.empty(); }),
      scienceStamps.end());

  if(filledTempl != nullptr) {
    *filledTempl = static_cast<double>(templStamps.size()) / oldCount;
  }

  if(filledScience != nullptr) {
    *filledScience = static_cast<double>(scienceStamps.size()) / oldCount;
  }

  if(args.verbose) {
    std::cout << "Non-Empty template stamps: " << templStamps.size()
              << std::endl;
    std::cout << "Non-Empty science stamps: " << scienceStamps.size()
              << std::endl;
  }
}

void createStampsMp(const Image& img, std::vector<StampMp>& stamps, const int w,
                    const int h, const Arguments& args) {
  for(int j = 0; j < args.stampsy; j++) {
    for(int i = 0; i < args.stampsx; i++) {
      int startx = i * (double(w) / double(args.stampsx));
      int starty = j * (double(h) / double(args.stampsy));
      int stopx = std::min(startx + args.fStampWidth, w);
      int stopy = std::min(starty + args.fStampWidth, h);
      int stampw = stopx - startx;
      int stamph = stopy - starty;

      int centerx = startx + stampw / 2;
      int centery = starty + stamph / 2;

      StampMp tmpS{};
      for(int y = 0; y < stamph; y++) {
        for(int x = 0; x < stampw; x++) {
          double tmp = img[(startx + x) + ((starty + y) * w)];
          tmpS.data.push_back(tmp);
        }
      }

      tmpS.coords = std::make_pair(startx, starty);
      tmpS.size = std::make_pair(stampw, stamph);
      tmpS.center = std::make_pair(centerx, centery);
      stamps.push_back(tmpS);
    }
  }
}

double checkSStampMp(const SubStampMp& sstamp, const Image& image,
                     ImageMask& mask, const StampMp& stamp,
                     const ImageMask::masks badMask, const bool isTemplate,
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
        mask.maskPix(
            x, y, isTemplate ? ImageMask::BAD_PIXEL_T : ImageMask::BAD_PIXEL_S);
        return 0.0;
      }
      if((image[absCoords] - stamp.stats.skyEst) / stamp.stats.fwhm >
         args.threshKernFit)
        retVal += image[absCoords];
    }
  }
  return retVal;
}

cl_int findSStampsMp(StampMp& stamp, const Image& image, ImageMask& mask,
                     const int index, const bool isTemplate,
                     const Arguments& args) {
  double floor = stamp.stats.skyEst + args.threshKernFit * stamp.stats.fwhm;

  double dfrac = 0.9;
  int maxSStamps = 2 * args.maxKSStamps;

  ImageMask::masks badMask = ImageMask::ALL & ~ImageMask::OK_CONV;

  if(isTemplate) {
    badMask &= ~(ImageMask::BAD_PIXEL_S | ImageMask::SKIP_S);
  } else {
    badMask &= ~(ImageMask::BAD_PIXEL_T | ImageMask::SKIP_T);
  }

  while(stamp.subStamps.size() < size_t(maxSStamps)) {
    double lowestPSFLim =
        std::max(floor, stamp.stats.skyEst +
                            (args.threshHigh - stamp.stats.skyEst) * dfrac);
    for(long y = 0; y < args.fStampWidth; y++) {
      long absy = y + stamp.coords.second;
      for(long x = 0; x < args.fStampWidth; x++) {
        long absx = x + stamp.coords.first;
        long coords = x + (y * stamp.size.first);
        long absCoords = absx + (absy * image.axis.first);

        if(mask.isMasked(absCoords, badMask)) {
          continue;
        }

        if(stamp[coords] > args.threshHigh) {
          mask.maskPix(
              absx, absy,
              isTemplate ? ImageMask::BAD_PIXEL_T : ImageMask::BAD_PIXEL_S);
          continue;
        }

        if((stamp[coords] - stamp.stats.skyEst) * (1.0 / stamp.stats.fwhm) <
           args.threshKernFit) {
          continue;
        }

        if(stamp[coords] > lowestPSFLim) {  // good candidate found
          SubStampMp s{{},
                       0.0,
                       std::make_pair(absx, absy),
                       std::make_pair(x, y),
                       stamp[coords]};

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
                mask.maskPix(kx, ky,
                             isTemplate ? ImageMask::BAD_PIXEL_T
                                        : ImageMask::BAD_PIXEL_S);
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
                s.stampCoords = std::make_pair(kx - stamp.coords.first,
                                               ky - stamp.coords.second);
              }
            }
          }
          s.val =
              checkSStampMp(s, image, mask, stamp, badMask, isTemplate, args);
          if(s.val == 0.0) continue;
          stamp.subStamps.push_back(s);

          for(int y = s.stampCoords.second - args.hSStampWidth;
              y <= s.stampCoords.second + args.hSStampWidth; y++) {
            int y2 = y + stamp.coords.second;
            for(int x = s.stampCoords.first - args.hSStampWidth;
                x <= s.stampCoords.first + args.hSStampWidth; x++) {
              int x2 = x + stamp.coords.first;
              if(x > 0 && x < stamp.size.first && y > 0 &&
                 y < stamp.size.second) {
                mask.maskPix(
                    x2, y2, isTemplate ? ImageMask::SKIP_T : ImageMask::SKIP_S);
              }
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
      std::cout << "No suitable substamps found in stamp " << index
                << std::endl;
    return 1;
  }
  int keepSStampCount = std::min<int>(stamp.subStamps.size(), args.maxKSStamps);
  std::partial_sort(stamp.subStamps.begin(),
                    stamp.subStamps.begin() + keepSStampCount,
                    stamp.subStamps.end(), std::greater<SubStampMp>());

  if(stamp.subStamps.size() > keepSStampCount) {
    stamp.subStamps.erase(stamp.subStamps.begin() + keepSStampCount,
                          stamp.subStamps.end());
  }

  if(args.verbose)
    std::cout << "Added " << stamp.subStamps.size() << " substamps to stamp "
              << index << std::endl;
  return 0;
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
double ran1(int* idum) {
  static long ix1, ix2, ix3;
  static double r[98];
  double temp;
  static int iff = 0;
  int j;
  /* void nrerror(char *error_text); */

  if(*idum < 0 || iff == 0) {
    iff = 1;
    ix1 = (IC1 - (*idum)) % M1;
    ix1 = (IA1 * ix1 + IC1) % M1;
    ix2 = ix1 % M2;
    ix1 = (IA1 * ix1 + IC1) % M1;
    ix3 = ix1 % M3;
    for(j = 1; j <= 97; j++) {
      ix1 = (IA1 * ix1 + IC1) % M1;
      ix2 = (IA2 * ix2 + IC2) % M2;
      r[j] = (ix1 + ix2 * RM2) * RM1;
    }
    *idum = 1;
  }
  ix1 = (IA1 * ix1 + IC1) % M1;
  ix2 = (IA2 * ix2 + IC2) % M2;
  ix3 = (IA3 * ix3 + IC3) % M3;
  j = 1 + ((97 * ix3) / M3);
  /* if (j > 97 || j < 1) nrerror("RAN1: This cannot happen."); */
  temp = r[j];
  r[j] = (ix1 + ix2 * RM2) * RM1;
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

void calcStatsMp(StampMp& stamp, const Image& image, ImageMask& mask,
                 const Arguments& args) {
  /* Heavily taken from HOTPANTS which itself copied it from Gary Bernstein
   * Calculates important values of stamps for futher calculations.
   */

  double median, sum;

  std::vector<cl_int> bins(256, 0);

  constexpr cl_int nValues = 100;
  double upProc = 0.9;
  double midProc = 0.5;
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
  double lowerBinVal = values[(int)(midProc * valuesCount)] - (128.0 * binSize);

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

      if(std::isnan(image[indexI])) {
        mask.maskPix(xI, yI, ImageMask::NAN_PIXEL | ImageMask::BAD_INPUT);
        continue;
      }

      maskedStamp.push_back(stamp[indexS]);
    }
  }

  // sigma clip of maskedStamp to get mean and sd.
  double mean, stdDev, invStdDev;
  sigmaClipMp(maskedStamp, mean, stdDev, 3, args);
  invStdDev = 1.0 / stdDev;

  int attempts = 0;
  cl_int okCount = 0;
  double sumBins = 0.0;
  double sumExpect = 0.0;
  double lower, upper;
  while(true) {
    if(attempts >= 5) {
      std::cout << "Creation of histogram unsuccessful after 5 attempts";
      return;
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
            (int)std::floor((stamp[indexS] - lowerBinVal) / binSize) + 1, 0,
            255);

        bins[index]++;
        sum += abs(stamp[indexS]);
        okCount++;
      }
    }

    if(okCount == 0 || binSize == 0.0) {
      std::cout << "No good pixels or variation in pixels" << std::endl;
      return;
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

    lower = okCount * 0.25;
    upper = okCount * 0.75;
    sumBins = 0.0;
    int i = 0;
    for(; sumBins < lower; sumBins += bins[i++]);
    lower = i - (sumBins - lower) / bins[i - 1];
    for(; sumBins < upper; sumBins += bins[i++]);
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
  stamp.stats.fwhm = binSize * (upper - lower) / args.iqRange;
  int i = 0;
  for(i = 0, sumBins = 0; sumBins < okCount / 2.0; sumBins += bins[i++]);
  median = i - (sumBins - okCount / 2.0) / bins[i - 1];
  median = lowerBinVal + binSize * (median - 1.0);
}