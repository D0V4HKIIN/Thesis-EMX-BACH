#define CL_HPP_TARGET_OPENCL_VERSION 300

#include "utils/bachUtil.h"
#include <cassert>

void identifySStamps(std::vector<Stamp>& templStamps, Image& templImage, std::vector<Stamp>& scienceStamps, Image& scienceImage, double* filledTempl, double* filledScience) {
  std::cout << "Identifying sub-stamps in " << templImage.name << " and " << scienceImage.name << "..." << std::endl;

  assert(templStamps.size() == scienceStamps.size());

  for (int i = 0; i < templStamps.size(); i++) {
    calcStats(templStamps[i], templImage);
    calcStats(scienceStamps[i], scienceImage);

    findSStamps(templStamps[i], templImage, i);
    findSStamps(scienceStamps[i], scienceImage, i);
  }

  int oldCount = templStamps.size();

  templStamps.erase(std::remove_if(templStamps.begin(), templStamps.end(),
                              [](Stamp& s) { return s.subStamps.empty(); }),
               templStamps.end());
  scienceStamps.erase(std::remove_if(scienceStamps.begin(), scienceStamps.end(),
                              [](Stamp& s) { return s.subStamps.empty(); }),
               scienceStamps.end());

  if (filledTempl != nullptr) {
    *filledTempl = static_cast<double>(templStamps.size()) / oldCount;
  }

  if (filledScience != nullptr) {
    *filledScience = static_cast<double>(scienceStamps.size()) / oldCount;
  }

  if(args.verbose) {
    std::cout << "Non-Empty template stamps: " << templStamps.size() << std::endl;
    std::cout << "Non-Empty science stamps: " << scienceStamps.size() << std::endl;
  }
}

void identifySStampsInStamp(std::vector<Stamp>& stamps, Image& image) {

}

int identifySStamps(std::vector<Stamp>& stamps, Image& image) {

  int index = 0, hasSStamps = 0;
  for(auto& s : stamps) {
    calcStats(s, image);
    findSStamps(s, image, index);
    if(!s.subStamps.empty()) hasSStamps++;
    index++;
  }

  stamps.erase(std::remove_if(stamps.begin(), stamps.end(),
                              [](Stamp& s) { return s.subStamps.empty(); }),
               stamps.end());

  if(args.verbose) {
    std::cout << "Non-Empty stamps: " << stamps.size() << std::endl;
  }

  return hasSStamps;
}

void createStamps(Image& img, std::vector<Stamp>& stamps, int w, int h) {
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

      Stamp tmpS{};
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

double checkSStamp(SubStamp& sstamp, Image& image, Stamp& stamp) {
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
      if(image.isMasked(absCoords, ~Image::OK_CONV))
        return 0.0;

      if(image[absCoords] >= args.threshHigh) {
        image.maskPix(x, y, Image::BAD_PIXEL);
        return 0.0;
      }
      if((image[absCoords] - stamp.stats.skyEst) / stamp.stats.fwhm >
         args.threshKernFit)
        retVal += image[absCoords];
    }
  }
  return retVal;
}

cl_int findSStamps(Stamp& stamp, Image& image, int index) {
  double floor = stamp.stats.skyEst + args.threshKernFit * stamp.stats.fwhm;

  double dfrac = 0.9;
  while(stamp.subStamps.size() < size_t(args.maxSStamps)) {
    double lowestPSFLim =
        std::max(floor, stamp.stats.skyEst +
                            (args.threshHigh - stamp.stats.skyEst) * dfrac);
    for(long y = 0; y < args.fStampWidth; y++) {
      long absy = y + stamp.coords.second;
      for(long x = 0; x < args.fStampWidth; x++) {
        long absx = x + stamp.coords.first;
        long coords = x + (y * stamp.size.first);
        long absCoords = absx + (absy * image.axis.first);

        if (image.isMasked(absCoords, ~Image::OK_CONV)) {
          continue;
        }

        if(stamp[coords] > args.threshHigh) {
          image.maskPix(absx, absy, Image::BAD_PIXEL);
          continue;
        }

        if((stamp[coords] - stamp.stats.skyEst) * (1.0 / stamp.stats.fwhm) <
           args.threshKernFit) {
          continue;
        }

        if(stamp[coords] > lowestPSFLim) {  // good candidate found
          SubStamp s{{},
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

              if (image.isMasked(absCoords, ~Image::OK_CONV)) {
                continue;
              }

              if(image[kCoords] >= args.threshHigh) {
                image.maskPix(kx, ky, Image::BAD_PIXEL);
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
          s.val = checkSStamp(s, image, stamp);
          if(s.val == 0.0) continue;
          stamp.subStamps.push_back(s);

          for(int y = s.stampCoords.second - args.hSStampWidth;
              y <= s.stampCoords.second + args.hSStampWidth; y++) {
            int y2 = y + stamp.coords.second;
            for(int x = s.stampCoords.first - args.hSStampWidth;
                x <= s.stampCoords.first + args.hSStampWidth; x++) {
              int x2 = x + stamp.coords.first;
              if (x > 0 && x < stamp.size.first && y > 0 && y < stamp.size.second) {
                image.maskPix(x2, y2, Image::SKIP);
              }
            }
          }
        }
        if(stamp.subStamps.size() >= size_t(args.maxSStamps)) break;
      }
      if(stamp.subStamps.size() >= size_t(args.maxSStamps)) break;
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
  std::sort(stamp.subStamps.begin(), stamp.subStamps.end(),
            std::greater<SubStamp>());
  if(args.verbose)
    std::cout << "Added " << stamp.subStamps.size() << " substamps to stamp "
              << index << std::endl;
  return 0;
}
