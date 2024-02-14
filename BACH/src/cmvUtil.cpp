#include "bachUtil.h"

void createB(Stamp& s, Image& img) {
  /* Does Equation 2.13 which create the right side of the Equation Ma=B */

  s.B = {};
  s.B.emplace_back();
  auto [ssx, ssy] = s.subStamps[0].imageCoords;

  for(int i = 0; i < args.nPSF; i++) {
    double p0 = 0.0;
    for(int x = -args.hSStampWidth; x <= args.hSStampWidth; x++) {
      for(int y = -args.hSStampWidth; y <= args.hSStampWidth; y++) {
        int k =
            x + args.hSStampWidth + args.fSStampWidth * (y + args.hSStampWidth);
        int imgIndex = x + ssx + (y + ssy) * img.axis.first;
        p0 += s.W[i][k] * img[imgIndex];
      }
    }
    s.B.push_back(p0);
  }

  double q = 0.0;
  for(int x = -args.hSStampWidth; x <= args.hSStampWidth; x++) {
    for(int y = -args.hSStampWidth; y <= args.hSStampWidth; y++) {
      int k =
          x + args.hSStampWidth + args.fSStampWidth * (y + args.hSStampWidth);
      int imgIndex = x + ssx + (y + ssy) * img.axis.first;
      q += s.W[args.nPSF][k] * img[imgIndex];
    }
  }
  s.B.push_back(q);
}

void convStamp(Stamp& s, Image& img, Kernel& k, int n, int odd) {
  /*
   * Fills a Stamp with a convolved version (using only gaussian basis functions
   * without amlitude) of the area around its selected substamp.
   *
   * This can result in nan values but which should be handeld later.
   */

  s.W.emplace_back();
  auto [ssx, ssy] = s.subStamps[0].imageCoords;

  std::vector<float> tmp{};

  // Convolve Image with filterY taking pixels in a (args.hSStampWidth +
  // args.hKernelWidth) area around a substamp.
  for(int j = ssy - args.hSStampWidth; j <= ssy + args.hSStampWidth; j++) {
    for(int i = ssx - args.hSStampWidth - args.hKernelWidth;
        i <= ssx + args.hSStampWidth + args.hKernelWidth; i++) {
      tmp.push_back(0.0);

      for(int y = -args.hKernelWidth; y <= args.hKernelWidth; y++) {
        int imgIndex = i + (j + y) * img.axis.first;
        // cl_double v = std::isnan(img[imgIndex]) ? 1e-10 : img[imgIndex];
        float v = img[imgIndex];
        tmp.back() += v * k.filterY[n][args.hKernelWidth - y];
      }
    }
  }

  int subWidth = args.fKernelWidth + args.fSStampWidth - 1;
  // Convolve Image with filterX, image data already there.
  for(int j = -args.hSStampWidth; j <= args.hSStampWidth; j++) {
    for(int i = -args.hSStampWidth; i <= args.hSStampWidth; i++) {
      s.W[n].push_back(0.0);
      for(int x = -args.hKernelWidth; x <= args.hKernelWidth; x++) {
        int index = (i + x) + subWidth / 2 + (j + args.hSStampWidth) * subWidth;
        s.W.back().back() += tmp[index] * k.filterX[n][args.hKernelWidth - x];
      }
    }
  }

  // Removes n = 0 vector from all odd vectors in s.W
  // TODO: Find out why this is done.....
  if(odd) {
    for(int i = 0; i < args.fSStampWidth * args.fSStampWidth; i++)
      s.W[n][i] -= s.W[0][i];
  }
}

void cutSStamp(SubStamp& ss, Image& img, ImageMask& mask) {
  /* Store the original image data around the substamp in said substamp */

  for(int y = 0; y < args.fSStampWidth; y++) {
    int imgY = ss.imageCoords.second + y - args.hSStampWidth;

    for(int x = 0; x < args.fSStampWidth; x++) {
      int imgX = ss.imageCoords.first + x - args.hSStampWidth;
      int imgCoords = imgX + imgY * img.axis.first;

      ss.data.push_back(img[imgX + imgY * img.axis.first]);
      ss.sum += mask.isMasked(imgCoords, ImageMask::BAD_INPUT)
                    ? 0.0
                    : std::abs(img[imgCoords]);
    }
  }
}

int fillStamp(Stamp& s, Image& tImg, Image& sImg, ImageMask& mask, Kernel& k) {
  /* Fills Substamp with gaussian basis convolved images around said substamp
   * and calculates CMV.
   */

  if(s.subStamps.empty()) {
    if(args.verbose) {
      std::cout << "No eligable substamps in stamp at x = " << s.coords.first
                << " y = " << s.coords.second << ", stamp rejected"
                << std::endl;
    }
    return 1;
  }

  int nvec = 0;
  s.W = std::vector<std::vector<double>>();
  for(int g = 0; g < cl_int(args.dg.size()); g++) {
    for(int x = 0; x <= args.dg[g]; x++) {
      for(int y = 0; y <= args.dg[g] - x; y++) {
        int odd = 0;

        int dx = (x / 2) * 2 - x;
        int dy = (y / 2) * 2 - y;
        if(dx == 0 && dy == 0 && nvec > 0) odd = 1;

        convStamp(s, tImg, k, nvec, odd);
        nvec++;
      }
    }
  }

  cutSStamp(s.subStamps[0], sImg, mask);

  auto [ssx, ssy] = s.subStamps[0].imageCoords;

  for(int j = 0; j <= args.backgroundOrder; j++) {
    for(int k = 0; k <= args.backgroundOrder - j; k++) {
      s.W.emplace_back();
    }
  }
  for(int y = ssy - args.hSStampWidth; y <= ssy + args.hSStampWidth; y++) {
    double yf =
        (y - float(tImg.axis.second * 0.5)) / float(tImg.axis.second * 0.5);
    for(int x = ssx - args.hSStampWidth; x <= ssx + args.hSStampWidth; x++) {
      double xf =
          (x - float(tImg.axis.first * 0.5)) / float(tImg.axis.first * 0.5);
      double ax = 1.0;
      cl_int nBGVec = 0;
      for(int j = 0; j <= args.backgroundOrder; j++) {
        double ay = 1.0;
        for(int k = 0; k <= args.backgroundOrder - j; k++) {
          s.W[args.nPSF + nBGVec++].push_back(ax * ay);
          ay *= yf;
        }
        ax *= xf;
      }
    }
  }

  s.createQ();  // TODO: is name accurate?
  createB(s, sImg);

  return 0;
}
