#define CL_HPP_TARGET_OPENCL_VERSION 300

#include "utils/bachUtil.h"

double testFit(std::vector<Stamp>& stamps, Image& tImg, Image& sImg) {
  int nComp1 = args.nPSF - 1;
  int nComp2 = ((args.kernelOrder + 1) * (args.kernelOrder + 2)) / 2;
  int nBGComp = ((args.backgroundOrder + 1) * (args.backgroundOrder + 2)) / 2;
  int matSize = nComp1 * nComp2 + nBGComp + 1;
  int nKernSolComp = args.nPSF * nComp2 + nBGComp + 1;

  std::vector<double> kernelSum(stamps.size(), 0.0);
  std::vector<int> index(nKernSolComp);  // Internal between ludcmp and lubksb.

  int count = 0;
  for(auto& s : stamps) {
    if(!s.subStamps.empty()) {
      double d;
      std::vector<double> testVec(args.nPSF + 2, 0.0);
      std::vector<std::vector<double>> testMat(
          args.nPSF + 2, std::vector<double>(args.nPSF + 2, 0.0));

      for(int i = 1; i <= args.nPSF + 1; i++) {
        testVec[i] = s.B[i];
        for(int j = 1; j <= i; j++) {
          testMat[i][j] = s.Q[i][j];
          testMat[j][i] = testMat[i][j];
        }
      }

      ludcmp(testMat, args.nPSF + 1, index, d);
      lubksb(testMat, args.nPSF + 1, index, testVec);
      s.stats.norm = testVec[1];
      kernelSum[count++] = testVec[1];
    }
  }

  double kernelMean, kernelStdev;
  sigmaClip(kernelSum, kernelMean, kernelStdev, 10);

  // normalise
  for(auto& s : stamps) {
    s.stats.diff = std::abs((s.stats.norm - kernelMean) / kernelStdev);
  }

  // global fit
  std::vector<Stamp> testStamps{};
  int c = 0;
  for(auto& s : stamps) {
    if(s.stats.diff < args.threshKernFit && !s.subStamps.empty()) {
      testStamps.push_back(s);
      c++;
    }
  }

  // do fit
  auto [matrix, weight] = createMatrix(testStamps, tImg.axis);
  std::vector<double> testKernSol = createScProd(testStamps, sImg, weight);

  double d;
  ludcmp(matrix, matSize, index, d);
  lubksb(matrix, matSize, index, testKernSol);

  Kernel testKern{};
  testKern.solution = testKernSol;
  kernelMean = makeKernel(testKern, tImg.axis, 0, 0);

  // calc merit value
  std::vector<double> merit{};
  double sig{};
  for(auto& ts : testStamps) {
    sig = calcSig(ts, testKern.solution, tImg, sImg);
    if(sig != -1 && sig <= 1e10) merit.push_back(sig);
  }
  double meritMean, meritStdDev;
  sigmaClip(merit, meritMean, meritStdDev, 10);
  //TODO: Add other merits before 666
  meritMean /= kernelMean;
  if(merit.size() > 0) return meritMean;
  return 666;
}

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
createMatrix(std::vector<Stamp>& stamps, std::pair<cl_long, cl_long>& imgSize) {
  int nComp1 = args.nPSF - 1;
  int nComp2 = ((args.kernelOrder + 1) * (args.kernelOrder + 2)) / 2;  // = 6
  int nComp = nComp1 * nComp2;
  int nBGVectors =
      ((args.backgroundOrder + 1) * (args.backgroundOrder + 2)) / 2;  // = 3
  int matSize = nComp + nBGVectors + 1;

  int pixStamp = args.fSStampWidth * args.fSStampWidth;
  float hPixX = imgSize.first / 2, hPixY = imgSize.second / 2;

  std::vector<std::vector<double>> matrix(
      matSize + 1, std::vector<double>(matSize + 1, 0.0));
  std::vector<std::vector<double>> weight(stamps.size(),
                                          std::vector<double>(nComp2, 0.0));

  for(size_t st = 0; st < stamps.size(); st++) {
    Stamp& s = stamps[st];
    if(s.subStamps.empty()) continue;

    auto [ssx, ssy] = s.subStamps[0].imageCoords;

    double fx = (ssx - hPixX) / hPixX;
    double fy = (ssy - hPixY) / hPixY;

    double a1 = 1.0;
    for(int k = 0, i = 0; i <= int(args.kernelOrder); i++) {
      double a2 = 1.0;
      for(int j = 0; j <= int(args.kernelOrder) - i; j++) {
        weight[st][k++] = a1 * a2;
        a2 *= fy;
      }
      a1 *= fx;
    }

    for(int i = 0; i < nComp; i++) {
      int i1 = i / nComp2;
      int i2 = i - i1 * nComp2;
      for(int j = 0; j <= i; j++) {
        int j1 = j / nComp2;
        int j2 = j - j1 * nComp2;

        matrix[i + 2][j + 2] +=
            weight[st][i2] * weight[st][j2] * s.Q[i1 + 2][j1 + 2];
      }
    }

    matrix[1][1] += s.Q[1][1];
    for(int i = 0; i < nComp; i++) {
      int i1 = i / nComp2;
      int i2 = i - i1 * nComp2;
      matrix[i + 2][1] += weight[st][i2] * s.Q[i1 + 2][1];
    }

    for(int iBG = 0; iBG < nBGVectors; iBG++) {
      int i = nComp + iBG + 1;
      int iVecBG = nComp1 + iBG + 1;
      for(int i1 = 1; i1 < nComp1 + 1; i1++) {
        double p0 = 0.0;

        for(int k = 0; k < pixStamp; k++) {
          p0 += s.W[i1][k] * s.W[iVecBG][k];
        }

        for(int i2 = 0; i2 < nComp2; i2++) {
          int jj = (i1 - 1) * nComp2 + i2 + 1;
          matrix[i + 1][jj + 1] += p0 * weight[st][i2];
        }
      }

      double p0 = 0.0;
      for(int k = 0; k < pixStamp; k++) {
        p0 += s.W[0][k] * s.W[iVecBG][k];
      }
      matrix[i + 1][1] += p0;

      for(int jBG = 0; jBG <= iBG; jBG++) {
        double q = 0.0;
        for(int k = 0; k < pixStamp; k++) {
          q += s.W[iVecBG][k] * s.W[nComp1 + jBG + 1][k];
        }
        matrix[i + 1][nComp + jBG + 2] += q;
      }
    }
  }

  for(int i = 0; i < matSize; i++) {
    for(int j = 0; j <= i; j++) {
      matrix[j + 1][i + 1] = matrix[i + 1][j + 1];
    }
  }

  return std::make_pair(matrix, weight);
}

std::vector<double> createScProd(std::vector<Stamp>& stamps, Image& img,
                                 std::vector<std::vector<double>>& weight) {
  int nComp1 = args.nPSF - 1;
  int nComp2 = ((args.kernelOrder + 1) * (args.kernelOrder + 2)) / 2;
  int nBGComp = ((args.backgroundOrder + 1) * (args.backgroundOrder + 2)) / 2;
  int nKernSolComp = args.nPSF * nComp2 + nBGComp + 1;

  std::vector<double> res(nKernSolComp, 0.0);

  int sI = 0;
  for(auto& s : stamps) {
    if(s.subStamps.empty()) {
      sI++;
      continue;
    }
    auto [ssx, ssy] = s.subStamps[0].imageCoords;

    double p0 = s.B[1];
    res[1] += p0;

    for(int i = 1; i < nComp1 + 1; i++) {
      p0 = s.B[i + 1];
      for(int j = 0; j < nComp2; j++) {
        int indx = (i - 1) * nComp2 + j + 1;
        res[indx + 1] += p0 * weight[sI][j];
      }
    }

    for(int bgIndex = 0; bgIndex < nBGComp; bgIndex++) {
      double q = 0.0;
      for(int x = -args.hSStampWidth; x <= args.hSStampWidth; x++) {
        for(int y = -args.hSStampWidth; y <= args.hSStampWidth; y++) {
          int index = x + args.hSStampWidth +
                      args.fSStampWidth * (y + args.hSStampWidth);
          q += s.W[nComp1 + bgIndex + 1][index] *
               img[x + ssx + (y + ssy) * img.axis.first];
        }
      }
      res[nComp1 * nComp2 + bgIndex + 2] += q;
    }

    sI++;
  }
  return res;
}

double calcSig(Stamp& s, std::vector<double>& kernSol, Image& tImg,
               Image& sImg) {
  if(s.subStamps.empty()) return -1.0;
  auto [ssx, ssy] = s.subStamps[0].imageCoords;

  double background = getBackground(ssx, ssy, kernSol, tImg.axis);
  std::vector<float> tmp{makeModel(s, kernSol, tImg.axis)};

  int sigCount = 0;
  double signal = 0.0;
  for(int y = 0; y < args.fSStampWidth; y++) {
    int absY = y - args.hSStampWidth + ssy;
    for(int x = 0; x < args.fSStampWidth; x++) {
      int absX = x - args.hSStampWidth + ssx;

      int intIndex = x + y * args.fSStampWidth;
      int absIndex = absX + absY * tImg.axis.first;
      double tDat = tmp[intIndex];

      double diff = tDat - sImg[absIndex] + background;
      if(sImg.badInputMask[absIndex] || sImg.nanMask[absIndex] ||
         std::abs(sImg[absIndex]) <= 1e-10) {
        continue;
      } else {
        tmp[intIndex] = diff;
      }
      if(std::isnan(tDat) || std::isnan(sImg[absIndex])) {
        tImg.maskPix(absX, absY, Image::nan);
        continue;
      }

      sigCount++;
      signal +=
          diff * diff / (std::abs(tImg[absIndex]) + std::abs(sImg[absIndex]));
    }
  }
  if(sigCount > 0) {
    signal /= sigCount;
    if(signal >= 1e10) signal = -1.0;
  } else {
    signal = -1.0;
  }
  return signal;
}

double getBackground(int x, int y, std::vector<double>& kernSol,
                     std::pair<cl_long, cl_long> imgSize) {
  int BGComp = (args.nPSF - 1) *
                   (((args.kernelOrder + 1) * (args.kernelOrder + 2)) / 2) +
               1;
  double bg = 0.0;
  double xf = (x - 0.5 * imgSize.first) / (0.5 * imgSize.first);
  double yf = (y - 0.5 * imgSize.second) / (0.5 * imgSize.second);

  double ax = 1.0;
  for(int i = 0, k = 1; i <= args.backgroundOrder; i++) {
    double ay = 1.0;
    for(int j = 0; j <= args.backgroundOrder - i; j++) {
      bg += kernSol[BGComp + k++] * ax * ay;
      ay *= yf;
    }
    ax *= xf;
  }
  return bg;
}

std::vector<float> makeModel(Stamp& s, std::vector<double>& kernSol,
                             std::pair<cl_long, cl_long> imgSize) {
  std::vector<float> model(args.fSStampWidth * args.fSStampWidth, 0.0);

  std::pair<float, float> hImgAxis =
      std::make_pair(0.5 * imgSize.first, 0.5 * imgSize.second);
  auto [ssx, ssy] = s.subStamps.front().imageCoords;

  for(int i = 0; i < args.fSStampWidth * args.fSStampWidth; i++) {
    model[i] += kernSol[1] * s.W[0][i];
  }

  for(int i = 1, k = 2; i < args.nPSF; i++) {
    double aX = 1.0, coeff = 0.0;
    for(int iX = 0; iX <= args.kernelOrder; iX++) {
      double aY = 1.0;
      for(int iY = 0; iY <= args.kernelOrder - iX; iY++) {
        coeff += kernSol[k++] * aX * aY;
        aY *= double(ssy - hImgAxis.second) / hImgAxis.second;
      }
      aX *= double(ssx - hImgAxis.first) / hImgAxis.first;
    }

    for(int j = 0; j < args.fSStampWidth * args.fSStampWidth; j++) {
      model[j] += coeff * s.W[i][j];
    }
  }

  return model;
}

void fitKernel(Kernel& k, std::vector<Stamp>& stamps, Image& tImg,
               Image& sImg) {
  int nComp1 = args.nPSF - 1;
  int nComp2 = ((args.kernelOrder + 1) * (args.kernelOrder + 2)) / 2;
  int nBGComp = ((args.backgroundOrder + 1) * (args.backgroundOrder + 2)) / 2;
  int matSize = nComp1 * nComp2 + nBGComp + 1;

  auto [fittingMatrix, weight] = createMatrix(stamps, tImg.axis);
  std::vector<double> solution = createScProd(stamps, sImg, weight);

  std::vector<int> index(matSize, 0);
  double d{};
  ludcmp(fittingMatrix, matSize, index, d);
  lubksb(fittingMatrix, matSize, index, solution);

  k.solution = solution;
  bool check = checkFitSolution(k, stamps, tImg, sImg);
  while(check) {
    if(args.verbose) std::cout << "Re-expanding matrix..." << std::endl;
    auto [fittingMatrix, weight] = createMatrix(stamps, tImg.axis);
    solution = createScProd(stamps, sImg, weight);

    ludcmp(fittingMatrix, matSize, index, d);
    lubksb(fittingMatrix, matSize, index, solution);

    k.solution = solution;
    check = checkFitSolution(k, stamps, tImg, sImg);
  }
}

bool checkFitSolution(Kernel& k, std::vector<Stamp>& stamps, Image& tImg,
                      Image& sImg) {
  std::vector<double> ssValues{};

  bool check = false;

  for(Stamp& s : stamps) {
    if(!s.subStamps.empty()) {
      double sig = calcSig(s, k.solution, tImg, sImg);

      if(sig == -1) {
        s.subStamps.erase(s.subStamps.begin(), next(s.subStamps.begin()));
        fillStamp(s, tImg, sImg, k);
        check = true;
      } else {
        s.stats.chi2 = sig;
        ssValues.push_back(sig);
      }
    }
  }

  double mean = 0.0, stdDev = 0.0;
  sigmaClip(ssValues, mean, stdDev, 10);

  if(args.verbose) {
    std::cout << "Mean sig: " << mean << " stdev: " << stdDev << '\n'
              << "    Iterating through stamps with sig >"
              << (mean + args.sigKernFit * stdDev) << std::endl;
  }

  for(Stamp& s : stamps) {
    if(!s.subStamps.empty()) {
      if((s.stats.chi2 - mean) > args.sigKernFit * stdDev) {
        s.subStamps.erase(s.subStamps.begin(), next(s.subStamps.begin()));
        fillStamp(s, tImg, sImg, k);
        check = true;
      }
    }
  }

  int cnt = 0;
  for(auto s : stamps) {
    if(!s.subStamps.empty()) cnt++;
  }
  if(args.verbose) {
    std::cout << "We use " << cnt << " sub-stamps" << std::endl;
    std::cout << "Remaining sub-stamps are:" << std::endl;
    for(auto s : stamps) {
      if(!s.subStamps.empty()) {
        std::cout << "x = " << s.coords.first << ", y = " << s.coords.second
                  << std::endl;
      }
    }
  }
  return check;
}
