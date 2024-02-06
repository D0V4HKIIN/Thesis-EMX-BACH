#ifndef DATATYPE_UTIL
#define DATATYPE_UTIL

#include <CL/opencl.h>

#include <cmath>
#include <concepts>
#include <iostream>
#include <string>
#include <vector>

#include "argsUtil.h"

struct kernelStats {
  cl_int gauss;
  cl_int x;
  cl_int y;
};

struct Kernel {
  std::vector<std::vector<double>> kernVec{};

  /*
   * filterX and filterY is basically a convolution kernel, we probably can
   * represent it a such
   *
   * TODO: Implementation closer to the math. (Will also probably make it so we
   * can use openCL convolution )
   */

  std::vector<double> currKernel{};
  std::vector<std::vector<double>> filterX{};
  std::vector<std::vector<double>> filterY{};
  std::vector<kernelStats> stats{};
  std::vector<double> solution{};

  Kernel()
      : currKernel(args.fKernelWidth * args.fKernelWidth, 0.0),
        filterX{},
        filterY{},
        stats{},
        solution{} {
    resetKernVec();
  }

  void resetKernVec() {
    /* Fill Kerenel Vector
     * TODO: Make parallel, should be very possible. You can interprate stats as
     * a Vec3 in a kernel.
     */
    if(args.verbose) std::cout << "Creating kernel vectors..." << std::endl;
    int i = 0;
    for(int gauss = 0; gauss < cl_int(args.dg.size()); gauss++) {
      for(int x = 0; x <= args.dg[gauss]; x++) {
        for(int y = 0; y <= args.dg[gauss] - x; y++) {
          stats.push_back({gauss, x, y});
          resetKernelHelper(i);
          i++;
        }
      }
    }
  }

 private:
  void resetKernelHelper(cl_int n) {
    /* Will perfom one iteration of equation 2.3 but without a fit parameter.
     *
     * TODO: Make into a clKernel, look at hotpants for c indexing instead.
     */

    std::vector<double> temp(args.fKernelWidth * args.fKernelWidth, 0.0);
    double sumX = 0.0, sumY = 0.0;
    // UNSURE: Don't really know why dx,dy are a thing
    cl_int dx = (stats[n].x / 2) * 2 - stats[n].x;
    cl_int dy = (stats[n].y / 2) * 2 - stats[n].y;

    filterX.emplace_back();
    filterY.emplace_back();

    // Calculate Equation (2.4)
    for(int i = 0; i < args.fKernelWidth; i++) {
      double x = double(i - args.hKernelWidth);
      double qe = std::exp(-x * x * args.bg[stats[n].gauss]);
      filterX[n].push_back(qe * pow(x, stats[n].x));
      filterY[n].push_back(qe * pow(x, stats[n].y));
      sumX += filterX[n].back();
      sumY += filterY[n].back();
    }

    sumX = 1. / sumX;
    sumY = 1. / sumY;

    // UNSURE: Why the two different calculations?
    // It checks if deg x and y are even/odd
    if(dx == 0 && dy == 0) {
      for(int uv = 0; uv < args.fKernelWidth; uv++) {
        filterX[n][uv] *= sumX;
        filterY[n][uv] *= sumY;
      }

      for(int u = 0; u < args.fKernelWidth; u++) {
        for(int v = 0; v < args.fKernelWidth; v++) {
          temp[u + v * args.fKernelWidth] = filterX[n][u] * filterY[n][v];
          if(n > 0) {
            temp[u + v * args.fKernelWidth] -=
                kernVec[0][u + v * args.fKernelWidth];
          }
        }
      }
    } else {
      for(int u = 0; u < args.fKernelWidth; u++) {
        for(int v = 0; v < args.fKernelWidth; v++) {
          temp[u + v * args.fKernelWidth] = (filterX[n][u] * filterY[n][v]);
        }
      }
    }
    kernVec.push_back(temp);
  }
};

struct SubStamp {
  std::vector<double> data;
  double sum = 0.0;
  std::pair<cl_long, cl_long> imageCoords{};
  std::pair<cl_long, cl_long> stampCoords{};
  double val;

  bool operator<(const SubStamp& other) const { return val < other.val; }
  bool operator>(const SubStamp& other) const { return val > other.val; }

  double& operator[](size_t index) { return data[index]; }
};

struct StampStats {
  double skyEst{};  // Mode of stamp
  double fwhm{};    // Middle part value diff (full width half max)
  double norm{};
  double diff{};
  double chi2{};
};

struct Stamp {
  std::pair<cl_long, cl_long> coords{};
  std::pair<cl_long, cl_long> size{};
  std::pair<cl_long, cl_long> center{};
  std::vector<SubStamp> subStamps{};
  std::vector<double> data{};
  StampStats stats{};
  std::vector<std::vector<double>> W{};
  std::vector<std::vector<double>> Q{};
  std::vector<double> B{};

  Stamp(){};
  Stamp(std::pair<cl_long, cl_long> stampCoords,
        std::pair<cl_long, cl_long> stampSize, std::pair<cl_long, cl_long> c,
        const std::vector<SubStamp>& subStamps,
        const std::vector<double>& stampData)
      : coords{stampCoords},
        size{stampSize},
        center{c},
        subStamps{subStamps},
        data{stampData} {}

  double operator[](size_t index) { return data[index]; }

  double pixels() { return size.first * size.second; }

  void createQ() {
    /* Does Equation 2.12 which create the left side of the Equation Ma=B */
    Q = std::vector<std::vector<double>>(
        args.nPSF + 2, std::vector<double>(args.nPSF + 2, 0.0));

    for(int i = 0; i < args.nPSF; i++) {
      for(int j = 0; j <= i; j++) {
        double q = 0.0;
        for(int k = 0; k < args.fSStampWidth * args.fSStampWidth; k++) {
          q += W[i][k] * W[j][k];
        }
        Q[i + 1][j + 1] = q;
      }
    }

    for(int i = 0; i < args.nPSF; i++) {
      double p0 = 0.0;
      for(int k = 0; k < args.fSStampWidth * args.fSStampWidth; k++) {
        p0 += W[i][k] * W[args.nPSF][k];
      }
      Q[args.nPSF + 1][i + 1] = p0;
    }

    double q = 0.0;
    for(int k = 0; k < args.fSStampWidth * args.fSStampWidth; k++)
      q += W[args.nPSF][k] * W[args.nPSF][k];
    Q[args.nPSF + 1][args.nPSF + 1] = q;
  }
};

struct Image {
  std::string name;
  std::string path;
  std::pair<cl_long, cl_long> axis;

  enum masks
  {
    nan, // FLAG_ISNAN
    badInput, // FLAG_INPUT_ISBAD
    badPixel, // FLAG_T_BAD, FLAG_I_BAD
    psf,
    edge
  };
  std::vector<double> data{};

  std::vector<bool> nanMask{};
  std::vector<bool> badInputMask{};
  std::vector<bool> badPixelMask{};
  std::vector<bool> psfMask{};
  std::vector<bool> edgeMask{};

 public:
  Image(const std::string n, std::pair<cl_long, cl_long> a = {0L, 0L},
        const std::string p = "res/")
      : name{n},
        path{p},
        axis{a},
        data(this->size()),
        nanMask(this->size(), false),
        badInputMask(this->size(), false),
        badPixelMask(this->size(), false),
        psfMask(this->size(), false),
        edgeMask(this->size(), false) {}

  Image(const std::string n, std::vector<double> d,
        std::pair<cl_long, cl_long> a, const std::string p = "res/")
      : name{n},
        path{p},
        axis{a},
        data{d},
        nanMask(this->size(), false),
        badInputMask(this->size(), false),
        badPixelMask(this->size(), false),
        psfMask(this->size(), false),
        edgeMask(this->size(), false) {}

  bool isMaskedAny(int index) const {
    return nanMask[index] || badInputMask[index] ||
      badPixelMask[index] || psfMask[index] || edgeMask[index];
  }

  double* operator&() { return &data[0]; }

  double operator[](size_t index) { return float(data[index]); }

  std::string getFile() { return path + name; }

  std::string getFileName() {
    size_t lastI = name.find_last_of(".");
    return name.substr(0, lastI);
  }

  size_t size() { return (size_t)axis.first * axis.second; }

  std::string getOutFile() { return "!" + path + name; }

  long* axis_to_array() {
    static long tmpAx[2];
    tmpAx[0] = axis.first;
    tmpAx[1] = axis.second;
    long* ptr = tmpAx;
    return ptr;
  }

  void maskPix(int x, int y, std::same_as<Image::masks> auto... mI) {
    std::vector<Image::masks> mL{mI...};
    for(Image::masks m : mL) {
      switch(m) {
        case nan:
          nanMask[x + (y * axis.first)] = true;
          break;
        case badInput:
          badInputMask[x + (y * axis.first)] = true;
          break;
        case badPixel:
          badPixelMask[x + (y * axis.first)] = true;
          break;
        case psf:
          psfMask[x + (y * axis.first)] = true;
          break;
        case edge:
          edgeMask[x + (y * axis.first)] = true;
          break;
        default:
          std::cout << "Error: Not caught by the switch case" << std::endl;
          exit(1);
      }
    }
  }

  void maskSStamp(SubStamp& sstamp, std::same_as<Image::masks> auto... mI) {
    for(int x = sstamp.imageCoords.first - args.hSStampWidth;
        x <= sstamp.imageCoords.first + args.hSStampWidth; x++) {
      if(x < 0 || x >= axis.first) continue;
      for(int y = sstamp.imageCoords.second - args.hSStampWidth;
          y <= sstamp.imageCoords.second + args.hSStampWidth; y++) {
        if(y < 0 || y >= axis.second) continue;
        this->maskPix(x, y, mI...);
      }
    }
  }
};
#endif
