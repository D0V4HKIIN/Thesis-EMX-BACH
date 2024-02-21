#include "bachUtil.h"
#include "mathUtil.h"

void createB(Stamp& s, const Image& img, const Arguments& args) {
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

void convStamp(Stamp& s, const Image& img, const Kernel& k, const int n, const int odd, const Arguments& args) {
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

void cutSStamp(SubStamp& ss, const Image& img, const ImageMask& mask, const Arguments& args) {
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

int fillStamps(std::vector<Stamp>& stamps, const Image& tImg, const Image& sImg, const cl::Buffer& tImgBuf, const cl::Buffer& sImgBuf, const ImageMask& mask, const Kernel& k, ClData& clData, ClStampsData& stampData, const Arguments& args) {
  /* Fills Substamp with gaussian basis convolved images around said substamp
   * and calculates CMV.
   */

  for (auto& s : stamps) {
    if(s.subStamps.empty()) {
      if(args.verbose) {
        std::cout << "No eligable substamps in stamp at x = " << s.coords.first
                  << " y = " << s.coords.second << ", stamp rejected"
                  << std::endl;
      }
      continue;
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

          convStamp(s, tImg, k, nvec, odd, args);
          nvec++;
        }
      }
    }
  }

  for (auto& s : stamps) {
    if(s.subStamps.empty()) {
      continue;
    }
    
    cutSStamp(s.subStamps[0], sImg, mask, args);
  }

  for (auto& s : stamps) {
    if(s.subStamps.empty()) {
      continue;
    }

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
  }

  for (auto& s : stamps) {
    if(s.subStamps.empty()) {
      continue;
    }

    s.createQ(args);
  }

  for (auto& s : stamps) {
    if(s.subStamps.empty()) {
      continue;
    }

    createB(s, sImg, args);
  }

  // TEMP: create sub stamp coordinates (should already be done)
  std::vector<cl_int> subStampCoords(2 * (2 * args.maxKSStamps) * stamps.size(), 0);

  for (int i = 0; i < stamps.size(); i++) {
    for (int j = 0; j < stamps[i].subStamps.size(); j++) {
      subStampCoords[2 * (i * 2 * args.maxKSStamps + j) + 0] = stamps[i].subStamps[j].imageCoords.first;
      subStampCoords[2 * (i * 2 * args.maxKSStamps + j) + 1] = stamps[i].subStamps[j].imageCoords.second;
    }
  }

  stampData.subStampCoords = cl::Buffer(clData.context, CL_MEM_READ_WRITE, sizeof(cl_int) * subStampCoords.size());
  clData.queue.enqueueWriteBuffer(stampData.subStampCoords, CL_TRUE, 0, sizeof(cl_int) * subStampCoords.size(), &subStampCoords[0]);

  clData.wColumns = args.fSStampWidth * args.fSStampWidth;
  clData.wRows = args.nPSF + triNum(args.backgroundOrder + 1);
  clData.qCount = args.nPSF + 2;
  clData.bCount = args.nPSF + 2;

  stampData.q = cl::Buffer(clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * clData.qCount * clData.qCount * stamps.size());
  stampData.b = cl::Buffer(clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * clData.bCount * stamps.size());

  // TEMP: create w buffer and transfer data to GPU
  stampData.w = cl::Buffer(clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * clData.wRows * clData.wColumns * stamps.size());

  std::vector<double> w{};

  for (auto& s : stamps) {
    for (auto& m : s.W) {
      for (auto& v : m) {
        w.push_back(v);
      }
    }
  }

  clData.queue.enqueueWriteBuffer(stampData.w, CL_TRUE, 0, sizeof(cl_double) * w.size(), &w[0]);

  // Create Q
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl_long, cl_long, cl_long, cl_long, cl_long>
                    qFunc(clData.program, "createQ");
  cl::EnqueueArgs qEargs(clData.queue, cl::NullRange, cl::NDRange(stamps.size(), clData.qCount, clData.qCount), cl::NullRange);
  cl::Event qEvent = qFunc(qEargs, stampData.w, stampData.q, clData.wRows, clData.wColumns,
                           clData.qCount, clData.qCount, args.fSStampWidth);

  // Create B
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                    cl_long, cl_long, cl_long, cl_long, cl_long, cl_long>
                    bFunc(clData.program, "createB");
  cl::EnqueueArgs bEargs(clData.queue, cl::NullRange, cl::NDRange(stamps.size(), clData.bCount), cl::NullRange);
  cl::Event bEvent = bFunc(bEargs, stampData.subStampCoords, sImgBuf,
                           stampData.w, stampData.b, clData.wRows, clData.wColumns, clData.bCount,
                           args.fSStampWidth, 2 * args.maxKSStamps, tImg.axis.first);

  qEvent.wait();
  bEvent.wait();

  // TEMP: transfer the data back to the CPU
  std::vector<double> gpuQ(clData.qCount * clData.qCount * stamps.size());
  std::vector<double> gpuB(clData.bCount * stamps.size());

  clData.queue.enqueueReadBuffer(stampData.q, CL_TRUE, 0, sizeof(cl_double) * gpuQ.size(), &gpuQ[0]);
  clData.queue.enqueueReadBuffer(stampData.b, CL_TRUE, 0, sizeof(cl_double) * gpuB.size(), &gpuB[0]);

  // TEMP: put data back in Q
  for (int i = 0; i < stamps.size(); i++) {
    auto& s = stamps[i];
    
    for (int j = 0; j < clData.qCount; j++) {
      s.Q[j].clear();

      for (int k = 0; k < clData.qCount; k++) {
        s.Q[j].push_back(gpuQ[i * clData.qCount * clData.qCount + j * clData.qCount + k]);
      }
    }
  }

  // TEMP: put data back in B
  for (int i = 0; i < stamps.size(); i++) {
    auto& s = stamps[i];
    s.B.clear();

    for (int j = 0; j < clData.bCount; j++) {
      s.B.push_back(gpuB[i * clData.bCount + j]);
    }
  }

  return 0;
}

int fillStamp(Stamp& s, const Image& tImg, const Image& sImg, const ImageMask& mask, const Kernel& k, const Arguments& args) {
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

        convStamp(s, tImg, k, nvec, odd, args);
        nvec++;
      }
    }
  }

  cutSStamp(s.subStamps[0], sImg, mask, args);

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

  s.createQ(args);  // TODO: is name accurate?
  createB(s, sImg, args);

  return 0;
}
