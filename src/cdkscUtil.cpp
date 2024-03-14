#include "bachUtil.h"
#include "mathUtil.h"

double testFit(std::vector<Stamp>& stamps, const std::pair<cl_long, cl_long> &axis, const cl::Buffer &tImgBuf, const cl::Buffer &sImgBuf, ClData& clData, ClStampsData& stampData, const Arguments& args) {
  const int nComp1 = args.nPSF - 1;
  const int nComp2 = triNum(args.kernelOrder + 1);
  const int nBGComp = triNum(args.backgroundOrder + 1);
  const int matSize = nComp1 * nComp2 + nBGComp + 1;
  const int nKernSolComp = args.nPSF * nComp2 + nBGComp + 1;

  std::vector<int> index1(nKernSolComp);  // Internal between ludcmp and lubksb.

  // Create buffers
  cl::Buffer index(clData.context, CL_MEM_READ_WRITE, sizeof(cl_int) * nKernSolComp * stamps.size());
  cl::Buffer vv(clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * std::max<int>(matSize + 1, (args.nPSF + 2) * stamps.size()));
  cl::Buffer testVec(clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * clData.bCount * stamps.size());
  cl::Buffer testMat(clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * clData.qCount * clData.qCount * stamps.size());
  cl::Buffer kernelSums(clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * stamps.size());
  cl::Buffer weights(clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * stamps.size() * nComp2);
  cl::Buffer matrix(clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * (matSize + 1) * (matSize + 1));
  cl::Buffer testKernSol(clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * nKernSolComp);

  // Create test vec
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl_long> testVecFunc(clData.program, "createTestVec");
  cl::EnqueueArgs testVecEargs(clData.queue, cl::NullRange, cl::NDRange(clData.bCount, stamps.size()), cl::NullRange);
  cl::Event testVecEvent = testVecFunc(testVecEargs, stampData.b, testVec, clData.bCount);

  // Create test mat
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl_long> testMatFunc(clData.program, "createTestMat");
  cl::EnqueueArgs testMatEargs(clData.queue, cl::NullRange, cl::NDRange(clData.qCount, clData.qCount, stamps.size()), cl::NullRange);
  cl::Event testMatEvent = testMatFunc(testMatEargs, stampData.q, testMat, clData.qCount);

  testVecEvent.wait();
  testMatEvent.wait();

  // LU-solve
  ludcmp(testMat, args.nPSF + 2, stamps.size(), index, vv, clData);
  lubksb(testMat, args.nPSF + 2, stamps.size(), index, testVec, clData);

  // Save kernel sums
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl_int> kernelSumFunc(clData.program, "saveKernelSums");
  cl::EnqueueArgs kernelSumEargs(clData.queue, cl::NullRange, cl::NDRange(stamps.size()), cl::NullRange);
  cl::Event kernelSumEvent = kernelSumFunc(kernelSumEargs, testVec, kernelSums, args.nPSF + 2);

  kernelSumEvent.wait();

  // TEMP: transfer to CPU
  std::vector<double> kernelSum(stamps.size(), 0.0);
  clData.queue.enqueueReadBuffer(kernelSums, CL_TRUE, 0, sizeof(cl_double) * kernelSum.size(), kernelSum.data());

  for (int i = 0; i < stamps.size(); i++) {
    stamps[i].stats.norm = kernelSum[i];
  }

  double kernelMean, kernelStdev;
  sigmaClip(kernelSums, stamps.size(), &kernelMean, &kernelStdev, 10, clData, args);

  // Fit stamps, generate test stamps
  int testStampCount = 0;
  
  cl::Buffer testStampCountBuf(clData.context, CL_MEM_READ_WRITE, sizeof(cl_int));
  cl::Buffer testStampIndices(clData.context, CL_MEM_READ_WRITE, sizeof(cl_int) * stamps.size());
  clData.queue.enqueueWriteBuffer(testStampCountBuf, CL_TRUE, 0, sizeof(cl_int), &testStampCount);

  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer,
                    cl_double, cl_double, cl_double> testStampFunc(clData.program, "genCdTestStamps");
  cl::EnqueueArgs testStampEargs(clData.queue, cl::NullRange, cl::NDRange(stamps.size()), cl::NullRange);
  cl::Event testStampEvent = testStampFunc(testStampEargs, kernelSums, testStampIndices, testStampCountBuf,
                                           kernelMean, kernelStdev, args.sigKernFit);

  clData.queue.enqueueReadBuffer(testStampCountBuf, CL_TRUE, 0, sizeof(cl_int), &testStampCount);

  if (testStampCount == 0) {
    return 666;
  }

  // Allocate test stamps, so we have continuous stamp data, since
  // some stamps may be removed
  ClStampsData testStampData{};
  testStampData.subStampCoords = cl::Buffer(clData.context, CL_MEM_READ_WRITE, sizeof(cl_int) * 2 * (2 * args.maxKSStamps) * testStampCount);
  testStampData.subStampCounts = cl::Buffer(clData.context, CL_MEM_READ_WRITE, sizeof(cl_int) * testStampCount);
  testStampData.w = cl::Buffer(clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * testStampCount * clData.wColumns * clData.wRows);
  testStampData.q = cl::Buffer(clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * testStampCount * clData.qCount * clData.qCount);
  testStampData.b = cl::Buffer(clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * testStampCount * clData.bCount);
  testStampData.stampCount = testStampCount;

  std::vector<cl::Event> testEvents{};

  // Copy substamp coordinates
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl_long> copySsCoordsFunc(clData.program, "copyTestSubStampsCoords");
  cl::EnqueueArgs copySsCoordsEargs(clData.queue, cl::NullRange, cl::NDRange(2, 2 * args.maxKSStamps, testStampCount), cl::NullRange);
  testEvents.push_back(copySsCoordsFunc(copySsCoordsEargs, stampData.subStampCoords, testStampIndices, testStampData.subStampCoords, 2 * args.maxKSStamps));

  // Copy substamp counts
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer> copySsCountsFunc(clData.program, "copyTestSubStampsCounts");
  cl::EnqueueArgs copySsCountsEargs(clData.queue, cl::NullRange, cl::NDRange(testStampCount), cl::NullRange);
  testEvents.push_back(copySsCountsFunc(copySsCountsEargs, stampData.subStampCounts, testStampIndices, testStampData.subStampCounts));

  // Copy W
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl_long, cl_long> copyWFunc(clData.program, "copyTestStampsW");
  cl::EnqueueArgs copyWEargs(clData.queue, cl::NullRange, cl::NDRange(clData.wColumns, clData.wRows, testStampCount), cl::NullRange);
  testEvents.push_back(copyWFunc(copyWEargs, stampData.w, testStampIndices, testStampData.w, clData.wRows, clData.wColumns));

  // Copy Q
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl_long> copyQFunc(clData.program, "copyTestStampsQ");
  cl::EnqueueArgs copyQEargs(clData.queue, cl::NullRange, cl::NDRange(clData.qCount, clData.qCount, testStampCount), cl::NullRange);
  testEvents.push_back(copyQFunc(copyQEargs, stampData.q, testStampIndices, testStampData.q, clData.qCount));

  // Copy B
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl_long> copyBFunc(clData.program, "copyTestStampsB");
  cl::EnqueueArgs copyBEargs(clData.queue, cl::NullRange, cl::NDRange(clData.bCount, testStampCount), cl::NullRange);
  testEvents.push_back(copyBFunc(copyBEargs, stampData.b, testStampIndices, testStampData.b, clData.bCount));

  cl::Event::waitForEvents(testEvents);

  // TEMP: read back data
  std::vector<cl_int> gpuSsCoords(2 * testStampCount * (2 * args.maxKSStamps), 0);
  std::vector<cl_int> gpuSsCounts(testStampCount, 0);
  std::vector<cl_double> gpuW(testStampCount * clData.wColumns * clData.wRows, 0.0);
  std::vector<cl_double> gpuQ(testStampCount * clData.qCount * clData.qCount, 0.0);
  std::vector<cl_double> gpuB(testStampCount * clData.bCount, 0.0);

  clData.queue.enqueueReadBuffer(testStampData.subStampCoords, CL_TRUE, 0, sizeof(cl_int) * gpuSsCoords.size(), gpuSsCoords.data());
  clData.queue.enqueueReadBuffer(testStampData.subStampCounts, CL_TRUE, 0, sizeof(cl_int) * gpuSsCounts.size(), gpuSsCounts.data());
  clData.queue.enqueueReadBuffer(testStampData.w, CL_TRUE, 0, sizeof(cl_double) * gpuW.size(), gpuW.data());
  clData.queue.enqueueReadBuffer(testStampData.q, CL_TRUE, 0, sizeof(cl_double) * gpuQ.size(), gpuQ.data());
  clData.queue.enqueueReadBuffer(testStampData.b, CL_TRUE, 0, sizeof(cl_double) * gpuB.size(), gpuB.data());

  std::vector<Stamp> testStamps2{};
  for (int i = 0; i < testStampCount; i++) {
    Stamp s{};

    for (int j = 0; j < gpuSsCounts[i]; j++) {
      SubStamp ss{};
      ss.imageCoords = std::make_pair(gpuSsCoords[2 * (i * (2 * args.maxKSStamps) + j) + 0],
                                      gpuSsCoords[2 * (i * (2 * args.maxKSStamps) + j) + 1]);

      s.subStamps.push_back(ss);
    }

    for (int j = 0; j < clData.wRows; j++) {
      s.W.emplace_back();

      for (int k = 0; k < clData.wColumns; k++) {
        s.W.back().push_back(gpuW[i * clData.wColumns * clData.wRows + j * clData.wColumns + k]);
      }
    }

    for (int j = 0; j < clData.qCount; j++) {
      s.Q.emplace_back();

      for (int k = 0; k < clData.qCount; k++) {
        s.Q.back().push_back(gpuQ[i * clData.qCount * clData.qCount + j * clData.qCount + k]);
      }
    }

    for (int j = 0; j < clData.bCount; j++) {
      s.B.push_back(gpuB[i * clData.bCount + j]);
    }

    testStamps2.push_back(s);
  }

  // normalise
  for(auto& s : stamps) {
    s.stats.diff = std::abs((s.stats.norm - kernelMean) / kernelStdev);
  }

  // global fit
  std::vector<Stamp> testStamps{};
  for(auto& s : stamps) {
    if(s.stats.diff < args.sigKernFit && !s.subStamps.empty()) {
      testStamps.push_back(s);
    }
  }

  testStamps = testStamps2;

  // Do fit
  createMatrix(matrix, weights, clData, testStampData, axis, args);
  createScProd(testKernSol, weights, sImgBuf, axis, clData, testStampData, args);

  // TEMP: parallel matrix solver is currently very slow, so temporarly use CPU version
#if true
  // TEMP: transfer matrix back to CPU
  std::vector<cl_double> matrixCpu2((matSize + 1) * (matSize + 1));
  clData.queue.enqueueReadBuffer(matrix, CL_TRUE, 0, sizeof(cl_double) * matrixCpu2.size(), matrixCpu2.data());

  std::vector<std::vector<cl_double>> matrixCpu(matSize + 1, std::vector<cl_double>(matSize + 1));

  for (int i = 0; i < matSize + 1; i++) {
    for (int j = 0; j < matSize + 1; j++) {
      matrixCpu[i][j] = matrixCpu2[i * (matSize + 1) + j];
    }
  }

  // TEMP: transfer kernel solution back to CPU
  std::vector<cl_double> testKernSolCpu(nKernSolComp);
  clData.queue.enqueueReadBuffer(testKernSol, CL_TRUE, 0, sizeof(cl_double) * testKernSolCpu.size(), testKernSolCpu.data());

  double d;
  ludcmp(matrixCpu, matSize, index1, d, args);
  lubksb(matrixCpu, matSize, index1, testKernSolCpu);

  // TEMP: transferback to GPU
  clData.queue.enqueueWriteBuffer(testKernSol, CL_TRUE, 0, sizeof(cl_double) * testKernSolCpu.size(), testKernSolCpu.data());
#else
  ludcmp(matrix, matSize + 1, 1, index, vv, clData);
  lubksb(matrix, matSize + 1, 1, index, testKernSol, clData);
  
  // TEMP: transfer kernel solution back (again) to CPU
  std::vector<cl_double> testKernSolCpu(nKernSolComp);
  clData.queue.enqueueReadBuffer(testKernSol, CL_TRUE, 0, sizeof(cl_double) * testKernSolCpu.size(), testKernSolCpu.data());
#endif
  
  cl::Buffer kernel(clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * args.fKernelWidth * args.fKernelWidth);
  kernelMean = makeKernel(kernel, testKernSol, axis, 0, 0, args, clData);

  // Calc merit value
  cl::Buffer model(clData.context, CL_MEM_READ_WRITE, sizeof(cl_float) * testStampCount * args.fSStampWidth * args.fSStampWidth);
  cl::Buffer merits(clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * testStampCount);
  int meritCount = calcSigs(tImgBuf, sImgBuf, axis, model, testKernSol, merits, testStampData, clData, args);

  if (meritCount == 0) {
    return 666;
  }

  double meritMean;
  double meritStdDev;
  sigmaClip(merits, meritCount, &meritMean, &meritStdDev, 10, clData, args);

  double normMeritMean = meritMean / kernelMean;
  return normMeritMean;
}

void createMatrix(const cl::Buffer &matrix, const cl::Buffer &weights, const ClData &clData, const ClStampsData &stampData, const std::pair<cl_long, cl_long>& imgSize, const Arguments& args) {
  const int nComp1 = args.nPSF - 1;
  const int nComp2 = triNum(args.kernelOrder + 1);
  const int nComp = nComp1 * nComp2;
  const int nBGVectors = triNum(args.backgroundOrder + 1);
  const int matSize = nComp + nBGVectors + 1;

  const int pixStamp = args.fSStampWidth * args.fSStampWidth;

  // Create weights
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer,
                    cl_long, cl_long, cl_long, cl_long> weightFunc(clData.program, "createMatrixWeights");
  cl::EnqueueArgs weightEargs(clData.queue, cl::NullRange, cl::NDRange(nComp2, stampData.stampCount), cl::NullRange);
  cl::Event weightEvent = weightFunc(weightEargs, stampData.subStampCoords, clData.cd.kernelXy, weights, imgSize.first, imgSize.second, 2 * args.maxKSStamps, nComp2);

  weightEvent.wait();

  // Create matrix
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                    cl_long, cl_long, cl_long, cl_long, cl_long,
                    cl_long, cl_long, cl_long> matrixFunc(clData.program, "createMatrix");
  cl::EnqueueArgs matrixEargs(clData.queue, cl::NDRange(matSize + 1, matSize + 1));
  cl::Event matrixEvent = matrixFunc(matrixEargs, weights, stampData.w, stampData.q, matrix,
                                     stampData.stampCount, matSize + 1, pixStamp, nComp1, nComp2,
                                     clData.wRows, clData.wColumns, clData.qCount);

  matrixEvent.wait();
}

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
createMatrix(const std::vector<Stamp>& stamps, const std::pair<cl_long, cl_long>& imgSize, const Arguments& args) {
  const int nComp1 = args.nPSF - 1;
  const int nComp2 = triNum(args.kernelOrder + 1);
  const int nComp = nComp1 * nComp2;
  const int nBGVectors = triNum(args.backgroundOrder + 1);
  const int matSize = nComp + nBGVectors + 1;

  const int pixStamp = args.fSStampWidth * args.fSStampWidth;
  const float hPixX = 0.5 * imgSize.first;
  const float hPixY = 0.5 * imgSize.second;

  std::vector<std::vector<double>> matrix(
      matSize + 1, std::vector<double>(matSize + 1, 0.0));
  std::vector<std::vector<double>> weight(stamps.size(),
                                          std::vector<double>(nComp2, 0.0));

  for(size_t st = 0; st < stamps.size(); st++) {
    const Stamp& s = stamps[st];
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

void createScProd(const cl::Buffer &res, const cl::Buffer &weights, const cl::Buffer &img, const std::pair<cl_long, cl_long>& imgSize, const ClData &clData, const ClStampsData &stampData, const Arguments& args) {
  const int nComp1 = args.nPSF - 1;
  const int nComp2 = triNum(args.kernelOrder + 1);
  const int nBgComp = triNum(args.backgroundOrder + 1);
  const int nKernSolComp = args.nPSF * nComp2 + nBgComp + 1;
  
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                    cl_long, cl_long, cl_long, cl_long, cl_long,
                    cl_long, cl_long, cl_long, cl_long, cl_long> prodFunc(clData.program, "createScProd");
  cl::EnqueueArgs prodEargs(clData.queue, cl::NDRange(nKernSolComp));
  cl::Event prodEvent = prodFunc(prodEargs, img, weights, stampData.b, stampData.w, stampData.subStampCoords, res,
                                 imgSize.first, stampData.stampCount, nComp1, nComp2, nBgComp,
                                 clData.bCount, clData.wRows, clData.wColumns, args.fSStampWidth, 2 * args.maxKSStamps);

  prodEvent.wait();
}

std::vector<double> createScProd(const std::vector<Stamp>& stamps, const Image& img,
                                 const std::vector<std::vector<double>>& weight, const Arguments& args) {
  const int nComp1 = args.nPSF - 1;
  const int nComp2 = triNum(args.kernelOrder + 1);
  const int nBGComp = triNum(args.backgroundOrder + 1);
  const int nKernSolComp = args.nPSF * nComp2 + nBGComp + 1;

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

int calcSigs(const cl::Buffer &tImgBuf, const cl::Buffer &sImgBuf, const std::pair<cl_long, cl_long> &axis,
              const cl::Buffer &model, const cl::Buffer &kernSol, const cl::Buffer &sigma,
              const ClStampsData &stampData, const ClData &clData, const Arguments& args) {
  static constinit int localSize = 32;

  int reduceCount = (args.fSStampWidth * args.fSStampWidth + localSize - 1) / localSize;
  int stampCount = stampData.stampCount;
  int sigmaCount = 0;

  // Create buffers
  cl::Buffer bg(clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * stampCount);
  cl::Buffer sigTemp1(clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * stampCount * reduceCount);
  cl::Buffer sigTemp2(clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * stampCount * reduceCount);
  cl::Buffer sigCount1(clData.context, CL_MEM_READ_WRITE, sizeof(cl_int) * stampCount * reduceCount);
  cl::Buffer sigCount2(clData.context, CL_MEM_READ_WRITE, sizeof(cl_int) * stampCount * reduceCount);
  cl::Buffer sigCounter(clData.context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(cl_int), &sigmaCount);

  // Create bg
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                    cl_long, cl_long, cl_long, cl_long, cl_long, cl_long> bgFunc(clData.program, "calcSigBg");
  cl::EnqueueArgs bgEargs(clData.queue, cl::NDRange(stampCount));
  cl::Event bgEvent = bgFunc(bgEargs, kernSol, stampData.subStampCoords, stampData.subStampCounts, bg,
                             2 * args.maxKSStamps, axis.first, axis.second, args.backgroundOrder,
                             triNum(args.backgroundOrder + 1), (args.nPSF - 1) * triNum(args.kernelOrder + 1) + 1);

  // Create model
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                    cl_long, cl_long, cl_long, cl_long, cl_long,
                    cl_long, cl_long, cl_long> modelFunc(clData.program, "makeModel");
  cl::EnqueueArgs modelEargs(clData.queue, cl::NDRange(args.fSStampWidth * args.fSStampWidth, stampCount));
  cl::Event modelEvent = modelFunc(modelEargs, stampData.w, kernSol, stampData.subStampCoords, stampData.subStampCounts,
                                   model, args.nPSF, args.kernelOrder, clData.wRows, clData.wColumns, 2 * args.maxKSStamps,
                                   axis.first, axis.second, args.fSStampWidth * args.fSStampWidth);

  bgEvent.wait();
  modelEvent.wait();

  // Create sigmas
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                    cl::Buffer, cl::Buffer, cl::Buffer,
                    cl_long, cl_long, cl_long, cl_long, cl_long> sigmaFunc(clData.program, "calcSig");
  cl::EnqueueArgs sigmaEargs(clData.queue, cl::NDRange(reduceCount * localSize, stampCount), cl::NDRange(localSize, 1));
  cl::Event sigmaEvent = sigmaFunc(sigmaEargs, model, bg, tImgBuf, sImgBuf, stampData.subStampCoords,
                                   sigTemp1, sigCount1, clData.maskBuf, axis.first, args.fSStampWidth,
                                   2 * args.maxKSStamps, args.fSStampWidth * args.fSStampWidth, reduceCount);

  sigmaEvent.wait();

  // Reduce
  bool isFirst = true;
  int count = reduceCount;
  cl::Buffer *sigIn = &sigTemp1;
  cl::Buffer *sigOut = &sigTemp2;
  cl::Buffer *sigCountIn = &sigCount1;
  cl::Buffer *sigCountOut = &sigCount2;

  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                    cl_long, cl_long> reduceFunc(clData.program, "reduceSig");

  while (count > 1 || isFirst) {
    int nextCount = (count + localSize - 1) / localSize;

    cl::EnqueueArgs reduceEargs(clData.queue, cl::NDRange(roundUpToMultiple(count, localSize), stampCount), cl::NDRange(localSize, 1));
    cl::Event reduceEvent = reduceFunc(reduceEargs, *sigIn, *sigCountIn, *sigOut, *sigCountOut, sigCounter, count, nextCount);

    count = nextCount;
    std::swap(sigIn, sigOut);
    std::swap(sigCountIn, sigCountOut);

    isFirst = false;
  }

  clData.queue.enqueueReadBuffer(sigCounter, CL_TRUE, 0, sizeof(cl_int), &sigmaCount);

  // Copy buffer
  cl::Event copyEvent{};
  clData.queue.enqueueCopyBuffer(*sigIn, sigma, 0, 0, sizeof(cl_double) * sigmaCount, nullptr, &copyEvent);

  copyEvent.wait();

  return sigmaCount;
}

double calcSig(Stamp& s, const std::vector<double>& kernSol, const Image& tImg,
               const Image& sImg, ImageMask& mask, const Arguments& args) {
  if(s.subStamps.empty()) return -1.0;
  auto [ssx, ssy] = s.subStamps[0].imageCoords;

  double background = getBackground(ssx, ssy, kernSol, tImg.axis, args);
  std::vector<float> tmp{makeModel(s, kernSol, tImg.axis, args)};

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
      if(mask.isMasked(absIndex, ImageMask::BAD_INPUT) ||
         std::abs(sImg[absIndex]) <= 1e-10) {
        continue;
      } else {
        tmp[intIndex] = diff;
      }
      if(std::isnan(tDat) || std::isnan(sImg[absIndex])) {
        mask.maskPix(absX, absY, ImageMask::NAN_PIXEL);
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

double getBackground(const int x, const int y, const std::vector<double>& kernSol,
                     const std::pair<cl_long, cl_long> imgSize, const Arguments& args) {
  int BGComp = (args.nPSF - 1) * triNum(args.kernelOrder + 1) + 1;
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

std::vector<float> makeModel(const Stamp& s, const std::vector<double>& kernSol,
                             const std::pair<cl_long, cl_long> imgSize, const Arguments& args) {
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

void fitKernel(Kernel& k, std::vector<Stamp>& stamps, const Image& tImg, const Image& sImg, ImageMask& mask,
               const cl::Buffer &tImgBuf, const cl::Buffer &sImgBuf, const ClData &clData, const ClStampsData &stampData, const Arguments& args) {
  const int nComp1 = args.nPSF - 1;
  const int nComp2 = triNum(args.kernelOrder + 1);
  const int nBGComp = triNum(args.backgroundOrder + 1);
  const int matSize = nComp1 * nComp2 + nBGComp + 1;
  const int nKernSolComp = args.nPSF * nComp2 + nBGComp + 1;

  // Create buffers
  cl::Buffer fitMatrix(clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * (matSize + 1) * (matSize + 1));
  cl::Buffer weights(clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * nComp2 * stampData.stampCount);
  cl::Buffer solution(clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * nKernSolComp);
  cl::Buffer index(clData.context, CL_MEM_READ_WRITE, sizeof(cl_int) * matSize);

  std::vector<int> index0(matSize, 0);

  int iteration = 0;
  bool check{};

  do
  {
    if (args.verbose && iteration > 0) {
      std::cout << "Re-expanding matrix..." << std::endl;
    }
    
    auto [fittingMatrix0, weight0] = createMatrix(stamps, tImg.axis, args);
    std::vector<double> solution0 = createScProd(stamps, sImg, weight0, args);

    // LU solve
    double d{};
    ludcmp(fittingMatrix0, matSize, index0, d, args);
    lubksb(fittingMatrix0, matSize, index0, solution0);

    k.solution = solution0;
    check = checkFitSolution(k, stamps, tImg, sImg, mask, args);

    iteration++;
  }
  while (check);
}

bool checkFitSolution(const Kernel& k, std::vector<Stamp>& stamps, const Image& tImg,
                      const Image& sImg, ImageMask& mask, const Arguments& args) {
  std::vector<double> ssValues{};

  bool check = false;

  for(Stamp& s : stamps) {
    if(!s.subStamps.empty()) {
      double sig = calcSig(s, k.solution, tImg, sImg, mask, args);

      if(sig == -1) {
        s.subStamps.erase(s.subStamps.begin(), next(s.subStamps.begin()));
        fillStamp(s, tImg, sImg, mask, k, args);
        check = true;
      } else {
        s.stats.chi2 = sig;
        ssValues.push_back(sig);
      }
    }
  }

  double mean = 0.0, stdDev = 0.0;
  sigmaClip(ssValues, mean, stdDev, 10, args);

  if(args.verbose) {
    std::cout << "Mean sig: " << mean << " stdev: " << stdDev << '\n'
              << "    Iterating through stamps with sig >"
              << (mean + args.sigKernFit * stdDev) << std::endl;
  }

  for(Stamp& s : stamps) {
    if(!s.subStamps.empty()) {
      if((s.stats.chi2 - mean) > args.sigKernFit * stdDev) {
        s.subStamps.erase(s.subStamps.begin(), next(s.subStamps.begin()));
        fillStamp(s, tImg, sImg, mask, k, args);
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
