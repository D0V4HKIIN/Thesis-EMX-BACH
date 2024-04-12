#include "bachUtil.h"
#include "mathUtil.h"

double testFit(std::vector<Stamp>& stamps, const std::pair<cl_long, cl_long> &axis, const cl::Buffer &tImgBuf, const cl::Buffer &sImgBuf, ClData& clData, ClStampsData& stampData, const Arguments& args) {
  const int nComp1 = args.nPSF - 1;
  const int nComp2 = triNum(args.kernelOrder + 1);
  const int nBGComp = triNum(args.backgroundOrder + 1);
  const int matSize = nComp1 * nComp2 + nBGComp + 1;
  const int nKernSolComp = args.nPSF * nComp2 + nBGComp + 1;
  cl_int meritsCount = 0;

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
  cl::Buffer meritsCounter(clData.context, CL_MEM_READ_WRITE, sizeof(cl_int));

  clData.queue.enqueueWriteBuffer(meritsCounter, CL_TRUE, 0, sizeof(cl_int), &meritsCount);

  // Create test vec
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl_int> testVecFunc(clData.program, "createTestVec");
  cl::EnqueueArgs testVecEargs(clData.queue, cl::NDRange(clData.bCount, stamps.size()));
  cl::Event testVecEvent = testVecFunc(testVecEargs, stampData.b, testVec, clData.bCount);

  // Create test mat
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl_int> testMatFunc(clData.program, "createTestMat");
  cl::EnqueueArgs testMatEargs(clData.queue, cl::NDRange(clData.qCount, clData.qCount, stamps.size()));
  cl::Event testMatEvent = testMatFunc(testMatEargs, stampData.q, testMat, clData.qCount);

  testVecEvent.wait();
  testMatEvent.wait();

  // LU-solve
  ludcmp(testMat, args.nPSF + 2, stamps.size(), index, vv, clData);
  lubksb(testMat, args.nPSF + 2, stamps.size(), index, testVec, clData);

  // Save kernel sums
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl_int> kernelSumFunc(clData.program, "saveKernelSums");
  cl::EnqueueArgs kernelSumEargs(clData.queue, cl::NDRange(stamps.size()));
  cl::Event kernelSumEvent = kernelSumFunc(kernelSumEargs, testVec, kernelSums, args.nPSF + 2);

  kernelSumEvent.wait();

  double kernelMean, kernelStdev;
  sigmaClip(kernelSums, 0, stamps.size(), &kernelMean, &kernelStdev, 10, clData, args);

  // Fit stamps, generate test stamps
  cl_int testStampCount = 0;
  
  cl::Buffer testStampCountBuf(clData.context, CL_MEM_READ_WRITE, sizeof(cl_int));
  cl::Buffer testStampIndices(clData.context, CL_MEM_READ_WRITE, sizeof(cl_int) * stamps.size());
  clData.queue.enqueueWriteBuffer(testStampCountBuf, CL_TRUE, 0, sizeof(cl_int), &testStampCount);

  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer,
                    cl_double, cl_double, cl_double, cl_int> testStampFunc(clData.program, "genCdTestStamps");
  cl::EnqueueArgs testStampEargs(clData.queue, cl::NDRange(roundUpToMultiple(stamps.size(), 8)), cl::NDRange(8));
  cl::Event testStampEvent = testStampFunc(testStampEargs, kernelSums, testStampIndices, testStampCountBuf,
                                           kernelMean, kernelStdev, args.sigKernFit, stamps.size());

  clData.queue.enqueueReadBuffer(testStampCountBuf, CL_TRUE, 0, sizeof(cl_int), &testStampCount);

  if (testStampCount == 0) {
    return 666;
  }

  // Allocate test stamps, so we have continuous stamp data, since
  // some stamps may be removed
  ClStampsData testStampData{};
  testStampData.subStampCoords = cl::Buffer(clData.context, CL_MEM_READ_WRITE, sizeof(cl_int2) * (2 * args.maxKSStamps) * testStampCount);
  testStampData.currentSubStamps = cl::Buffer(clData.context, CL_MEM_READ_WRITE, sizeof(cl_int) * testStampCount);
  testStampData.subStampCounts = cl::Buffer(clData.context, CL_MEM_READ_WRITE, sizeof(cl_int) * testStampCount);
  testStampData.w = cl::Buffer(clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * testStampCount * clData.wColumns * clData.wRows);
  testStampData.q = cl::Buffer(clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * testStampCount * clData.qCount * clData.qCount);
  testStampData.b = cl::Buffer(clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * testStampCount * clData.bCount);
  testStampData.stampCount = testStampCount;

  std::vector<cl::Event> testEvents{};

  // Copy sub-stamp coordinates
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl_int> copySsCoordsFunc(clData.program, "copyTestSubStampsCoords");
  cl::EnqueueArgs copySsCoordsEargs(clData.queue, cl::NDRange(2 * args.maxKSStamps, testStampCount));
  testEvents.push_back(copySsCoordsFunc(copySsCoordsEargs, stampData.subStampCoords, testStampIndices, testStampData.subStampCoords, 2 * args.maxKSStamps));

  // Copy substamp counts
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> copySsCountsFunc(clData.program, "copyTestSubStampsCounts");
  cl::EnqueueArgs copySsCountsEargs(clData.queue, cl::NDRange(testStampCount));
  testEvents.push_back(copySsCountsFunc(copySsCountsEargs, stampData.currentSubStamps, stampData.subStampCounts, testStampIndices,
                                        testStampData.currentSubStamps, testStampData.subStampCounts));

  // Copy W
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl_int, cl_int> copyWFunc(clData.program, "copyTestStampsW");
  cl::EnqueueArgs copyWEargs(clData.queue, cl::NDRange(clData.wColumns, clData.wRows, testStampCount));
  testEvents.push_back(copyWFunc(copyWEargs, stampData.w, testStampIndices, testStampData.w, clData.wRows, clData.wColumns));

  // Copy Q
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl_int> copyQFunc(clData.program, "copyTestStampsQ");
  cl::EnqueueArgs copyQEargs(clData.queue, cl::NDRange(clData.qCount, clData.qCount, testStampCount));
  testEvents.push_back(copyQFunc(copyQEargs, stampData.q, testStampIndices, testStampData.q, clData.qCount));

  // Copy B
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl_int> copyBFunc(clData.program, "copyTestStampsB");
  cl::EnqueueArgs copyBEargs(clData.queue, cl::NDRange(clData.bCount, testStampCount));
  testEvents.push_back(copyBFunc(copyBEargs, stampData.b, testStampIndices, testStampData.b, clData.bCount));

  cl::Event::waitForEvents(testEvents);

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

  // TEMP: transfer back to GPU
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
  calcSigs(tImgBuf, sImgBuf, axis, model, testKernSol, merits, testStampData, clData, args);

  // Remove bad merits
  static constexpr int badLocalSize = 16;
  cl::Buffer cleanMerits(clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * testStampCount);

  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::LocalSpaceArg, cl_int> badMeritsFunc(clData.program, "removeBadSigs");
  cl::EnqueueArgs badMeritsEargs(clData.queue, cl::NDRange(roundUpToMultiple(testStampCount, badLocalSize)), cl::NDRange(badLocalSize));
  cl::Event badMeritsEvent = badMeritsFunc(badMeritsEargs, merits, cleanMerits, meritsCounter, cl::Local(badLocalSize * sizeof(cl_double)), testStampCount);

  badMeritsEvent.wait();

  clData.queue.enqueueReadBuffer(meritsCounter, CL_TRUE, 0, sizeof(cl_int), &meritsCount);

  if (meritsCount == 0) {
    return 666;
  }

  double meritMean;
  double meritStdDev;
  sigmaClip(cleanMerits, 0, meritsCount, &meritMean, &meritStdDev, 10, clData, args);

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
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                    cl_int, cl_int, cl_int, cl_int> weightFunc(clData.program, "createMatrixWeights");
  cl::EnqueueArgs weightEargs(clData.queue, cl::NDRange(nComp2, stampData.stampCount));
  cl::Event weightEvent = weightFunc(weightEargs, stampData.subStampCoords, stampData.currentSubStamps, stampData.subStampCounts, clData.cd.kernelXy,
                                     weights, imgSize.first, imgSize.second, 2 * args.maxKSStamps, nComp2);

  weightEvent.wait();

  // Create matrix
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                    cl_int, cl_int, cl_int, cl_int, cl_int,
                    cl_int, cl_int, cl_int> matrixFunc(clData.program, "createMatrix");
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
  
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                    cl::Buffer, cl::Buffer, cl::Buffer,
                    cl_int, cl_int, cl_int, cl_int, cl_int,
                    cl_int, cl_int, cl_int, cl_int, cl_int> prodFunc(clData.program, "createScProd");
  cl::EnqueueArgs prodEargs(clData.queue, cl::NDRange(nKernSolComp));
  cl::Event prodEvent = prodFunc(prodEargs, img, weights, stampData.b, stampData.w,
                                 stampData.subStampCoords, stampData.currentSubStamps, stampData.subStampCounts, res,
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

void calcSigs(const cl::Buffer &tImgBuf, const cl::Buffer &sImgBuf, const std::pair<cl_long, cl_long> &axis,
              const cl::Buffer &model, const cl::Buffer &kernSol, const cl::Buffer &sigma,
              const ClStampsData &stampData, const ClData &clData, const Arguments& args) {
  static constexpr int localSize = 32;

  int reduceCount = (args.fSStampWidth * args.fSStampWidth + localSize - 1) / localSize;
  int stampCount = stampData.stampCount;

  // Create buffers
  cl::Buffer bg(clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * stampCount);
  cl::Buffer sigTemp1(clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * stampCount * reduceCount);
  cl::Buffer sigTemp2(clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * stampCount * reduceCount);
  cl::Buffer sigCount1(clData.context, CL_MEM_READ_WRITE, sizeof(cl_int) * stampCount * reduceCount);
  cl::Buffer sigCount2(clData.context, CL_MEM_READ_WRITE, sizeof(cl_int) * stampCount * reduceCount);

  // Create bg
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                    cl_int, cl_int, cl_int, cl_int, cl_int> bgFunc(clData.program, "calcSigBg");
  cl::EnqueueArgs bgEargs(clData.queue, cl::NDRange(stampCount));
  cl::Event bgEvent = bgFunc(bgEargs, kernSol, stampData.subStampCoords, stampData.currentSubStamps, stampData.subStampCounts, bg,
                             2 * args.maxKSStamps, axis.first, axis.second, args.backgroundOrder,
                             (args.nPSF - 1) * triNum(args.kernelOrder + 1) + 1);

  // Create model
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                    cl_int, cl_int, cl_int, cl_int, cl_int,
                    cl_int, cl_int, cl_int> modelFunc(clData.program, "makeModel");
  cl::EnqueueArgs modelEargs(clData.queue, cl::NDRange(args.fSStampWidth * args.fSStampWidth, stampCount));
  cl::Event modelEvent = modelFunc(modelEargs, stampData.w, kernSol, stampData.subStampCoords, stampData.currentSubStamps, stampData.subStampCounts,
                                   model, args.nPSF, args.kernelOrder, clData.wRows, clData.wColumns, 2 * args.maxKSStamps,
                                   axis.first, axis.second, args.fSStampWidth * args.fSStampWidth);

  bgEvent.wait();
  modelEvent.wait();

  // Create sigmas
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                    cl::Buffer, cl::Buffer, cl::Buffer,
                    cl::Buffer, cl::Buffer, cl::Buffer, cl::LocalSpaceArg,
                    cl_int, cl_int, cl_int, cl_int, cl_int> sigmaFunc(clData.program, "calcSig");
  cl::EnqueueArgs sigmaEargs(clData.queue, cl::NDRange(reduceCount * localSize, stampCount), cl::NDRange(localSize, 1));
  cl::Event sigmaEvent = sigmaFunc(sigmaEargs, model, bg, tImgBuf, sImgBuf,
                                   stampData.subStampCoords, stampData.currentSubStamps, stampData.subStampCounts,
                                   sigTemp1, sigCount1, clData.maskBuf, cl::Local(localSize * sizeof(cl_double)), axis.first, args.fSStampWidth,
                                   2 * args.maxKSStamps, args.fSStampWidth * args.fSStampWidth, reduceCount);

  sigmaEvent.wait();

  // Reduce
  bool isFirst = true;
  int count = reduceCount;
  cl::Buffer *sigIn = &sigTemp1;
  cl::Buffer *sigOut = &sigTemp2;
  cl::Buffer *sigCountIn = &sigCount1;
  cl::Buffer *sigCountOut = &sigCount2;

  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::LocalSpaceArg,
                    cl_int, cl_int> reduceFunc(clData.program, "reduceSig");

  while (count > 1 || isFirst) {
    int nextCount = (count + localSize - 1) / localSize;

    cl::EnqueueArgs reduceEargs(clData.queue, cl::NDRange(roundUpToMultiple(count, localSize), stampCount), cl::NDRange(localSize, 1));
    cl::Event reduceEvent = reduceFunc(reduceEargs, *sigIn, *sigCountIn, *sigOut, *sigCountOut, cl::Local(localSize * sizeof(cl_double)), count, nextCount);

    reduceEvent.wait();

    count = nextCount;
    std::swap(sigIn, sigOut);
    std::swap(sigCountIn, sigCountOut);

    isFirst = false;
  }

  // Copy buffer
  cl::Event copyEvent{};
  clData.queue.enqueueCopyBuffer(*sigIn, sigma, 0, 0, sizeof(cl_double) * stampCount, nullptr, &copyEvent);

  copyEvent.wait();
}

void fitKernel(Kernel& k, std::vector<Stamp>& stamps, const Image& tImg, const Image& sImg,
               const cl::Buffer &tImgBuf, const cl::Buffer &sImgBuf, ClData &clData, const ClStampsData &stampData, const Arguments& args) {
  const int nComp1 = args.nPSF - 1;
  const int nComp2 = triNum(args.kernelOrder + 1);
  const int nBGComp = triNum(args.backgroundOrder + 1);
  const int matSize = nComp1 * nComp2 + nBGComp + 1;
  const int nKernSolComp = args.nPSF * nComp2 + nBGComp + 1;

  // Create buffers
  cl::Buffer fitMatrix(clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * (matSize + 1) * (matSize + 1));
  cl::Buffer weights(clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * nComp2 * stampData.stampCount);
  clData.kernel.solution = cl::Buffer(clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * nKernSolComp);
  cl::Buffer index(clData.context, CL_MEM_READ_WRITE, sizeof(cl_int) * (matSize + 1));
  cl::Buffer vv(clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * (matSize + 1));

  std::vector<int> index0(matSize, 0);

  int iteration = 0;
  bool check{};

  do
  {
    if (args.verbose && iteration > 0) {
      std::cout << "Re-expanding matrix..." << std::endl;
    }
    
    // Create matrix
#if false
    createMatrix(fitMatrix, weights, clData, stampData, tImg.axis, args);
    createScProd(solution, weights, sImgBuf, tImg.axis, clData, stampData, args);
    
    // TEMP: transfer back to CPU
    std::vector<std::vector<double>> fittingMatrixCpu(matSize + 1, std::vector<double>(matSize + 1));
    std::vector<cl_double> solutionCpu(nKernSolComp);
    std::vector<cl_double> fittingMatrixTemp((matSize + 1) * (matSize + 1));

    clData.queue.enqueueReadBuffer(fitMatrix, CL_TRUE, 0, sizeof(cl_double) * fittingMatrixTemp.size(), fittingMatrixTemp.data());
    clData.queue.enqueueReadBuffer(solution, CL_TRUE, 0, sizeof(cl_double) * solutionCpu.size(), solutionCpu.data());
    
    for (int i = 0; i <= matSize; i++) {
      for (int j = 0; j <= matSize; j++) {
        fittingMatrixCpu[i][j] = fittingMatrixTemp[i * (matSize + 1) + j];
      }
    }
#else
    auto [fittingMatrix0, weight0] = createMatrix(stamps, tImg.axis, args);
    std::vector<double> solution0 = createScProd(stamps, sImg, weight0, args);

    std::vector<std::vector<double>> fittingMatrixCpu = std::move(fittingMatrix0);
    std::vector<double>solutionCpu = std::move(solution0);
#endif

    // LU solve
#if false
    // TEMP: transfer matrix to GPU
    std::vector<cl_double> flatFitMatrix{};

    for (auto& row : fittingMatrixCpu) {
      for (double value : row) {
        flatFitMatrix.push_back(value);
      }
    }

    clData.queue.enqueueWriteBuffer(fitMatrix, CL_TRUE, 0, sizeof(cl_double) * flatFitMatrix.size(), flatFitMatrix.data());

    // TEMP: transfer solution to GPU
    clData.queue.enqueueWriteBuffer(solution, CL_TRUE, 0, sizeof(cl_double) * solutionCpu.size(), solutionCpu.data());

    ludcmp(fitMatrix, matSize + 1, 1, index, vv, clData);
    lubksb(fitMatrix, matSize + 1, 1, index, solution, clData);

    // TEMP: transfer solution back to CPU
    clData.queue.enqueueReadBuffer(solution, CL_TRUE, 0, sizeof(cl_double) * solutionCpu.size(), solutionCpu.data());

#else
    double d{};
    ludcmp(fittingMatrixCpu, matSize, index0, d, args);
    lubksb(fittingMatrixCpu, matSize, index0, solutionCpu);
#endif

    k.solution = solutionCpu;
    
    // TEMP: transfer kernel solution to GPU
    clData.queue.enqueueWriteBuffer(clData.kernel.solution, CL_TRUE, 0, sizeof(cl_double) * solutionCpu.size(), solutionCpu.data());

    check = checkFitSolution(k, stamps, tImg, sImg, clData, stampData, tImgBuf, sImgBuf, clData.kernel.solution, args);

    iteration++;
  }
  while (check);
}

bool checkFitSolution(const Kernel& k, std::vector<Stamp>& stamps, const Image& tImg,
                      const Image& sImg,  const ClData &clData, const ClStampsData &stampData, const cl::Buffer &tImgBuf, const cl::Buffer &sImgBuf, const cl::Buffer &kernSol, const Arguments& args) {
  cl_int chi2Count = 0;
  
  // Create buffers
  cl::Buffer model(clData.context, CL_MEM_READ_WRITE, sizeof(cl_float) * stampData.stampCount * args.fSStampWidth * args.fSStampWidth);
  cl::Buffer sigmaVals(clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * stampData.stampCount);
  cl::Buffer chi2(clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * stampData.stampCount);
  cl::Buffer invalidatedSubStampsBuf(clData.context, CL_MEM_READ_WRITE, sizeof(cl_uchar) * stampData.stampCount);
  cl::Buffer chi2Counter(clData.context, CL_MEM_READ_WRITE, sizeof(cl_int));

  clData.queue.enqueueWriteBuffer(chi2Counter, CL_TRUE, 0, sizeof(cl_int), &chi2Count);

  // Calculate sigmas
  calcSigs(tImgBuf, sImgBuf, tImg.axis, model, kernSol, sigmaVals, stampData, clData, args);

  // Find bad sub-stamps
  static constexpr int badLocalSize = 16;
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::LocalSpaceArg, cl_int> badSsFunc(clData.program, "checkBadSubStamps");
  cl::EnqueueArgs badSsEargs(clData.queue, cl::NDRange(roundUpToMultiple(stampData.stampCount, badLocalSize)), cl::NDRange(badLocalSize));
  cl::Event badSsEvent = badSsFunc(badSsEargs, sigmaVals, stampData.subStampCounts,
                                   chi2, invalidatedSubStampsBuf, stampData.currentSubStamps, chi2Counter,
                                   cl::Local(badLocalSize * sizeof(cl_double)), stampData.stampCount);

  badSsEvent.wait();

  clData.queue.enqueueReadBuffer(chi2Counter, CL_TRUE, 0, sizeof(cl_int), &chi2Count);

  // Read which sub-stamps are bad
  std::vector<cl_uchar> invalidatedSubStamps(stampData.stampCount);
  clData.queue.enqueueReadBuffer(invalidatedSubStampsBuf, CL_TRUE, 0, sizeof(cl_uchar) * invalidatedSubStamps.size(), invalidatedSubStamps.data());

  // Remove the bad sub-stamps
  bool check = false;
  removeBadSubStamps(&check, stampData, stamps, invalidatedSubStamps, tImg, sImg, sImgBuf, tImgBuf, k, clData, args);

  // Sigma clip
  double mean = 0.0;
  double stdDev = 0.0;
  sigmaClip(chi2, 0, chi2Count, &mean, &stdDev, 10, clData, args);

  // Find bad sub-stamps based on the sigma clip
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl_double, cl_double, cl_double> badSsClipFunc(clData.program, "checkBadSubStampsFromSigmaClip");
  cl::EnqueueArgs badSsClipEargs(clData.queue, cl::NDRange(stampData.stampCount));
  cl::Event badSsClipEvent = badSsClipFunc(badSsClipEargs, sigmaVals, stampData.subStampCounts, invalidatedSubStampsBuf, stampData.currentSubStamps, mean, stdDev, args.sigKernFit);

  badSsClipEvent.wait();
  
  // Read which sub-stamps are bad (again)
  clData.queue.enqueueReadBuffer(invalidatedSubStampsBuf, CL_TRUE, 0, sizeof(cl_uchar) * invalidatedSubStamps.size(), invalidatedSubStamps.data());

  // Remove the bad sub-stamps
  removeBadSubStamps(&check, stampData, stamps, invalidatedSubStamps, tImg, sImg, sImgBuf, tImgBuf, k, clData, args);

  return check;
}

void removeBadSubStamps(bool *check, const ClStampsData &stampData, std::vector<Stamp> &stamps, const std::vector<cl_uchar> &invalidatedSubStamps, const Image &tImg, const Image &sImg,
                        const cl::Buffer &sImgBuf, const cl::Buffer &tImgBuf, const Kernel &k, const ClData &clData, const Arguments &args) {
  for (int i = 0; i < invalidatedSubStamps.size(); i++) {
    if (invalidatedSubStamps[i] == 1) {
      // Count concecutive invalidated sub-stamps, so that
      // multiple new sub-stamps can be filled at the same time
      int firstIndex = i;
      int count = 1;

      while (firstIndex + count < invalidatedSubStamps.size() && invalidatedSubStamps[i + 1] == 1) {
        count++;
        i++;
      }

      // TEMP: delete bad sub-stamps on CPU
      for (int j = 0; j < count; j++) {
        Stamp &s = stamps[firstIndex + j];
        s.subStamps.erase(s.subStamps.begin(), std::next(s.subStamps.begin()));
      }
      
      fillStamps(stamps, tImg, sImg, tImgBuf, sImgBuf, firstIndex, count, k, clData, stampData, args);
      *check = true;
    }
  }
}
