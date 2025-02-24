#include "bachUtil.h"

#include <algorithm>
#include <numeric>

#include "mathUtil.h"

int timeDiff(std::chrono::time_point<std::chrono::steady_clock> end,
             std::chrono::time_point<std::chrono::steady_clock> start) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
      .count();
}

void ludcmp(const cl::Buffer& matrix, int matrixSize, int stampCount,
            const cl::Buffer& index, const cl::Buffer& vv,
            const ClData& clData) {
  // Find big values
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl_int> bigFunc(clData.program,
                                                            "ludcmpBig");
  cl::EnqueueArgs bigEargs(clData.queue, cl::NDRange(matrixSize, stampCount));
  cl::Event bigEvent = bigFunc(bigEargs, matrix, vv, matrixSize);

  bigEvent.wait();

  // Rest of LU-decomposition
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl_int> restFunc(
      clData.program, "ludcmpRest");
  cl::EnqueueArgs restEargs(clData.queue, cl::NDRange(stampCount));
  cl::Event restEvent = restFunc(restEargs, vv, matrix, index, matrixSize);

  restEvent.wait();
}

void lubksb(const cl::Buffer& matrix, int matrixSize, int stampCount,
            const cl::Buffer& index, const cl::Buffer& result,
            const ClData& clData) {
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl_int> func(
      clData.program, "lubksb");
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

double makeKernel(const cl::Buffer& kernel, const cl::Buffer& kernSolution,
                  const std::pair<cl_int, cl_int>& imgSize, const int x,
                  const int y, const Arguments& args, const ClData& clData) {
  double hWidth = 0.5 * imgSize.first;
  double hHeight = 0.5 * imgSize.second;

  double xf = (x - hWidth) / hWidth;
  double yf = (y - hHeight) / hHeight;

  static constexpr int localCount = 32;

  // Create buffers
  cl::Buffer kernCoeffs(clData.context, CL_MEM_READ_WRITE,
                        sizeof(cl_double) * args.nPSF);
  cl::Buffer kernelSum(
      clData.context, CL_MEM_READ_WRITE,
      sizeof(cl_double) * args.fKernelWidth * args.fKernelWidth);
  cl::Buffer kernelSum2(
      clData.context, CL_MEM_READ_WRITE,
      sizeof(cl_double) *
          ((args.fKernelWidth * args.fKernelWidth + localCount - 1) /
           localCount));

  // Create coefficients
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl_int, cl_int, cl_double,
                    cl_double>
      coeffFunc(clData.program, "makeKernelCoeffs");
  cl::EnqueueArgs coeffEargs(clData.queue, cl::NDRange(args.nPSF));
  cl::Event coeffEvent =
      coeffFunc(coeffEargs, kernSolution, kernCoeffs, args.kernelOrder,
                triNum(args.kernelOrder + 1), xf, yf);

  coeffEvent.wait();

  // Create kernel
  static constexpr int kernelLocalSize = 16;
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::LocalSpaceArg,
                    cl_int, cl_int>
      kernelFunc(clData.program, "makeKernel");
  cl::EnqueueArgs kernelEargs(
      clData.queue,
      cl::NDRange(roundUpToMultiple(args.fKernelWidth * args.fKernelWidth,
                                    kernelLocalSize)),
      cl::NDRange(kernelLocalSize));
  cl::Event kernelEvent =
      kernelFunc(kernelEargs, kernCoeffs, clData.kernel.vec, kernel,
                 cl::Local(kernelLocalSize * sizeof(cl_double)), args.nPSF,
                 args.fKernelWidth);

  kernelEvent.wait();

  // Sum kernel
  cl::Event copyEvent{};
  clData.queue.enqueueCopyBuffer(
      kernel, kernelSum, 0, 0,
      sizeof(cl_double) * args.fKernelWidth * args.fKernelWidth, nullptr,
      &copyEvent);

  copyEvent.wait();

  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::LocalSpaceArg, cl_int> sumFunc(
      clData.program, "sumKernel");
  int sumCount = args.fKernelWidth * args.fKernelWidth;

  cl::Buffer* src = &kernelSum;
  cl::Buffer* dst = &kernelSum2;

  while(sumCount > 1) {
    cl::EnqueueArgs sumEargs(
        clData.queue, cl::NDRange(roundUpToMultiple(sumCount, localCount)),
        cl::NDRange(localCount));
    cl::Event sumEvent =
        sumFunc(sumEargs, *src, *dst, cl::Local(localCount * sizeof(cl_double)),
                sumCount);

    sumEvent.wait();

    sumCount = (sumCount + localCount - 1) / localCount;
    std::swap(src, dst);
  }

  // Transfer sum to CPU
  cl_double sumKernel = 0.0;
  clData.queue.enqueueReadBuffer(*src, CL_TRUE, 0, sizeof(cl_double),
                                 &sumKernel);

  return sumKernel;
}

double makeKernel(Kernel& kern, const std::pair<cl_int, cl_int>& imgSize,
                  const int x, const int y, const Arguments& args) {
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
