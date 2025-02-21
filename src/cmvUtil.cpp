#include "bachUtil.h"
#include "mathUtil.h"

void initFillStamps(std::vector<Stamp>& stamps,
                    const std::pair<cl_int, cl_int>& axis,
                    const cl::Buffer& tImgBuf, const cl::Buffer& sImgBuf,
                    const Kernel& k, ClData& clData, ClStampsData& stampData,
                    const Arguments& args) {
  clData.wColumns = args.fSStampWidth * args.fSStampWidth;
  clData.wRows = args.nPSF + triNum(args.backgroundOrder + 1);
  clData.qCount = args.nPSF + 2;
  clData.bCount = args.nPSF + 2;

  // Create buffers
  stampData.w = cl::Buffer(
      clData.context, CL_MEM_READ_WRITE,
      sizeof(cl_double) * clData.wRows * clData.wColumns * stamps.size());
  stampData.q = cl::Buffer(
      clData.context, CL_MEM_READ_WRITE,
      sizeof(cl_double) * clData.qCount * clData.qCount * stamps.size());
  stampData.b = cl::Buffer(clData.context, CL_MEM_READ_WRITE,
                           sizeof(cl_double) * clData.bCount * stamps.size());

  // TEMP: initialize vectors for W, Q and B
  for(Stamp& stamp : stamps) {
    stamp.W = std::vector<std::vector<double>>(
        clData.wRows, std::vector<double>(clData.wColumns));
    stamp.Q = std::vector<std::vector<double>>(
        clData.qCount, std::vector<double>(clData.qCount));
    stamp.B = std::vector<double>(clData.bCount);
  }

  fillStamps(stamps, axis, tImgBuf, sImgBuf, 0, stamps.size(), k, clData,
             stampData, args);
}

void fillStamps(std::vector<Stamp>& stamps,
                const std::pair<cl_int, cl_int>& axis,
                const cl::Buffer& tImgBuf, const cl::Buffer& sImgBuf,
                int stampOffset, int stampCount, const Kernel& k,
                const ClData& clData, const ClStampsData& stampData,
                const Arguments& args) {
  /* Fills Substamp with gaussian basis convolved images around said substamp
   * and calculates CMV.
   */

  // Convolve stamps on Y
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                    cl::Buffer, cl_int, cl_int, cl_int, cl_int, cl_int>
      yConvFunc(clData.program, "convStampY");
  cl::EnqueueArgs yConvEargs(
      clData.queue, cl::NDRange(0, 0, stampOffset),
      cl::NDRange((2 * (args.hSStampWidth + args.hKernelWidth) + 1) *
                      (2 * args.hSStampWidth + 1),
                  clData.gaussCount, stampCount),
      cl::NullRange);
  cl::Event yConvEvent = yConvFunc(
      yConvEargs, tImgBuf, stampData.subStampCoords, stampData.currentSubStamps,
      stampData.subStampCounts, clData.kernel.filterY, clData.cmv.yConvTmp,
      args.fKernelWidth, args.fSStampWidth, axis.first, clData.gaussCount,
      2 * args.maxKSStamps);

  yConvEvent.wait();

  // Convolve stamps on X
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl_int, cl_int, cl_int,
                    cl_int, cl_int>
      xConvFunc(clData.program, "convStampX");
  cl::EnqueueArgs xConvEargs(clData.queue, cl::NDRange(0, 0, stampOffset),
                             cl::NDRange(args.fSStampWidth * args.fSStampWidth,
                                         clData.gaussCount, stampCount),
                             cl::NullRange);
  cl::Event xConvEvent =
      xConvFunc(xConvEargs, clData.cmv.yConvTmp, clData.kernel.filterX,
                stampData.w, args.fKernelWidth, args.fSStampWidth, clData.wRows,
                clData.wColumns, clData.gaussCount);

  xConvEvent.wait();

  // Subtract for odd
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl_int, cl_int> oddConvFunc(
      clData.program, "convStampOdd");
  cl::EnqueueArgs oddConvEargs(
      clData.queue, cl::NDRange(0, 1, stampOffset),
      cl::NDRange(args.fSStampWidth * args.fSStampWidth, clData.gaussCount - 1,
                  stampCount),
      cl::NullRange);
  cl::Event oddConvEvent =
      oddConvFunc(oddConvEargs, clData.kernel.xy, stampData.w, clData.wRows,
                  clData.wColumns);

  oddConvEvent.wait();

  // Compute background
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                    cl_int, cl_int, cl_int, cl_int, cl_int, cl_int, cl_int>
      bgConvFunc(clData.program, "convStampBg");
  cl::EnqueueArgs bgConvEargs(
      clData.queue, cl::NDRange(0, 0, stampOffset),
      cl::NDRange(clData.wColumns, clData.wRows - clData.gaussCount,
                  stampCount),
      cl::NullRange);
  cl::Event bgConvEvent = bgConvFunc(
      bgConvEargs, stampData.subStampCoords, stampData.currentSubStamps,
      stampData.subStampCounts, clData.bg.xy, stampData.w, axis.first,
      axis.second, args.fSStampWidth, clData.wRows, clData.wColumns,
      clData.gaussCount, 2 * args.maxKSStamps);

  bgConvEvent.wait();

  // TEMP: move back w to CPU
  std::vector<cl_double> wGpu(clData.wRows * clData.wColumns * stampCount);
  clData.queue.enqueueReadBuffer(
      stampData.w, CL_TRUE,
      sizeof(cl_double) * stampOffset * clData.wRows * clData.wColumns,
      sizeof(cl_double) * wGpu.size(), wGpu.data());

  // TEMP: replace w with GPU data
  for(int i = 0; i < stampCount; i++) {
    Stamp& s = stamps[stampOffset + i];

    for(int j = 0; j < clData.wRows; j++) {
      for(int k = 0; k < clData.wColumns; k++) {
        s.W[j][k] =
            wGpu[i * clData.wRows * clData.wColumns + j * clData.wColumns + k];
      }
    }
  }

  // Create Q
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl_int, cl_int, cl_int, cl_int,
                    cl_int>
      qFunc(clData.program, "createQ");
  cl::EnqueueArgs qEargs(clData.queue, cl::NDRange(0, 0, stampOffset),
                         cl::NDRange(clData.qCount, clData.qCount, stampCount),
                         cl::NullRange);
  cl::Event qEvent =
      qFunc(qEargs, stampData.w, stampData.q, clData.wRows, clData.wColumns,
            clData.qCount, clData.qCount, args.fSStampWidth);

  // Create B
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                    cl::Buffer, cl_int, cl_int, cl_int, cl_int, cl_int, cl_int>
      bFunc(clData.program, "createB");
  cl::EnqueueArgs bEargs(clData.queue, cl::NDRange(0, stampOffset),
                         cl::NDRange(clData.bCount, stampCount), cl::NullRange);
  cl::Event bEvent =
      bFunc(bEargs, stampData.subStampCoords, stampData.currentSubStamps,
            stampData.subStampCounts, sImgBuf, stampData.w, stampData.b,
            clData.wRows, clData.wColumns, clData.bCount, args.fSStampWidth,
            2 * args.maxKSStamps, axis.first);

  qEvent.wait();
  bEvent.wait();

  // TEMP: transfer the data back to the CPU
  std::vector<cl_double> gpuQ(clData.qCount * clData.qCount * stampCount);
  std::vector<cl_double> gpuB(clData.bCount * stampCount);

  clData.queue.enqueueReadBuffer(
      stampData.q, CL_TRUE,
      sizeof(cl_double) * stampOffset * clData.qCount * clData.qCount,
      sizeof(cl_double) * gpuQ.size(), gpuQ.data());
  clData.queue.enqueueReadBuffer(
      stampData.b, CL_TRUE, sizeof(cl_double) * stampOffset * clData.bCount,
      sizeof(cl_double) * gpuB.size(), gpuB.data());

  // TEMP: put data back in Q
  for(int i = 0; i < stampCount; i++) {
    Stamp& s = stamps[stampOffset + i];

    for(int j = 0; j < clData.qCount; j++) {
      for(int k = 0; k < clData.qCount; k++) {
        s.Q[j][k] =
            gpuQ[i * clData.qCount * clData.qCount + j * clData.qCount + k];
      }
    }
  }

  // TEMP: put data back in B
  for(int i = 0; i < stampCount; i++) {
    Stamp& s = stamps[stampOffset + i];

    for(int j = 0; j < clData.bCount; j++) {
      s.B[j] = gpuB[i * clData.bCount + j];
    }
  }
}
