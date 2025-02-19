#include "bachUtil.h"
#include "mathUtil.h"
#include <cassert>
#include <iostream>

void identifySStamps(const std::pair<cl_int, cl_int> &axis, const Arguments& args, ClData& clData) {
  std::cout << "Identifying sub-stamps..." << std::endl;

  if (args.verbose) std::cout << "calcStats (template)" << std::endl;
  calcStats(axis, args, clData.tImgBuf, clData.tmpl, clData);
  if (args.verbose) std::cout << "calcStats (science)" << std::endl;
  calcStats(axis, args, clData.sImgBuf, clData.sci, clData);

  if (args.verbose) std::cout << "findSStamps (template)" << std::endl;
  findSStamps(axis, true, args, clData.tImgBuf, clData.tmpl, clData);
  if (args.verbose) std::cout << "findSStamps (science)" << std::endl;
  findSStamps(axis, false, args, clData.sImgBuf, clData.sci, clData);
}

void createStamps(std::vector<Stamp>& stamps, const int w, const int h, ClStampsData& stampsData, const ClData& clData, const Arguments& args) {
  cl::EnqueueArgs eargsBounds{clData.queue, cl::NDRange(args.stampsx * args.stampsy)};
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl_int, cl_int, cl_int, cl_int, cl_int>
  boundsFunc(clData.program, "createStampBounds");

  cl::Event boundsEvent{
    boundsFunc(eargsBounds,
                stampsData.stampCoords, stampsData.stampSizes,
                args.stampsx, args.stampsy, args.fStampWidth,
                w, h)};
  boundsEvent.wait();
  stampsData.stampCount = args.stampsx * args.stampsy;
}

// why is this not a void function?
cl_int findSStamps(const std::pair<cl_int, cl_int> &axis, const bool isTemplate, const Arguments& args, const cl::Buffer& imgBuf, const ClStampsData& stampsData, const ClData& clData) {
  auto [imgW, imgH] = axis;

  cl::size_type nStamps{static_cast<cl::size_type>(args.stampsx) * static_cast<cl::size_type>(args.stampsy)};

  ImageMasks badMask = ImageMasks::ALL & ~ImageMasks::OK_CONV;
  ImageMasks badPixelMask, skipMask;

  if (isTemplate) {
    badMask &= ~(ImageMasks::BAD_PIXEL_S | ImageMasks::SKIP_S);
    badPixelMask = ImageMasks::BAD_PIXEL_T;
    skipMask     = ImageMasks::SKIP_T;
  }
  else {
    badMask &= ~(ImageMasks::BAD_PIXEL_T | ImageMasks::SKIP_T);
    badPixelMask = ImageMasks::BAD_PIXEL_S;
    skipMask     = ImageMasks::SKIP_S;
  }
  
  cl_int maxSStamps{2 * args.maxKSStamps};

  constexpr int localSize{1};

  cl::EnqueueArgs eargsFindSStamps(clData.queue, cl::NDRange(roundUpToMultiple(nStamps, localSize)), cl::NDRange(localSize));
  cl::KernelFunctor<cl::Buffer, cl::Buffer,
                    cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                    cl::Buffer, cl::Buffer, cl::Buffer,
                    cl_double, cl_double,
                    cl_int, cl_int, cl_int, 
                    cl_int, cl_int,
                    cl_ushort, cl_ushort, cl_ushort,
                    cl::LocalSpaceArg, cl::LocalSpaceArg> 
  findSStampsFunc{clData.program, "findSubStamps"};
  
  cl::Event findSStampsEvent{findSStampsFunc(eargsFindSStamps, 
                  imgBuf, clData.maskBuf,
                  stampsData.stampCoords, stampsData.stampSizes,
                  stampsData.stats.skyEsts, stampsData.stats.fwhms,
                  stampsData.subStampCoords, stampsData.subStampValues, stampsData.subStampCounts,
                  args.threshHigh, args.threshKernFit, imgW,
                  args.fStampWidth, args.hSStampWidth, maxSStamps, static_cast<cl_int>(nStamps),
                  static_cast<cl_ushort>(badMask),
                  static_cast<cl_ushort>(badPixelMask),
                  static_cast<cl_ushort>(skipMask),
                  cl::Local(sizeof(cl_int2) * maxSStamps * localSize),
                  cl::Local(sizeof(cl_double) * maxSStamps * localSize))};

  findSStampsEvent.wait();

  if(args.verbose) {  
    std::vector<cl_int> sstampCounts(nStamps);
    clData.queue.enqueueReadBuffer(stampsData.subStampCounts, CL_TRUE, 0, sizeof(cl_int)    * sstampCounts.size(), &sstampCounts[0]);
    
    for (int i{0}; i < nStamps; i++) {
      if(sstampCounts[i] == 0) {
        std::cout << "No suitable substamps found in stamp " << i << std::endl;
      }   
      else {
        std::cout << "Added " << sstampCounts[i]
                  << " substamps to stamp " << i << std::endl;
      }
    }
  }
  return 0;
}

void removeEmptyStamps(const Arguments& args, ClStampsData& stampsData, const ClData& clData) {
  
  int maxSStamps{2 * args.maxKSStamps};
  
  cl::size_type nStamps{static_cast<cl::size_type>(args.stampsx * args.stampsy)};
  cl::size_type paddedNStamps{static_cast<cl::size_type>(leastGreaterPow2(args.stampsx * args.stampsy))};

  cl::Buffer filteredStampCoords{clData.context, CL_MEM_READ_WRITE, sizeof(cl_int2) * nStamps};
  cl::Buffer filteredStampSizes{clData.context, CL_MEM_READ_WRITE, sizeof(cl_int2) * nStamps};
  cl::Buffer filteredSkyEsts{clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * nStamps};
  cl::Buffer filteredFwhms{clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * nStamps};
  cl::Buffer filteredSubStampCoords{clData.context, CL_MEM_READ_WRITE, sizeof(cl_int2) * maxSStamps * nStamps};
  cl::Buffer filteredSubStampValues{clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * maxSStamps * nStamps};
  cl::Buffer filteredSubStampCounts{clData.context, CL_MEM_READ_WRITE, sizeof(cl_int) * maxSStamps * nStamps};

  cl::Buffer keepCounter{clData.context, CL_MEM_READ_WRITE, sizeof(cl_int)};
  cl::Buffer keepIndeces{clData.context, CL_MEM_READ_WRITE, sizeof(cl_int) * paddedNStamps};
  
  cl::EnqueueArgs eargsMark{clData.queue,cl::NDRange{nStamps}};
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer>
  markFunc(clData.program, "markStampsToKeep");

  cl::EnqueueArgs eargsSort{clData.queue,cl::NDRange{paddedNStamps}};
  cl::KernelFunctor<cl::Buffer, cl::Buffer>
  padFunc(clData.program, "padMarks");

  cl::KernelFunctor<cl::Buffer, cl_int, cl_int>
  sortFunc(clData.program, "sortMarks");

  cl::EnqueueArgs eargsRemove{clData.queue,cl::NDRange{nStamps}};  
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, 
                    cl::Buffer, cl::Buffer, cl::Buffer,
                    cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                    cl::Buffer, cl::Buffer, cl::Buffer,
                    cl::Buffer, cl::Buffer, cl::Buffer, cl_int>
  removeFunc(clData.program, "removeEmptyStamps");

  cl_int zero{0};
  clData.queue.enqueueWriteBuffer(keepCounter, CL_TRUE, 0, sizeof(cl_int), &zero);

  cl::Event markEvent{markFunc(eargsMark, stampsData.subStampCounts, keepIndeces, keepCounter)};
  markEvent.wait();

  cl::Event padEvent{padFunc(eargsSort, keepIndeces, keepCounter)};
  padEvent.wait();
  
  cl::Event sortEvent;
  for (cl_int k=2;k<=paddedNStamps;k=2*k) // Outer loop, double size for each step
  {
    for (cl_int j=k>>1;j>0;j=j>>1)  // Inner loop, half size for each step
    {
      sortEvent = sortFunc(eargsSort, keepIndeces, j, k);
      sortEvent.wait();
    }
  }
  
  cl_int removedStampCount{};
  clData.queue.enqueueReadBuffer(keepCounter, CL_TRUE, 0, sizeof(cl_int), &removedStampCount);

  stampsData.stampCount = removedStampCount;
  stampsData.currentSubStamps = {clData.context, CL_MEM_READ_WRITE, sizeof(cl_int) * removedStampCount};

  cl::Event removeEvent = removeFunc(eargsRemove, 
      stampsData.stampCoords, stampsData.stampSizes,
      stampsData.stats.skyEsts, stampsData.stats.fwhms,
      stampsData.subStampCounts, stampsData.subStampCoords, stampsData.subStampValues,
      filteredStampCoords, filteredStampSizes, 
      filteredSkyEsts, filteredFwhms,
      filteredSubStampCounts, filteredSubStampCoords, filteredSubStampValues,
      keepIndeces, keepCounter, stampsData.currentSubStamps, maxSStamps);
  removeEvent.wait();
  
  stampsData.stampCoords    = filteredStampCoords;
  stampsData.stampSizes     = filteredStampSizes;
  stampsData.stats.skyEsts  = filteredSkyEsts;
  stampsData.stats.fwhms    = filteredFwhms;
  stampsData.subStampCoords = filteredSubStampCoords;
  stampsData.subStampValues = filteredSubStampValues;
  stampsData.subStampCounts = filteredSubStampCounts;
}

void resetSStampSkipMask(const int w, const int h, const ClData& clData) {
  cl::EnqueueArgs eargs{clData.queue, cl::NDRange(w * h)};
  cl::KernelFunctor<cl::Buffer> resetFunc(clData.program, "resetSkipMask");
  cl::Event unmaskEvent{resetFunc(eargs, clData.maskBuf)};
  unmaskEvent.wait();
}

void readFinalStamps(std::vector<Stamp>& stamps, const ClStampsData& stampsData, const ClData& clData, const Arguments& args) {
  cl::size_type maxSStamps(2 * args.maxKSStamps);

  std::vector<cl_int2> subStampCoords(maxSStamps * stampsData.stampCount);
  std::vector<cl_double> subStampValues(maxSStamps * stampsData.stampCount);
  std::vector<cl_int> subStampCounts(maxSStamps * stampsData.stampCount);
   
  static constexpr int nStampBuffers{3};
  std::vector<cl::Event> readEvents(nStampBuffers);
  clData.queue.enqueueReadBuffer(stampsData.subStampCoords, CL_FALSE, 0, sizeof(cl_int2) * maxSStamps * stampsData.stampCount, &subStampCoords[0], nullptr, &readEvents[0]);
  clData.queue.enqueueReadBuffer(stampsData.subStampValues, CL_FALSE, 0, sizeof(cl_double) * maxSStamps * stampsData.stampCount, &subStampValues[0], nullptr, &readEvents[1]);
  clData.queue.enqueueReadBuffer(stampsData.subStampCounts, CL_FALSE, 0, sizeof(cl_int) * maxSStamps * stampsData.stampCount, &subStampCounts[0], nullptr, &readEvents[2]);
  cl::Event::waitForEvents(readEvents);

  stamps.clear();
  stamps.reserve(stampsData.stampCount);

  for (size_t i{0}; i < stampsData.stampCount; i++)
  {
    auto &stamp{stamps.emplace_back(std::vector<SubStamp>{})};
    
    auto &sstamps{stamp.subStamps};
    
    for (size_t j{0}; j < subStampCounts[i]; j++)
    {
      size_t offset{i * maxSStamps + j};
      std::pair<cl_int, cl_int> imageCoords{subStampCoords[offset].s[0], subStampCoords[offset].s[1]};
      sstamps.emplace_back(SubStamp{imageCoords, subStampValues[offset]});
    }
  }

  assert(stampsData.stampCount == stamps.size());
 }