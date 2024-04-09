#include "bachUtil.h"
#include "mathUtil.h"
#include <cassert>
#include <ranges>

extern double ran1(int *idum);

void identifySStamps(std::vector<Stamp>& templStamps, const Image& templImage, std::vector<Stamp>& scienceStamps, const Image& scienceImage, ImageMask& mask, double* filledTempl, double* filledScience, const Arguments& args, ClData& clData) {
  std::cout << "Identifying sub-stamps in " << templImage.name << " and " << scienceImage.name << "..." << std::endl;

  assert(templStamps.size() == scienceStamps.size());

  std::cout << "calcStats (template)" << std::endl;
  calcStats(templStamps, templImage, mask, args, clData.tImgBuf, clData.tmpl, clData);
  std::cout << "calcStats (science)" << std::endl;
  calcStats(scienceStamps, scienceImage, mask, args, clData.sImgBuf, clData.sci, clData);

  std::cout << "findSStamps (template)" << std::endl;
  findSStamps(templStamps, templImage, mask, true, args, clData.tImgBuf, clData.tmpl, clData);
  std::cout << "findSStamps (science)" << std::endl;
  findSStamps(scienceStamps, scienceImage, mask, false, args, clData.sImgBuf, clData.sci, clData);
  
  int oldCount = templStamps.size();
  int newTemplCount{removeEmptyStamps(templStamps, args, clData.tmpl, clData)};
  int newScienceCount{removeEmptyStamps(scienceStamps, args, clData.sci, clData)};

  if (filledTempl != nullptr) {
    *filledTempl = static_cast<double>(newTemplCount) / oldCount;
  }

  if (filledScience != nullptr) {
    *filledScience = static_cast<double>(newScienceCount) / oldCount;
  }

  if(args.verbose) {
    std::cout << "Non-Empty template stamps: " << newTemplCount << std::endl;
    std::cout << "Non-Empty science stamps: " << newScienceCount << std::endl;
  }
}

void createStamps(const Image& templateImg, const Image& scienceImg, std::vector<Stamp>& templateStamps, std::vector<Stamp>& scienceStamps, const int w, const int h, const Arguments& args, const ClData& clData) {
  cl::EnqueueArgs eargsBounds{clData.queue, cl::NullRange, cl::NDRange(args.stampsx * args.stampsy), cl::NullRange};

  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl_int, cl_int, cl_long, cl_long, cl_long>
  boundsFunc(clData.program, "createStampBounds");

  std::vector<cl::Event> boundsEvents(2); 
  boundsEvents[0] = boundsFunc(eargsBounds,
                  clData.tmpl.stampCoords, clData.tmpl.stampSizes,
                  args.stampsx, args.stampsy, args.fStampWidth,
                  w, h);
  boundsEvents[1] = boundsFunc(eargsBounds,
                  clData.sci.stampCoords, clData.sci.stampSizes,
                  args.stampsx, args.stampsy, args.fStampWidth,
                  w, h);
  cl::Event::waitForEvents(boundsEvents);

  std::vector<cl_long> stampCoords(2 * args.stampsx * args.stampsy, 0);
  std::vector<cl_long> stampSizes(2 * args.stampsx * args.stampsy, 0);

  cl_int err = clData.queue.enqueueReadBuffer(clData.tmpl.stampCoords, CL_TRUE, 0,
    sizeof(cl_long) * 2 * args.stampsx * args.stampsy, &stampCoords[0]);
  checkError(err);
  err = clData.queue.enqueueReadBuffer(clData.tmpl.stampSizes, CL_TRUE, 0,
    sizeof(cl_long) * 2 * args.stampsx * args.stampsy, &stampSizes[0]);
  checkError(err);
  
  for(int j = 0; j < args.stampsy; j++) {
    for(int i = 0; i < args.stampsx; i++) {
      int startx = stampCoords[2*(j*args.stampsx + i) + 0];
      int starty = stampCoords[2*(j*args.stampsx + i) + 1];
      int stampw =  stampSizes[2*(j*args.stampsx + i) + 0];
      int stamph =  stampSizes[2*(j*args.stampsx + i) + 1];

      templateStamps.emplace_back();
      scienceStamps.emplace_back();
      
      templateStamps.back().data.reserve(args.fStampWidth*args.fStampWidth);
      templateStamps.back().coords = std::make_pair(startx, starty);
      templateStamps.back().size = std::make_pair(stampw, stamph);
      
      scienceStamps.back().data.reserve(args.fStampWidth*args.fStampWidth);
      scienceStamps.back().coords = std::make_pair(startx, starty);
      scienceStamps.back().size = std::make_pair(stampw, stamph);
      
      for(int y = 0; y < stamph; y++) {
        for(int x = 0; x < stampw; x++) {
          templateStamps.back().data.push_back(
            templateImg[(startx + x) + ((starty + y) * w)]);
          scienceStamps.back().data.push_back(
            scienceImg[(startx + x) + ((starty + y) * w)]);
        }
      }
    }
  }
}

cl_int findSStamps(std::vector<Stamp>& stamps, const Image& image, ImageMask& mask, const bool isTemplate, const Arguments& args, const cl::Buffer& imgBuf, const ClStampsData& stampsData, const ClData& clData) {
  auto [imgW, imgH] = image.axis;
  clData.queue.enqueueWriteBuffer(clData.maskBuf, CL_TRUE, 0, sizeof(cl_ushort) * imgW * imgH, &mask);

  ImageMask::masks badMask = ImageMask::ALL & ~ImageMask::OK_CONV;
  ImageMask::masks badPixelMask, skipMask;

  if (isTemplate) {
    badMask &= ~(ImageMask::BAD_PIXEL_S | ImageMask::SKIP_S);
    badPixelMask = ImageMask::BAD_PIXEL_T;
    skipMask     = ImageMask::SKIP_T;
  }
  else {
    badMask &= ~(ImageMask::BAD_PIXEL_T | ImageMask::SKIP_T);
    badPixelMask = ImageMask::BAD_PIXEL_S;
    skipMask     = ImageMask::SKIP_S;
  }
  
  cl_int maxSStamps{2 * args.maxKSStamps};

  constexpr int localSize{1};

  cl::EnqueueArgs eargsFindSStamps(clData.queue, cl::NDRange(roundUpToMultiple(stamps.size(), localSize)), cl::NDRange(localSize));
  cl::KernelFunctor<cl::Buffer, cl::Buffer,
                    cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                    cl::Buffer, cl::Buffer, cl::Buffer,
                    cl_double, cl_double,
                    cl_long, cl_int, cl_int, 
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
                  args.fStampWidth, args.hSStampWidth, maxSStamps, static_cast<cl_int>(stamps.size()),
                  static_cast<cl_ushort>(badMask),
                  static_cast<cl_ushort>(badPixelMask),
                  static_cast<cl_ushort>(skipMask),
                  cl::Local(sizeof(cl_int2) * maxSStamps * localSize),
                  cl::Local(sizeof(cl_double) * maxSStamps * localSize))};

  findSStampsEvent.wait();
  
  std::vector<cl_int2>  sstampCoords(maxSStamps * stamps.size(), {0,0});
  std::vector<cl_double> sstampValues(maxSStamps * stamps.size());
  std::vector<cl_int>    sstampCounts(stamps.size());

  
  clData.queue.enqueueReadBuffer(stampsData.subStampCoords, CL_TRUE, 0, sizeof(cl_int2)   * sstampCoords.size(), &sstampCoords[0]);
  clData.queue.enqueueReadBuffer(stampsData.subStampValues, CL_TRUE, 0, sizeof(cl_double) * sstampValues.size(), &sstampValues[0]);
  clData.queue.enqueueReadBuffer(stampsData.subStampCounts, CL_TRUE, 0, sizeof(cl_int)    * sstampCounts.size(), &sstampCounts[0]);
  clData.queue.enqueueReadBuffer(clData.maskBuf, CL_TRUE, 0, sizeof(cl_ushort) * imgW * imgH, &mask);

  int index{0};
  for (auto&& stamp : stamps) {
    for (size_t i = 0; i < sstampCounts[index]; i++)
    {
      size_t offset{index * maxSStamps + i};
      std::pair<cl_int, cl_int> imageCoords{sstampCoords[offset].x, sstampCoords[offset].y};
      constexpr std::pair<cl_int, cl_int> stampCoords(
        std::numeric_limits<cl_int>::max(),
        std::numeric_limits<cl_int>::max()
      );
      stamp.subStamps.emplace_back(std::vector<double>{}, 0.0,
                                   imageCoords, stampCoords,
                                   sstampValues[offset]);
    }

    if(args.verbose) {
      if(sstampCounts[index] == 0) {
        std::cout << "No suitable substamps found in stamp " << index << std::endl;
      }   
      else {
        std::cout << "Added " << sstampCounts[index]
                  << " substamps to stamp " << index << std::endl;
      }
    }
    index++;
  }
  return 0;
}

int removeEmptyStamps(std::vector<Stamp>& stamps, const Arguments& args, ClStampsData& stampsData, const ClData& clData) {
  
  int maxSStamps{2 * args.maxKSStamps};
  cl::Buffer filteredStampCoords{clData.context, CL_MEM_READ_WRITE, sizeof(cl_long2) * stamps.size()};
  cl::Buffer filteredStampSizes{clData.context, CL_MEM_READ_WRITE, sizeof(cl_long2) * stamps.size()};
  cl::Buffer filteredSkyEsts{clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * stamps.size()};
  cl::Buffer filteredFwhms{clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * stamps.size()};
  cl::Buffer filteredSubStampCoords{clData.context, CL_MEM_READ_WRITE, sizeof(cl_int2) * maxSStamps * stamps.size()};
  cl::Buffer filteredSubStampValues{clData.context, CL_MEM_READ_WRITE, sizeof(cl_double) * maxSStamps * stamps.size()};
  cl::Buffer filteredSubStampCounts{clData.context, CL_MEM_READ_WRITE, sizeof(cl_int) * maxSStamps * stamps.size()};

  cl::Buffer keepCounter{clData.context, CL_MEM_READ_WRITE, sizeof(cl_int)};
  cl::Buffer keepIndeces{clData.context, CL_MEM_READ_WRITE, sizeof(cl_int) * stamps.size()};
  
  cl::EnqueueArgs eargsMark{clData.queue,cl::NDRange{stamps.size()}};
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer>
  markFunc(clData.program, "markStampsToKeep");

  cl::EnqueueArgs eargsRemove{clData.queue,cl::NDRange{stamps.size()}};  
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, 
                    cl::Buffer, cl::Buffer, cl::Buffer,
                    cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                    cl::Buffer, cl::Buffer, cl::Buffer,
                    cl::Buffer, cl::Buffer, cl_int>
  removeFunc(clData.program, "removeEmptyStamps");

  cl_int zero{0};
  clData.queue.enqueueWriteBuffer(keepCounter, CL_TRUE, 0, sizeof(cl_int), &zero);

  cl::Event markEvent = markFunc(eargsMark, stampsData.subStampCounts, keepIndeces, keepCounter);
  markEvent.wait();
  
  cl::Event removeEvent = removeFunc(eargsRemove, 
      stampsData.stampCoords, stampsData.stampSizes,
      stampsData.stats.skyEsts, stampsData.stats.fwhms,
      stampsData.subStampCounts, stampsData.subStampCoords, stampsData.subStampValues,
      filteredStampCoords, filteredStampSizes, 
      filteredSkyEsts, filteredFwhms,
      filteredSubStampCounts, filteredSubStampCoords, filteredSubStampValues,
      keepIndeces, keepCounter, args.maxKSStamps);
  removeEvent.wait();
  
  stampsData.stampCoords    = filteredStampCoords;
  stampsData.stampSizes     = filteredStampSizes;
  stampsData.stats.skyEsts  = filteredSkyEsts;
  stampsData.stats.fwhms    = filteredFwhms;
  stampsData.subStampCoords = filteredSubStampCoords;
  stampsData.subStampValues = filteredSubStampValues;
  stampsData.subStampCounts = filteredSubStampCounts;
  
  cl_int newSize{0};
  clData.queue.enqueueReadBuffer(keepCounter, CL_TRUE, 0, sizeof(cl_int), &newSize);
  
  stamps.erase(std::remove_if(stamps.begin(), stamps.end(),
                              [](Stamp& s) { return s.subStamps.empty(); }),
               stamps.end());

  assert(newSize == stamps.size());
  
  return newSize;
}