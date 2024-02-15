#include <CL/opencl.hpp>

#include <iterator>
#include <iostream>
#include <vector>

#include "fitsUtil.h"
#include "clUtil.h"
#include "argsUtil.h"
#include "bachUtil.h"

#include "bach.h"

void init(Image &templateImg, Image &scienceImg, ImageMask &mask) {

  cl_int err{};

  err = readImage(templateImg);
  checkError(err);
  err = readImage(scienceImg);
  checkError(err);

  if(templateImg.axis != scienceImg.axis) {
    std::cout << "Template image and science image must be the same size!"
              << std::endl;
    exit(1);
  }
  mask = ImageMask(templateImg.axis);
  maskInput(templateImg, scienceImg, mask);
}

void sss(Image &templateImg, Image &scienceImg, ImageMask &mask, std::vector<Stamp> &templateStamps, std::vector<Stamp> &sciStamps) {
  std::cout << "\nCreating stamps..." << std::endl;
    
  auto [w, h] = templateImg.axis;
  args.fStampWidth = std::min(int(templateImg.axis.first / args.stampsx),
                              int(templateImg.axis.second / args.stampsy));
  args.fStampWidth -= args.fKernelWidth;
  args.fStampWidth -= args.fStampWidth % 2 == 0 ? 1 : 0;

  if(args.fStampWidth < args.fSStampWidth) {
    args.fStampWidth = args.fSStampWidth + args.fKernelWidth;
    args.fStampWidth -= args.fStampWidth % 2 == 0 ? 1 : 0;

    args.stampsx = int(templateImg.axis.first / args.fStampWidth);
    args.stampsy = int(templateImg.axis.second / args.fStampWidth);

  if(args.verbose)
      std::cout << "Too many stamps requested, using " << args.stampsx << "x"
                << args.stampsy << " stamps instead." << std::endl;
  }

  
  createStamps(templateImg, templateStamps, w, h);
  if(args.verbose) {
    std::cout << "Stamps created for " << templateImg.name << std::endl;
  }

  createStamps(scienceImg, sciStamps, w, h);
  if(args.verbose) {
    std::cout << "Stamps created for " << scienceImg.name << std::endl;
  }

  /* == Check Template Stamps  ==*/
  double filledTempl{};
  double filledScience{};
  identifySStamps(templateStamps, templateImg, sciStamps, scienceImg, mask, &filledTempl, &filledScience);
  if(filledTempl < 0.1 || filledScience < 0.1) {
    if(args.verbose)
      std::cout << "Not enough substamps found in " << templateImg.name
                << " trying again with lower thresholds..." << std::endl;
    args.threshLow *= 0.5;
    
    templateStamps.clear();
    sciStamps.clear();

    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        int index = y * w + x;
        mask.unmask(index, ImageMask::SKIP_S | ImageMask::SKIP_T);
      }
    }

    createStamps(templateImg, templateStamps, w, h);

    createStamps(scienceImg, sciStamps, w, h);

    identifySStamps(templateStamps, templateImg, sciStamps, scienceImg, mask, &filledTempl, &filledScience);
    args.threshLow /= 0.5;
  }

  if(templateStamps.size() == 0 && sciStamps.size() == 0) {
    std::cout << "No substamps found" << std::endl;
    exit(1);
  }
}

void cmv(Image &templateImg, Image &scienceImg, ImageMask &mask, std::vector<Stamp> &templateStamps, std::vector<Stamp> &sciStamps, Kernel &convolutionKernel) {
  std::cout << "\nCalculating matrix variables..." << std::endl;

  for(auto& s : templateStamps) {
    fillStamp(s, templateImg, scienceImg, mask, convolutionKernel);
  }
  for(auto& s : sciStamps) {
    fillStamp(s, scienceImg, templateImg, mask, convolutionKernel);
  }
}

void cd(Image &templateImg, Image &scienceImg, ImageMask &mask, std::vector<Stamp> &templateStamps, std::vector<Stamp> &sciStamps) {
  std::cout << "\nChoosing convolution direction..." << std::endl;

  double templateMerit = testFit(templateStamps, templateImg, scienceImg, mask);
  double scienceMerit = testFit(sciStamps, scienceImg, templateImg, mask);
  if(args.verbose)
    std::cout << "template merit value = " << templateMerit
              << ", science merit value = " << scienceMerit << std::endl;
  if(scienceMerit <= templateMerit) {
    std::swap(scienceImg, templateImg);
    std::swap(sciStamps, templateStamps);
  }
  if(args.verbose)
    std::cout << templateImg.name << " chosen to be convolved." << std::endl;
}

void ksc(Image &templateImg, Image &scienceImg, ImageMask &mask, std::vector<Stamp> &templateStamps, Kernel &convolutionKernel) {

  std::cout << "\nFitting kernel..." << std::endl;

  fitKernel(convolutionKernel, templateStamps, templateImg, scienceImg, mask);

}