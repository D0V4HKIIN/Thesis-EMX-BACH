#pragma once

#include <CCfits/CCfits>
#include <CL/opencl.hpp>
#include <memory>
#include <string>
#include <vector>

#include "argsUtil.h"
#include "datatypeUtil.h"

inline cl_int readImage(Image& input, const Arguments& args) {
  CCfits::FITS* pIn{};
  try {
    pIn = new CCfits::FITS(input.getFile(), CCfits::RWmode::Read, true);
  } catch(CCfits::FITS::CantOpen err) {
    std::cout << "Unable to open file '" << input.getFile() << "'" << std::endl << err.message() << std::endl;
    return -1;
  }
  CCfits::PHDU& img = pIn->pHDU();

  // Ifloat = -32, Idouble = -64
  cl_long type = img.bitpix();
  if(type != CCfits::Ifloat && type != CCfits::Idouble) {
    throw std::invalid_argument("fits image of type" + std::to_string(type) +
                                " is not supported.");
    return -1;
  }

  img.readAllKeys();

  std::valarray<double> temp(0.0, img.axis(0) * img.axis(1));
  img.read(temp);
  input =
      Image{input.name, std::vector<double>{std::begin(temp), std::end(temp)},
            std::make_pair(img.axis(0), img.axis(1))};

  if(args.verbose) {
    std::cout << img << std::endl;
    std::cout << pIn->extension().size() << std::endl;
  }

  delete pIn;
  return 0;
}

inline cl_int writeImage(const Image& img, const Arguments& args) {
  constexpr cl_long nAxis = 2;
  CCfits::FITS* pFits{};

  try {
    long axisArr[nAxis]{img.axis.first, img.axis.second};

    pFits = new CCfits::FITS(img.getOutFile(), FLOAT_IMG, nAxis, axisArr);
  } catch(CCfits::FITS::CantCreate) {
    delete pFits;
    return -1;
  }

  cl_long fpixel(1);

  valarray<double> data{&img, img.size()};

  pFits->pHDU().write(fpixel, data.size(), data);

  if(args.verbose) {
    std::cout << pFits->pHDU() << std::endl;
    std::cout << pFits->extension().size() << std::endl;
  }

  delete pFits;
  return 0;
}
