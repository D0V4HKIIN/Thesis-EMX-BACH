#include "fitsUtil.h"

#include <CL/opencl.hpp>
#include <memory>
#include <string>
#include <vector>

void readImage(Image& input, const Arguments& args) {
  CCfits::FITS* pIn{};
  try {
    pIn = new CCfits::FITS(input.getFile(), CCfits::RWmode::Read, true);
  } catch(const CCfits::FITS::CantOpen& err) {
    std::cout << "Unable to open file '" << input.getFile() << "'" << std::endl
              << err.message() << std::endl;
    throw;
  }
  CCfits::PHDU& img = pIn->pHDU();

  // Ifloat = -32, Idouble = -64
  long type = img.bitpix();
  if(type != CCfits::Ifloat && type != CCfits::Idouble) {
    throw std::invalid_argument("fits image of type" + std::to_string(type) +
                                " is not supported.");
  }

  input = Image(input.name, img.axis(0) * img.axis(1),
                std::make_pair(img.axis(0), img.axis(1)));

  img.readAllKeys();
  img.read(input.data);

  if(args.verbose) {
    std::cout << img << std::endl;
    std::cout << pIn->extension().size() << std::endl;
  }

  delete pIn;
}

void writeImage(const Image& img, const Arguments& args) {
  constexpr int nAxis = 2;
  CCfits::FITS* pFits{};

  try {
    long axisArr[nAxis]{img.axis.first, img.axis.second};

    pFits = new CCfits::FITS(img.getOutFile(), FLOAT_IMG, nAxis, axisArr);
  } catch(const CCfits::FITS::CantCreate& err) {
    std::cout << "Unable to save file '" << img.getFile() << "'" << std::endl
              << err.message() << std::endl;
    throw;
  }

  long fpixel(1);

  pFits->pHDU().write(fpixel, img.data.size(), img.data);

  if(args.verbose) {
    std::cout << pFits->pHDU() << std::endl;
    std::cout << pFits->extension().size() << std::endl;
  }

  delete pFits;
}
