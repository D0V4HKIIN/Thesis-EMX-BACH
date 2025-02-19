#include "argsUtil.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <utility>

const char* getCmdOption(const char** begin, const char** end,
                         const std::string& option) {
  const char** itr = std::find(begin, end, option);
  if(itr != end && ++itr != end) {
    return *itr;
  }
  return 0;
}

bool cmdOptionExists(const char** begin, const char** end,
                     const std::string& option) {
  return std::find(begin, end, option) != end;
}

void getArguments(const int argc, const char* argv[], Arguments& args) {
  if(cmdOptionExists(argv, argv + argc, "-o")) {
    args.outName = getCmdOption(argv, argv + argc, "-o");
  }

  if(cmdOptionExists(argv, argv + argc, "-op")) {
    args.outPath = getCmdOption(argv, argv + argc, "-op");
  }

  if(cmdOptionExists(argv, argv + argc, "-ip")) {
    args.inputPath = getCmdOption(argv, argv + argc, "-ip");
  }

  if(cmdOptionExists(argv, argv + argc, "-sx")) {
    std::stringstream sstr{getCmdOption(argv, argv + argc, "-sx")};
    sstr >> args.stampsx;
  }

  if(cmdOptionExists(argv, argv + argc, "-sy")) {
    std::stringstream sstr{getCmdOption(argv, argv + argc, "-sy")};
    sstr >> args.stampsy;
  }

  if(cmdOptionExists(argv, argv + argc, "-v")) {
    args.verbose = true;
  }

  if(cmdOptionExists(argv, argv + argc, "-vt")) {
    args.verboseTime = true;
  }

  if(cmdOptionExists(argv, argv + argc, "-p")) {
    args.platform = atoi(getCmdOption(argv, argv + argc, "-p"));
  }

  if(cmdOptionExists(argv, argv + argc, "-d")) {
    args.device = atoi(getCmdOption(argv, argv + argc, "-d"));
  }

  if(cmdOptionExists(argv, argv + argc, "-t")) {
    args.templateName = getCmdOption(argv, argv + argc, "-t");
  } else {
    throw std::invalid_argument("Template file Input is required!");
    return;
  }

  if(cmdOptionExists(argv, argv + argc, "-s")) {
    args.scienceName = getCmdOption(argv, argv + argc, "-s");
  } else {
    throw std::invalid_argument("Science file input is required!");
    return;
  }
}