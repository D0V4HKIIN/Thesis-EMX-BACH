#pragma once

#include <CL/opencl.hpp>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>

inline cl::Device getDefaultDevice() {
  // get all platforms (drivers)
  std::vector<cl::Platform> allPlatforms;
  cl::Platform::get(&allPlatforms);
  if(allPlatforms.size() == 0) {
    std::cout << " No platforms found. Check OpenCL installation!\n";
    exit(1);
  }
  cl::Platform defaultPlatform = allPlatforms[0];
  std::cout << "Using platform: "
            << defaultPlatform.getInfo<CL_PLATFORM_NAME>() << "\n";

  // get default device of the default platform
  std::vector<cl::Device> allDevices;
  defaultPlatform.getDevices(CL_DEVICE_TYPE_ALL, &allDevices);
  if(allDevices.size() == 0) {
    std::cout << " No devices found. Check OpenCL installation!\n";
    exit(1);
  }
  cl::Device defaultDevice = allDevices[0];
  std::cout << "Using device: " << defaultDevice.getInfo<CL_DEVICE_NAME>()
            << "\n";

  return defaultDevice;
}

inline std::string getKernelFunc(std::string &&fileName, const std::filesystem::path& rootPath,
                                   std::string &&path = "cl_kern/") {
  std::ifstream t((rootPath / (path + fileName)).c_str());
  std::string tmp{std::istreambuf_iterator<char>{t},
                  std::istreambuf_iterator<char>{}};

  return tmp;
}

inline auto getTime() -> decltype(std::chrono::high_resolution_clock::now()) {
  auto tmp{std::chrono::high_resolution_clock::now()};
  return tmp;
}

using timePoint = std::chrono::high_resolution_clock::time_point;

inline void printTime(std::ostream &os, const timePoint start, const timePoint stop) {
  os << "Time in ms: "
     << std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
            .count()
     << std::endl;
}

template <typename... Args>
cl::Program loadBuildPrograms(const cl::Context context, const cl::Device defaultDevice,
                                const std::filesystem::path& rootPath, Args... names) {
  cl::Program::Sources sources;
  for(auto n : {names...}) {
    std::string code = getKernelFunc(n, rootPath);

    sources.push_back({code.c_str(), code.length()});
  }

  cl::Program program(context, sources);
  if(program.build(defaultDevice, "-cl-fp32-correctly-rounded-divide-sqrt") != CL_SUCCESS) {
    std::cout << " Error building: "
              << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(defaultDevice)
              << "\n";
    exit(1);
  }

  return program;
}
