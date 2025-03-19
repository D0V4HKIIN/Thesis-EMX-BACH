#pragma once

#include <CL/opencl.hpp>
#include <filesystem>
#include <iostream>

#include "argsUtil.h"

cl::Platform getDefaultPlatform(const Arguments &args);

cl::Device getDefaultDevice(const cl::Platform &platform,
                            const Arguments &args);
std::vector<cl::Device> getAcceleratorDevices(const Arguments &args);

void printVerboseClInfo(const cl::Device &device);

std::string getKernelFunc(const std::string &fileName,
                          const std::filesystem::path &rootPath);

template <typename... Args>
cl::Program loadBuildPrograms(const cl::Context &context,
                              const cl::Device &device,
                              const std::filesystem::path &rootPath,
                              Args... names) {
  cl::Program::Sources sources;
  for(auto n : {names...}) {
    std::string code = getKernelFunc(n, rootPath / "cl_kern");

    sources.push_back({code.c_str(), code.length()});
  }

  cl::Program program(context, sources);
  if(program.build(device, "-cl-fp32-correctly-rounded-divide-sqrt") !=
     CL_SUCCESS) {
    std::cout << " Error building: "
              << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";
    std::exit(1);
  }

  return program;
}
