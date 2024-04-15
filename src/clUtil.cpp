#include "clUtil.h"

#include <fstream>

cl::Platform getDefaultPlatform() {
  // get all platforms (drivers)
  std::vector<cl::Platform> allPlatforms;
  cl::Platform::get(&allPlatforms);
  if(allPlatforms.size() == 0) {
    std::cout << " No platforms found. Check OpenCL installation!\n";
    std::exit(1);
  }
  cl::Platform defaultPlatform = allPlatforms[0];
  std::cout << "Using platform: "
            << defaultPlatform.getInfo<CL_PLATFORM_NAME>() << "\n";

  return defaultPlatform;
}

cl::Device getDefaultDevice(const cl::Platform &platform) {
  // get default device of the default platform
  std::vector<cl::Device> allDevices;
  platform.getDevices(CL_DEVICE_TYPE_ALL, &allDevices);
  if(allDevices.size() == 0) {
    std::cout << " No devices found. Check OpenCL installation!\n";
    std::exit(1);
  }
  cl::Device defaultDevice = allDevices[0];
  std::cout << "Using device: " << defaultDevice.getInfo<CL_DEVICE_NAME>()
            << "\n";

  return defaultDevice;
}

void printVerboseClInfo(const cl::Platform &platform, const cl::Device &device) {
  cl::size_type maxWorkGroupSize = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
  std::vector<cl::size_type> maxWorkItemSizes = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
  cl_ulong globalMemorySize = device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
  cl_ulong localMemorySize = device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();

  std::cout << "Max work group size: " << maxWorkGroupSize << std::endl;
  std::cout << "Max work item sizes: (";

  for (int i = 0; i < maxWorkItemSizes.size(); i++) {
    std::cout << maxWorkItemSizes[i];

    if (i < maxWorkItemSizes.size() - 1) {
      std::cout << "; ";
    }
  }

  std::cout << ")" << std::endl;

  std::cout << "Global memory size: " << globalMemorySize << " B" << std::endl;
  std::cout << "Local memory size: " << localMemorySize << " B" << std::endl;
}

std::string getKernelFunc(const std::string &fileName, const std::filesystem::path& rootPath) {
  std::ifstream t((rootPath / fileName).c_str());
  std::string tmp{std::istreambuf_iterator<char>{t},
                  std::istreambuf_iterator<char>{}};

  return tmp;
}