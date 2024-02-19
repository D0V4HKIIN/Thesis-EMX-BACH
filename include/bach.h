#pragma once

#include <filesystem>

#include "datatypeUtil.h"

struct ClData
{
    cl::Device &device;
    cl::Context &context;
    cl::Program &program;
    cl::CommandQueue &queue;

    cl::Buffer tImgBuf;
    cl::Buffer sImgBuf;
    cl::Buffer maskBuf;
};

void init(Image &templateImg, Image &scienceImg, ImageMask &mask, ClData& clData, const Arguments& args);
void sss(const Image &templateImg, const Image &scienceImg, ImageMask &mask, std::vector<Stamp> &templateStamps, std::vector<Stamp> &sciStamps, Arguments& args);
void cmv(const Image &templateImg, const Image &scienceImg, ImageMask &mask, std::vector<Stamp> &templateStamps, std::vector<Stamp> &sciStamps, const Kernel &convolutionKernel, const Arguments& args);
bool cd(Image &templateImg, Image &scienceImg, ImageMask &mask, std::vector<Stamp> &templateStamps, std::vector<Stamp> &sciStamps, const Arguments& args);
void ksc(const Image &templateImg, const Image &scienceImg, ImageMask &mask, std::vector<Stamp> &templateStamps, Kernel &convolutionKernel, const Arguments& args);
double conv(const Image &templateImg, const Image &scienceImg, ImageMask &mask, Image &convImg, Kernel &convolutionKernel,
          const cl::Context &context, const cl::Program &program, cl::CommandQueue &queue, const Arguments& args);
void sub(const Image &convImg, const Image &scienceImg, const ImageMask &mask, Image &diffImg, bool convTemplate, double kernSum,
         const cl::Context &context, const cl::Program &program, cl::CommandQueue &queue, const Arguments& args);
void fin(const Image &convImg, const Image &diffImg, const Arguments& args);
