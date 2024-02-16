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

void init(Image &templateImg, Image &scienceImg, ImageMask &mask, ClData& clData);
void sss(Image &templateImg, Image &scienceImg, ImageMask &mask, std::vector<Stamp> &templateStamps, std::vector<Stamp> &sciStamps);
void cmv(Image &templateImg, Image &scienceImg, ImageMask &mask, std::vector<Stamp> &templateStamps, std::vector<Stamp> &sciStamps, Kernel &convolutionKernel);
void cd(Image &templateImg, Image &scienceImg, ImageMask &mask, std::vector<Stamp> &templateStamps, std::vector<Stamp> &sciStamps);
void ksc(Image &templateImg, Image &scienceImg, ImageMask &mask, std::vector<Stamp> &templateStamps, Kernel &convolutionKernel);
void conv(Image &templateImg, Image &scienceImg, ImageMask &mask, Image &convImg, Kernel &convolutionKernel, cl::Context &context, cl::Program &program, cl::CommandQueue &queue);
void sub(Image &convImg, Image &scienceImg, ImageMask &mask, Image &diffImg, cl::Context &context, cl::Program &program, cl::CommandQueue &queue);
void fin(Image &convImg, Image &diffImg);
