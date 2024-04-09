#pragma once

#include <filesystem>

#include "datatypeUtil.h"

struct ClStampsData {
    cl::Buffer subStampCoords; // (x, y) coordinates
    cl::Buffer currentSubStamps;
    cl::Buffer subStampCounts;
    cl::Buffer w;
    cl::Buffer q;
    cl::Buffer b;
    int stampCount;
};

struct ClData {
    cl::Device &device;
    cl::Context &context;
    cl::Program &program;
    cl::CommandQueue &queue;

    cl::Buffer tImgBuf;
    cl::Buffer sImgBuf;
    cl::Buffer maskBuf;
    cl::Buffer convImg;

    struct {
        cl::Buffer xy;
        cl::Buffer filterX;
        cl::Buffer filterY;
        cl::Buffer vec;
        cl::Buffer solution;
    } kernel;

    struct {
        cl::Buffer yConvTmp;
    } cmv;

    struct {
        cl::Buffer xy;
    } bg;

    struct {
        cl::Buffer kernelXy;
    } cd;

    int gaussCount;
    int qCount;
    int bCount;
    int wRows;
    int wColumns;

    ClStampsData tmpl;
    ClStampsData sci;
};

void init(Image &templateImg, Image &scienceImg, ImageMask &mask, ClData& clData, const Arguments& args);
void sss(const Image &templateImg, const Image &scienceImg, ImageMask &mask, std::vector<Stamp> &templateStamps, std::vector<Stamp> &sciStamps, Arguments& args);
void cmv(const Image &templateImg, const Image &scienceImg, ImageMask &mask, std::vector<Stamp> &templateStamps, std::vector<Stamp> &sciStamps, const Kernel &convolutionKernel, ClData &clData, const Arguments& args);
bool cd(Image &templateImg, Image &scienceImg, ImageMask &mask, std::vector<Stamp> &templateStamps, std::vector<Stamp> &sciStamps, ClData &clData, const Arguments& args);
void ksc(const Image &templateImg, const Image &scienceImg, ImageMask &mask, std::vector<Stamp> &templateStamps, Kernel &convolutionKernel,
         const cl::Buffer &tImgBuf, const cl::Buffer &sImgBuf, ClData &clData, const ClStampsData &stampData, const Arguments& args);
double conv(const std::pair<cl_long, cl_long> &imgSize, Image &convImg, Kernel &convolutionKernel, bool convTemplate,
            ClData &clData, const Arguments& args);
void sub(const std::pair<cl_long, cl_long> &imgSize, Image &diffImg, bool convTemplate, double kernSum,
         const ClData &clData, const Arguments& args);
void fin(const Image &convImg, const Image &diffImg, const Arguments& args);
