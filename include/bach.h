#pragma once

#include <filesystem>

#include "datatypeUtil.h"

enum StatsIndeces {SKY_EST, FWHM, NORM, DIFF, CHI2};

// struct ClData
// {
//     cl::Device &device;
//     cl::Context &context;
//     cl::Program &program;
//     cl::CommandQueue &queue;

//     cl::Buffer tImgBuf;
//     cl::Buffer sImgBuf;
//     cl::Buffer maskBuf;

//     cl::Buffer swizzledTImgBuf;
//     cl::Buffer swizzledSImgBuf;
//     cl::Buffer swizzledMaskBuf;
    
//     cl::Buffer tmplStampsBuf;
//     cl::Buffer tmplStampsStatsBuf;
//     cl::Buffer sciStampsBuf;
//     cl::Buffer sciStampsStatsBuf;
//     cl::Buffer tmplStampsCoordsBuf;
//     cl::Buffer tmplStampsSizesBuf;
//     cl::Buffer sciStampsCoordsBuf;
//     cl::Buffer sciStampsSizesBuf;

//     cl::Buffer tmplSStampsCoordsBuf;
//     cl::Buffer tmplSStampsValuesBuf;
//     cl::Buffer tmplSStampsCountsBuf;
//     cl::Buffer sciSStampsCoordsBuf;
//     cl::Buffer sciSStampsValuesBuf;
//     cl::Buffer sciSStampsCountsBuf;
// };

struct ClStampsData {
    cl::Buffer stampCoords; // (x, y) coordinates
    cl::Buffer stampSizes;
    cl::Buffer stampStats;
    cl::Buffer subStampCoords; // (x, y) coordinates
    cl::Buffer subStampValues;
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

    struct {
        cl::Buffer gauss;
        cl::Buffer xy;
        cl::Buffer bg;
        cl::Buffer filterX;
        cl::Buffer filterY;
        cl::Buffer vec;
    } kernel;

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
void sss(const Image &templateImg, const Image &scienceImg, ImageMask &mask, std::vector<Stamp> &templateStamps, std::vector<Stamp> &sciStamps, Arguments& args, ClData& clData);
void cmv(const Image &templateImg, const Image &scienceImg, ImageMask &mask, std::vector<Stamp> &templateStamps, std::vector<Stamp> &sciStamps, const Kernel &convolutionKernel, const Arguments& args);
bool cd(Image &templateImg, Image &scienceImg, ImageMask &mask, std::vector<Stamp> &templateStamps, std::vector<Stamp> &sciStamps, const Arguments& args);
void ksc(const Image &templateImg, const Image &scienceImg, ImageMask &mask, std::vector<Stamp> &templateStamps, Kernel &convolutionKernel, const Arguments& args);
double conv(const Image &templateImg, const Image &scienceImg, ImageMask &mask, Image &convImg, Kernel &convolutionKernel, bool convTemplate,
          const cl::Context &context, const cl::Program &program, cl::CommandQueue &queue, const Arguments& args);
void sub(const Image &convImg, const Image &scienceImg, const ImageMask &mask, Image &diffImg, bool convTemplate, double kernSum,
         const cl::Context &context, const cl::Program &program, cl::CommandQueue &queue, const Arguments& args);
void fin(const Image &convImg, const Image &diffImg, const Arguments& args);
