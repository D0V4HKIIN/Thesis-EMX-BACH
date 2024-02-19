#pragma once

#include <CL/opencl.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <utility>

#include "argsUtil.h"
#include "bach.h"
#include "datatypeUtil.h"

/* Utils */
void checkError(const cl_int err);
void maskInput(const Image& tImg, const Image& sImg, ImageMask& mask, const ClData& clData, const Arguments& args);
void sigmaClip(const std::vector<double>& data, double& mean, double& stdDev,
               const int iter, const Arguments& args);
void calcStats(Stamp& stamp, const Image& image, ImageMask& mask, const Arguments& args);

int ludcmp(std::vector<std::vector<double>>& matrix, const int matrixSize,
           std::vector<int>& index, double& rowInter, const Arguments& args);
void lubksb(std::vector<std::vector<double>>& matrix, const int matrixSize,
            const std::vector<int>& index, std::vector<double>& result);
double makeKernel(Kernel& kern, const std::pair<cl_long, cl_long> imgSize, const int x,
                  const int y, const Arguments& args);

/* SSS */
void createStamps(const Image& img, std::vector<Stamp>& stamps, const int w, const int h, const Arguments& args);
double checkSStamp(const SubStamp& sstamp, const Image& image, ImageMask& mask, const Stamp&, const ImageMask::masks badMask, const bool isTemplate, const Arguments& args);
cl_int findSStamps(Stamp& stamp, const Image& image, ImageMask& mask, const int index, const bool isTemplate, const Arguments& args);
void identifySStamps(std::vector<Stamp>& templStamps, const Image& templImage, std::vector<Stamp>& scienceStamps, const Image& scienceImage, ImageMask& mask, double* filledTempl, double* filledScience, const Arguments& args);

/* CMV */
void createB(Stamp& s, const Image& img, const Arguments& args);
void convStamp(Stamp&s , const Image& img, const Kernel& k, const int n, const int odd, const Arguments& args);
void cutSStamp(SubStamp& ss, const Image& img, const ImageMask& mask, const Arguments& args);
int fillStamp(Stamp& s, const Image& tImg, const Image& sImg, const ImageMask& mask, const Kernel& k, const Arguments& args);

/* CD && KSC */
double testFit(std::vector<Stamp>& stamps, const Image& tImg, const Image& sImg, ImageMask& mask, const Arguments& args);
std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
createMatrix(const std::vector<Stamp>& stamps, const std::pair<cl_long, cl_long>& imgSize, const Arguments& args);
std::vector<double> createScProd(const std::vector<Stamp>& stamps, const Image& img,
                                 const std::vector<std::vector<double>>& weight, const Arguments& args);
double calcSig(Stamp& s, const std::vector<double>& kernSol, const Image& tImg, const Image& sImg, ImageMask& mask, const Arguments& args);
double getBackground(const int x, const int y, const std::vector<double>& kernSol,
                     const std::pair<cl_long, cl_long> imgSize, const Arguments& args);
std::vector<float> makeModel(const Stamp& s, const std::vector<double>& kernSol,
                             const std::pair<cl_long, cl_long> imgSize, const Arguments& args);
void fitKernel(Kernel& k, std::vector<Stamp>& stamps, const Image& tImg, const Image& sImg, ImageMask& mask, const Arguments& args);
bool checkFitSolution(const Kernel& k, std::vector<Stamp>& stamps, const Image& tImg,
                      const Image& sImg, ImageMask& mask, const Arguments& args);
