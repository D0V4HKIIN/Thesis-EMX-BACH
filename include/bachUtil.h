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
void maskInput(const Image& tImg, const Image& sImg, ImageMask& mask, const ClData& clData);
void sigmaClip(const std::vector<double>& data, double& mean, double& stdDev,
               const int iter);
void calcStats(Stamp& stamp, const Image& image, ImageMask& mask);

int ludcmp(std::vector<std::vector<double>>& matrix, const int matrixSize,
           std::vector<int>& index, double& rowInter);
void lubksb(std::vector<std::vector<double>>& matrix, const int matrixSize,
            const std::vector<int>& index, std::vector<double>& result);
double makeKernel(Kernel& kern, const std::pair<cl_long, cl_long> imgSize, const int x,
                  const int y);

/* SSS */
void createStamps(const Image& img, std::vector<Stamp>& stamps, const int w, const int h);
double checkSStamp(const SubStamp& sstamp, const Image& image, ImageMask& mask, const Stamp&, const ImageMask::masks badMask, const bool isTemplate);
cl_int findSStamps(Stamp& stamp, const Image& image, ImageMask& mask, const int index, const bool isTemplate);
void identifySStamps(std::vector<Stamp>& templStamps, const Image& templImage, std::vector<Stamp>& scienceStamps, const Image& scienceImage, ImageMask& mask, double* filledTempl, double* filledScience);

/* CMV */
void createB(Stamp& s, const Image& img);
void convStamp(Stamp&s , const Image& img, const Kernel& k, const int n, const int odd);
void cutSStamp(SubStamp& ss, const Image& img, const ImageMask& mask);
int fillStamp(Stamp& s, const Image& tImg, const Image& sImg, const ImageMask& mask, const Kernel& k);

/* CD && KSC */
double testFit(std::vector<Stamp>& stamps, const Image& tImg, const Image& sImg, ImageMask& mask);
std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
createMatrix(const std::vector<Stamp>& stamps, const std::pair<cl_long, cl_long>& imgSize);
std::vector<double> createScProd(const std::vector<Stamp>& stamps, const Image& img,
                                 const std::vector<std::vector<double>>& weight);
double calcSig(Stamp& s, const std::vector<double>& kernSol, const Image& tImg, const Image& sImg, ImageMask& mask);
double getBackground(const int x, const int y, const std::vector<double>& kernSol,
                     const std::pair<cl_long, cl_long> imgSize);
std::vector<float> makeModel(const Stamp& s, const std::vector<double>& kernSol,
                             const std::pair<cl_long, cl_long> imgSize);
void fitKernel(Kernel& k, std::vector<Stamp>& stamps, const Image& tImg, const Image& sImg, ImageMask& mask);
bool checkFitSolution(const Kernel& k, std::vector<Stamp>& stamps, const Image& tImg,
                      const Image& sImg, ImageMask& mask);
