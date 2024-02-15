#pragma once

#include <CL/opencl.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <utility>

#include "argsUtil.h"
#include "datatypeUtil.h"

/* Utils */
void checkError(const cl_int err);
void maskInput(const Image& tImg, const Image& sImg, ImageMask& mask);
void spreadMask(ImageMask& mask, int width);
void sigmaClip(std::vector<double>& data, double& mean, double& stdDev,
               int iter);
void calcStats(Stamp& stamp, Image& image, ImageMask& mask);

int ludcmp(std::vector<std::vector<double>>& matrix, const int matrixSize,
           std::vector<int>& index, double& rowInter);
void lubksb(std::vector<std::vector<double>>& matrix, const int matrixSize,
            std::vector<int>& index, std::vector<double>& result);
double makeKernel(Kernel& kern, const std::pair<const cl_long, const cl_long> imgSize, const int x,
                  const int y);

/* SSS */
void createStamps(const Image&, std::vector<Stamp>& stamps, const int w, const int h);
double checkSStamp(const SubStamp& sstamp, const Image& image, ImageMask& mask, const Stamp&, const ImageMask::masks badMask, const bool isTemplate);
cl_int findSStamps(Stamp& stamp, const Image& image, ImageMask& mask, const int index, const bool isTemplate);
void identifySStamps(std::vector<Stamp>& templStamps, Image& templImage, std::vector<Stamp>& scienceStamps, Image& scienceImage, ImageMask& mask, double* filledTempl, double* filledScience);

/* CMV */
void createB(Stamp&, Image&);
void convStamp(Stamp&, Image&, Kernel&, int n, int odd);
void cutSStamp(SubStamp&, Image&, ImageMask&);
int fillStamp(Stamp&, Image& tImg, Image& sImg, ImageMask&, Kernel&);

/* CD && KSC */
double testFit(std::vector<Stamp>& stamps, Image& tImg, Image& sImg, ImageMask& mask);
std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
createMatrix(std::vector<Stamp>& stamps, std::pair<cl_long, cl_long>& imgSize);
std::vector<double> createScProd(std::vector<Stamp>& stamps, Image&,
                                 std::vector<std::vector<double>>& weight);
double calcSig(Stamp&, std::vector<double>& kernSol, const Image& tImg, const Image& sImg, ImageMask& mask);
double getBackground(int x, int y, const std::vector<double>& kernSol,
                     std::pair<cl_long, cl_long> imgSize);
std::vector<float> makeModel(const Stamp&, std::vector<double>& kernSol,
                             const std::pair<const cl_long, const cl_long> imgSize);
void fitKernel(Kernel&, std::vector<Stamp>& stamps, Image& tImg, Image& sImg, ImageMask& mask);
bool checkFitSolution(Kernel&, std::vector<Stamp>& stamps, Image& tImg,
                      Image& sImg, ImageMask& mask);
