#ifndef BACH_UTIL
#define BACH_UTIL

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
void checkError(cl_int err);
void maskInput(Image& tImg, Image& sImg);
void spreadMask(Image& image, int width);
void sigmaClip(std::vector<double>& data, double& mean, double& stdDev,
               int iter);
bool inImage(Image&, int x, int y);
void calcStats(Stamp&, Image&);

int ludcmp(std::vector<std::vector<double>>& matrix, int matrixSize,
           std::vector<int>& index, double& rowInter);
void lubksb(std::vector<std::vector<double>>& matrix, int matrixSize,
            std::vector<int>& index, std::vector<double>& result);
double makeKernel(Kernel&, std::pair<cl_long, cl_long>, int x, int y);

/* SSS */
void createStamps(Image&, std::vector<Stamp>& stamps, int w, int h);
double checkSStamp(SubStamp&, Image&, Stamp&);
cl_int findSStamps(Stamp&, Image&, int index);
void identifySStamps(std::vector<Stamp>& templStamps, Image& templImage, std::vector<Stamp>& scienceStamps, Image& scienceImage, double* filledTempl, double* filledScience);
void identifySStampsInStamp(std::vector<Stamp>& stamps, Image& image);

/* CMV */
void createB(Stamp&, Image&);
void convStamp(Stamp&, Image&, Kernel&, int n, int odd);
void cutSStamp(SubStamp&, Image&);
int fillStamp(Stamp&, Image& tImg, Image& sImg, Kernel&);

/* CD && KSC */
double testFit(std::vector<Stamp>& stamps, Image& tImg, Image& sImg);
std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
createMatrix(std::vector<Stamp>& stamps, std::pair<cl_long, cl_long>& imgSize);
std::vector<double> createScProd(std::vector<Stamp>& stamps, Image&,
                                 std::vector<std::vector<double>>& weight);
double calcSig(Stamp&, std::vector<double>& kernSol, Image& tImg, Image& sImg);
double getBackground(int x, int y, std::vector<double>& kernSol,
                     std::pair<cl_long, cl_long> imgSize);
std::vector<float> makeModel(Stamp&, std::vector<double>& kernSol,
                             std::pair<cl_long, cl_long> imgSize);
void fitKernel(Kernel&, std::vector<Stamp>& stamps, Image& tImg, Image& sImg);
bool checkFitSolution(Kernel&, std::vector<Stamp>& stamps, Image& tImg,
                      Image& sImg);
#endif
