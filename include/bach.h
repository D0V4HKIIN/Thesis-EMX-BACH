#pragma once

#include <filesystem>

#include "datatypeUtil.h"

void init(Image &templateImg, Image &scienceImg, ImageMask &mask);
void sss(Image &templateImg, Image &scienceImg, ImageMask &mask, std::vector<Stamp> &templateStamps, std::vector<Stamp> &sciStamps);
void cmv(Image &templateImg, Image &scienceImg, ImageMask &mask, std::vector<Stamp> &templateStamps, std::vector<Stamp> &sciStamps, Kernel &convolutionKernel);