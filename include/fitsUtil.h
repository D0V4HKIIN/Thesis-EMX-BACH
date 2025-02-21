#pragma once

#include <CCfits/CCfits>

#include "argsUtil.h"
#include "datatypeUtil.h"

void readImage(Image& input, const Arguments& args);

void writeImage(const Image& img, const Arguments& args);
