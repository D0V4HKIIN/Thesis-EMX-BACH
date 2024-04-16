#define MASK_BAD_PIX_VAL (1 << 0)
#define MASK_SAT_PIXEL (1 << 1)
#define MASK_LOW_PIXEL (1 << 2)
#define MASK_NAN_PIXEL (1 << 3)
#define MASK_BAD_CONV (1 << 4)
#define MASK_INPUT_MASK (1 << 5)
#define MASK_OK_CONV (1 << 6)
#define MASK_BAD_INPUT (1 << 7)
#define MASK_BAD_PIXEL_T (1 << 8)
#define MASK_SKIP_T (1 << 9)
#define MASK_BAD_PIXEL_S (1 << 10)
#define MASK_SKIP_S (1 << 11)
#define MASK_BAD_OUTPUT (1 << 12)

#define ZEROVAL ((double)1e-10)

//TODO: Fix swizzling for stamp data

void kernel createStampBounds(global int2 *stampsCoords, global int2 *stampsSizes,
                              const int nStampsX, const int nStampsY, const int fullStampWidth, 
                              const int w, const int h) {
    const int id = get_global_id(0);
    const int stampX = id % nStampsX;
    const int stampY = id / nStampsX;


    int startX = stampX * w / nStampsX;
    int startY = stampY * h / nStampsY;
    
    int stopX = min(startX + fullStampWidth, w);
    int stopY = min(startY + fullStampWidth, h);

    int stampW = stopX - startX;
    int stampH = stopY - startY;

    stampsCoords[id] = (int2)(startX, startY);
    stampsSizes[id] = (int2)(stampW, stampH);
}


#define M1 259200
#define IA1 7141
#define IC1 54773
#define RM1 (1.0/M1)
#define M2 134456
#define IA2 8121
#define IC2 28411
#define RM2 (1.0/M2)
#define M3 243000
#define IA3 4561
#define IC3 51349
double ran1(int *idum, long *ix1, long *ix2, long *ix3, double *r, int *iff) {
    double temp;
    int j;
    /* void nrerror(char *error_text); */
    
    if (*idum < 0 || *iff == 0) {
        *iff=1;
        *ix1=(IC1-(*idum)) % M1;
        *ix1=(IA1*(*ix1)+IC1) % M1;
        *ix2=*ix1 % M2;
        *ix1=(IA1*(*ix1)+IC1) % M1;
        *ix3=(*ix1) % M3;
        for (j=1;j<=97;j++) {
            *ix1=(IA1*(*ix1)+IC1) % M1;
            *ix2=(IA2*(*ix2)+IC2) % M2;
            r[j]=((*ix1)+(*ix2)*RM2)*RM1;
        }
        *idum=1;
    }
    *ix1=(IA1*(*ix1)+IC1) % M1;
    *ix2=(IA2*(*ix2)+IC2) % M2;
    *ix3=(IA3*(*ix3)+IC3) % M3;
    j=1 + ((97*(*ix3))/M3);
    /* if (j > 97 || j < 1) nrerror("RAN1: This cannot happen."); */
    temp=r[j];
    r[j]=(*ix1+(*ix2)*RM2)*RM1;
    return temp;
}

void kernel sampleStamp(global const double *img, global const ushort *mask, 
                        global const int2 *stampsCoords, global const int2 *stampsSizes,
                        global double *samples, global int *sampleCounts,
                        const long w, const int nSamples) {
    
    int stamp = get_global_id(0);
    int2 stampCoords = stampsCoords[stamp];
    int2 stampSize = stampsSizes[stamp];
    int stampNumPix = stampSize.x * stampSize.y;
    int sampleCounter = 0;

    int idum = -666;
    long ix1,ix2,ix3;
    double r[98];
    int iff=0;

    for(int iter = 0; sampleCounter < nSamples && iter < stampNumPix; iter++) {
        int randX = (int)floor(ran1(&idum, &ix1, &ix2, &ix3, r, &iff) * (int)stampSize.x);
        int randY = (int)floor(ran1(&idum, &ix1, &ix2, &ix3, r, &iff) * (int)stampSize.y);
        
        int xI = randX + stampCoords.x;
        int yI = randY + stampCoords.y;
        int indexI = xI + yI*w;

        if(mask[indexI] > 0 || fabs(img[indexI]) <= ZEROVAL) {
        continue;
        }

        samples[stamp * nSamples + sampleCounter++] = img[indexI];
    }

    sampleCounts[stamp] = sampleCounter;
    for (int i = sampleCounter; i < nSamples; i++) {
        samples[stamp * nSamples + i] = 0;
    }
}

void kernel pad(global const double *buffer, global double *paddedBuffer,
                const int n, const int paddedN){
    int idx = get_global_id(0);
    int stamp  = get_global_id(1);

    paddedBuffer[stamp * paddedN + idx] =
        idx < n ? buffer[stamp * n + idx] : INFINITY;
}

static void exchangeDouble(global double *i, global double *j)
{
	double k;
	k = *i;
	*i = *j;
	*j = k;
}

void kernel sortSamples(global double *paddedSamples, const int paddedNSamples, const int j, const int k){
    int id = get_global_id(0);                        
    
    const int i = id % paddedNSamples;
    const int stamp = id / paddedNSamples;
    const int ixj=i^j; // Calculate indexing!
    if ((ixj)>i)
    {
        if ((i&k)==0 && paddedSamples[id] > paddedSamples[stamp * paddedNSamples + ixj]) {
        exchangeDouble(&paddedSamples[id], &paddedSamples[stamp * paddedNSamples + ixj]);
        }
        if ((i&k)!=0 && paddedSamples[id] < paddedSamples[stamp * paddedNSamples + ixj]) {
        exchangeDouble(&paddedSamples[id], &paddedSamples[stamp * paddedNSamples + ixj]);
        }
    }
}

void kernel resetGoodPixelCounts(global int *goodPixelCounts) {
    int id = get_global_id(0);
    goodPixelCounts[id] = 0;
}

void kernel maskStamp(global const double *img, global ushort *mask, 
                      global const int2 *stampCoords, 
                      global double *goodPixels, global int *goodPixelCounts,
                      const int fullStampWidth, const int w, const int h){
    int indexS = get_global_id(0);
    int stamp  = get_global_id(1);

    int2 coords = stampCoords[stamp];

    int x = indexS % fullStampWidth;
    int y = indexS / fullStampWidth;

    int xI = x + coords.x;
    int yI = y + coords.y;
    int indexI = xI + yI*w;

    if (xI >= w || yI >= h) return;
    
    double pixel = img[indexI];
    if (mask[indexI] > 0 || pixel <= ZEROVAL) return;

    if (isnan(pixel)) {
        mask[indexI] |= (ushort)(MASK_NAN_PIXEL | MASK_BAD_INPUT);
        return;
    }

    int stampOffset = stamp * get_global_size(0);
    int pixelOffset = atomic_inc(&goodPixelCounts[stamp]);
    
    goodPixels[stampOffset + pixelOffset] = pixel;

}

void kernel createHistogram(global const double *img, global const ushort *mask,
                            global const int2 *stampCoords, global const int2 *stampSizes,
                            global const double *means, global const double *invStdDevs,
                            global const double *paddedSamples, global const int *sampleCounts,
                            global int *bins, global double *fwhms, global double *skyEsts,
                            const int width, const int stampCount,
                            const int nSamples, const int paddedNSamples,
                            const double iqRange, const double sigClipAlpha) {
    int stampId = get_global_id(0);

    if (stampId >= stampCount) {
        return;
    }

    int2 stampCoord = stampCoords[stampId];
    int2 stampSize = stampSizes[stampId];

    int sampleCount = sampleCounts[stampId];
    
    double upProc = 0.9;
    double midProc = 0.5;

    double mean = means[stampId];
    double invStdDev = invStdDevs[stampId];

    double upProcSample = paddedSamples[stampId * paddedNSamples + (int)(upProc * sampleCount)];
    double midProcSample = paddedSamples[stampId * paddedNSamples + (int)(midProc * sampleCount)];
    
    double binSize = (upProcSample - midProcSample) / (double)nSamples;
    double lowerBinVal = midProcSample - (128.0 * binSize);
    
    int firstBin = 256 * stampId;

    int attempts = 0;
    int okCount = 0;
    double lower = 0.0;
    double upper = 0.0;
    bool setFwhm = true;
    double skyEst = 0.0;

    while(true) {
        if(attempts >= 5) {
            setFwhm = false;
            break;
        }

        for (int i = 0; i < 256; i++) {
            bins[firstBin + i] = 0;
        }
        okCount = 0;

        for(int y = 0; y < stampSize.y; y++) {
            for(int x = 0; x < stampSize.x; x++) {
                int indexI = (stampCoord.x + x) + (stampCoord.y + y) * width;
                double imgV = img[indexI];

                if(mask[indexI] != 0 || imgV <= 1e-10) {
                    continue;
                }

                if((fabs(imgV - mean) * invStdDev) > sigClipAlpha) {
                    continue;
                }
                
                int index = clamp((int)floor((imgV - lowerBinVal) / binSize) + 1, 0, 255);

                bins[firstBin + index] = bins[firstBin + index] + 1;
                okCount++;
            }
        }

        if(okCount == 0 || binSize == 0.0) {
            setFwhm = false;
            break;
        }

        double sumBins = 0.0;
        double maxDens = 0.0;
        int lowerIndex = 1;
        int upperIndex = 1;
        int maxIndex = -1;
        while (upperIndex < 255) {
            while(sumBins < okCount / 10.0 && upperIndex < 255) {
                sumBins += bins[firstBin + upperIndex++];
            }
            if(sumBins / (upperIndex - lowerIndex) > maxDens) {
                maxDens = sumBins / (upperIndex - lowerIndex);
                maxIndex = lowerIndex;
            }
            sumBins -= bins[firstBin + lowerIndex++];
        }
        if(maxIndex < 0 || maxIndex > 255) maxIndex = 0;

        sumBins = 0.0;
        double sumExpect = 0.0;
        for(int i = maxIndex; sumBins < okCount / 10.0 && i < 255; i++) {
            sumBins += bins[firstBin + i];
            sumExpect += i * bins[firstBin + i];
        }

        double modeBin = sumExpect / sumBins + 0.5;
        skyEst = lowerBinVal + binSize * (modeBin - 1.0);

        lower = okCount * 0.25;
        upper = okCount * 0.75;
        sumBins = 0.0;

        int i = 0;
        while (sumBins < lower) {
            sumBins += bins[firstBin + i++];
        }
        lower = i - (sumBins - lower) / bins[firstBin + i - 1];
        while (sumBins < upper) {
            sumBins += bins[firstBin + i++];
        }
        upper = i - (sumBins - upper) / bins[firstBin + i - 1];

        if(lower < 1.0 || upper > 255.0) {
            lowerBinVal -= 128.0 * binSize;
            binSize *= 2;
        }
        else if(upper - lower < 40.0) {
            binSize /= 3.0;
            lowerBinVal = skyEst - 128.0 * binSize;
        }
        else {
            break;
        }
        
        attempts++;
    }

    fwhms[stampId] = setFwhm ? binSize * (upper - lower) / iqRange : 0.0;
    skyEsts[stampId] = skyEst;
}

double checkSStamp(global const double *img, global ushort *mask,
                   const double skyEst, const double fwhm, const long imgW,
                   const int2 sstampCoords, const int hSStampWidth,
                   const int2 stampCoords, const int2 stampSize,
                   const double threshHigh, const double threshKernFit,
                   const ushort badMask, const ushort badPixelMask) {
    double retVal = 0.0;
    int stamp = get_global_id(0);

    int startX = max(sstampCoords.x - hSStampWidth, stampCoords.x);
    int startY = max(sstampCoords.y - hSStampWidth, stampCoords.y);
    int endX   = min(sstampCoords.x + hSStampWidth, stampCoords.x + stampSize.x - 1);
    int endY   = min(sstampCoords.y + hSStampWidth, stampCoords.y + stampSize.y - 1);
    
    for(int y = startY; y <= endY; y++) {
        for(int x = startX; x <= endX; x++) {
            
            int absCoords = x + y * imgW;
            if ((mask[absCoords] & badMask) > 0) {
                return 0.0;
            }

            double imgValue = img[absCoords];
            if(imgValue > threshHigh) {
                mask[absCoords] |= badPixelMask;
                return 0.0;
            }
            
            double kernFit = (imgValue - skyEst) / fwhm;
            if((imgValue - skyEst) / fwhm > threshKernFit) {
                retVal += imgValue;
            }
        }
    }
    return retVal;
}

void sortSubStamps(const int substampCount, local int2 *coords, local double *values)
{
    int i = 1;
    while (i < substampCount) {
        int j = i;
        while ((j > 0) && (values[j-1] < values[j])){
            int2 tmpCoords = coords[j-1];
            coords[j-1] = coords[j];
            coords[j] = tmpCoords;
            double tmpVal = values[j-1];
            values[j-1] = values[j];
            values[j] = tmpVal;
            j--;
        }
        i++;
    }
}

void kernel findSubStamps(global const double* img, global ushort *mask, 
                          global const int2 *stampsCoords, global const int2 *stampsSizes,
                          global const double *skyEsts, global const double *fwhms,
                          global int2 *sstampsCoords, global double *sstampsValues,
                          global int *sstampsCounts,
                          const double threshHigh, const double threshKernFit,
                          const long imgW, const int fStampWidth, const int hSStampWidth,
                          const int maxSStamps, const int maxStamps, const ushort badMask, const ushort badPixelMask, const ushort skipMask,
                          local int2 *localSubStampCoords, local double *localSubStampValues) {
    int stamp      = get_global_id(0);
    int localStamp = get_local_id(0);
    if (stamp >= maxStamps) return;

    double skyEst = skyEsts[stamp];
    double fwhm = fwhms[stamp];

    double floor = skyEst + threshKernFit * fwhm;
    double dfrac = 0.9;
    
    int2 stampCoords = stampsCoords[stamp];
    int2 stampSize =  stampsSizes[stamp];

    int sstampCounter = 0;
    while(sstampCounter < maxSStamps) {
        double lowestPSFLim = max(floor, skyEst + (threshHigh - skyEst) * dfrac);
        for(int y = 0; y < fStampWidth; y++) {
            int absy = y + stampCoords.y;
            for(int x = 0; x < fStampWidth; x++) {
                int absx = x + stampCoords.x;
                int absCoords  = absx + (absy * imgW);
                
                if ((mask[absCoords] & badMask) > 0) {
                    continue;
                }

                double imgValue = img[absCoords];
                if(imgValue > threshHigh) {
                    mask[absCoords] |= badPixelMask;
                    continue;
                }

                if((imgValue - skyEst) * (1.0 / fwhm) < threshKernFit) {
                    continue;
                }

                if(imgValue > lowestPSFLim) {  // good candidate found
                    double maxVal = 0;
                    int2  maxCoords;
                    maxCoords.x = (int)absx;
                    maxCoords.y = (int)absy;
                    int startX = max(absx - hSStampWidth, stampCoords.x);
                    int startY = max(absy - hSStampWidth, stampCoords.y);
                    int endX   = min(absx + hSStampWidth, stampCoords.x + fStampWidth - 1);
                    int endY   = min(absy + hSStampWidth, stampCoords.y + fStampWidth - 1);
                    
                    for(int ky = startY; ky <= endY; ky++) {
                        for(int kx = startX; kx <= endX; kx++) {
                            
                            int kCoords = kx + (ky * imgW);
                            double kImgValue = img[kCoords];
                            if ((mask[kCoords] & badMask) > 0) {
                                continue;
                            }

                            if(kImgValue >= threshHigh) {
                                mask[kCoords] |= badPixelMask;
                                continue;
                            }
                            if((kImgValue - skyEst) * (1.0 / fwhm) < threshKernFit) {
                                continue;
                            }

                            if(kImgValue > maxVal) {
                                maxVal = kImgValue;
                                maxCoords.x = (int)kx;
                                maxCoords.y = (int)ky;
                            }
                        }
                    }
                    
                    maxVal = checkSStamp(img, mask, skyEst, fwhm, imgW,
                                         maxCoords, hSStampWidth,
                                         stampCoords, stampSize,
                                         threshHigh, threshKernFit,
                                         badMask, badPixelMask);
                    
                    if(maxVal == 0.0) continue;
                
                    localSubStampCoords[localStamp * maxSStamps + sstampCounter] = maxCoords;
                    localSubStampValues[localStamp * maxSStamps + sstampCounter] = maxVal;
                    sstampCounter++;
    
                    int startX2 = max(maxCoords.x - hSStampWidth, stampCoords.x);
                    int startY2 = max(maxCoords.y - hSStampWidth, stampCoords.y);
                    int endX2 = min(maxCoords.x + hSStampWidth, stampCoords.x + stampSize.x - 1);
                    int endY2 = min(maxCoords.y + hSStampWidth, stampCoords.y + stampSize.y - 1);

                    for(int y = startY2; y <= endY2; y++) {
                        for(int x = startX2; x <= endX2; x++) {
                            mask[x + y*imgW] |= skipMask;
                        }
                    }
                }
                if(sstampCounter >= maxSStamps) break;
            }
            if(sstampCounter >= maxSStamps) break;
        }
        if(lowestPSFLim == floor) break;
        dfrac -= 0.2;
    }
        
    sortSubStamps(sstampCounter, 
        &localSubStampCoords[localStamp*maxSStamps], 
        &localSubStampValues[localStamp*maxSStamps]);

    sstampsCounts[stamp] = min(sstampCounter, maxSStamps / 2);
    for(int i = 0; i < sstampCounter; i++) {
        sstampsCoords[stamp * maxSStamps + i] = localSubStampCoords[localStamp*maxSStamps + i];
        sstampsValues[stamp * maxSStamps + i] = localSubStampValues[localStamp*maxSStamps + i];
    }
    for (int i = sstampCounter; i<maxSStamps; i++) {
        sstampsCoords[stamp * maxSStamps + i] = (int2)(INT_MAX, INT_MAX);
        sstampsValues[stamp * maxSStamps + i] = -INFINITY;
    }
}

void kernel markStampsToKeep(global const int *sstampCounts, global int *keepIndeces, global int *keepCounter){
    int stamp = get_global_id(0);
    if (sstampCounts[stamp] > 0) {
        keepIndeces[atomic_inc(keepCounter)] = stamp;
    }
}

void kernel padMarks(global int *keepIndeces, global const int *keepCounter) {
    int stamp = get_global_id(0);
    if (stamp >= *keepCounter) {
        keepIndeces[stamp] = INT_MAX;
    }
}

void exchangeInt(global int *i, global int *j) {
	int k;
	k = *i;
	*i = *j;
	*j = k;
}

void kernel sortMarks(global int *keepIndeces, const int j, const int k) {
    int i = get_global_id(0);

    const int ixj=i^j; // Calculate indexing!
    if ((ixj)>i)
    {
        if ((i&k)==0 && keepIndeces[i] > keepIndeces[ixj]) {
        exchangeInt(&keepIndeces[i], &keepIndeces[ixj]);
        }
        if ((i&k)!=0 && keepIndeces[i] < keepIndeces[ixj]) {
        exchangeInt(&keepIndeces[i], &keepIndeces[ixj]);
        }
    }
}

void kernel removeEmptyStamps(global const int2 *stampCoords, global const int2 *stampSizes,
                              global const double *skyEsts, global const double *fwhms,
                              global const int *subStampCounts,
                              global const int2 *subStampCoords, global const double *subStampValues,
                              global int2 *filteredStampCoords, global int2 *filteredStampSizes,
                              global double *filteredSkyEsts, global double *filteredFwhms,
                              global int *filteredSubStampCounts,
                              global int2 *filteredSubStampCoords, global double *filteredSubStampValues,
                              global const int *keepIndeces, global const int *keepCounter,
                              global int *currentSubStamps, const int maxSStamps) {
    int stamp = get_global_id(0);
    if(stamp >= *keepCounter) return;
    
    int index = keepIndeces[stamp];
    filteredStampCoords[stamp] = stampCoords[index];
    filteredStampSizes[stamp] = stampSizes[index];
    filteredSkyEsts[stamp] = skyEsts[index];
    filteredFwhms[stamp] = fwhms[index];
    
    int sstampCount = subStampCounts[index];
    filteredSubStampCounts[stamp] = sstampCount;
    
    for(int i = 0; i < sstampCount; i++){
        filteredSubStampCoords[maxSStamps*stamp + i] = subStampCoords[maxSStamps*index + i];
        filteredSubStampValues[maxSStamps*stamp + i] = subStampValues[maxSStamps*index + i];
    }
    
    currentSubStamps[stamp] = 0;
} 

void kernel resetSkipMask(global ushort *mask) {
    int id = get_global_id(0);
    mask[id] &= ~(MASK_SKIP_S | MASK_SKIP_T);
}
