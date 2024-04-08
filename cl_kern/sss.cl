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

#define STATS_SIZE (5)
#define STAT_SKY_EST (0)
#define STAT_FWHM (1)
#define STAT_NORM (2)
#define STAT_DIFF (3)
#define STAT_CHI2 (4)

//TODO: Fix swizzling for stamp data

void kernel createStampBounds(global long *stampsCoords, global long *stampsSizes,
                              const int nStampsX, const int nStampsY, const long fullStampWidth, 
                              const long w, const long h) {
    const int id = get_global_id(0);
    const int stampX = id % nStampsX;
    const int stampY = id / nStampsX;


    long startX = stampX * w / nStampsX;
    long startY = stampY * h / nStampsY;
    
    long stopX = min(startX + fullStampWidth, w);
    long stopY = min(startY + fullStampWidth, h);

    long stampW = stopX - startX;
    long stampH = stopY - startY;

    stampsCoords[2*id + 0] = startX;
    stampsCoords[2*id + 1] = startY;
    stampsSizes[2*id + 0] = stampW;
    stampsSizes[2*id + 1] = stampH;
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
                        global const long2 *stampsCoords, global const long2 *stampsSizes,
                        global double *samples, global int *sampleCounts,
                        const long w, const int nSamples) {
    
    int stamp = get_global_id(0);
    long2 stampCoords = stampsCoords[stamp];
    long2 stampSize = stampsSizes[stamp];
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

void kernel sortSamples(global double *paddedSamples, int paddedNSamples, const int j, const int k){
    int id = get_global_id(0);                        
    
    const long i = id % paddedNSamples;
    const long stamp = id / paddedNSamples;
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

void kernel maskStamp(global const double *img, global ushort *mask, 
                      global const long2 *stampCoords, 
                      global double *goodPixels, global int *goodPixelCounts,
                      const int fullStampWidth, const int w, const int h){
    int indexS = get_global_id(0);
    int stamp  = get_global_id(1);

    long2 coords = stampCoords[stamp];

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

    long stampOffset = stamp * get_global_size(0);
    long pixelOffset = atomic_inc(&goodPixelCounts[stamp]);
    
    goodPixels[stampOffset + pixelOffset] = pixel;

}

double checkSStamp(global const double *img, global ushort *mask,
                   const double skyEst, const double fwhm, const long imgW,
                   const int2 sstampCoords, const long hSStampWidth,
                   const long2 stampCoords, const long2 stampSize,
                   const double threshHigh, const double threshKernFit,
                   const ushort badMask, const ushort badPixelMask) {
    double retVal = 0.0;
    int stamp = get_global_id(0);

    long startX = max(sstampCoords.x - hSStampWidth, stampCoords.x);
    long startY = max(sstampCoords.y - hSStampWidth, stampCoords.y);
    long endX   = min(sstampCoords.x + hSStampWidth, stampCoords.x + stampSize.x - 1);
    long endY   = min(sstampCoords.y + hSStampWidth, stampCoords.y + stampSize.y - 1);
    
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
                          global const long2 *stampsCoords, global const long2 *stampsSizes,
                          global const double *stampsStats,
                          global int2 *sstampsCoords, global double *sstampsValues,
                          global int *sstampsCounts,
                          const double threshHigh, const double threshKernFit,
                          const long imgW, const int fStampWidth, const int hSStampWidth,
                          const int maxSStamps, const int maxStamps, const ushort badMask, const ushort badPixelMask, const ushort skipMask,
                          local int2 *localSubStampCoords, local double *localSubStampValues) {
    int stamp      = get_global_id(0);
    int localStamp = get_local_id(0);
    if (stamp >= maxStamps) return;

    double skyEst = stampsStats[STATS_SIZE * stamp + STAT_SKY_EST];
    double fwhm = stampsStats[STATS_SIZE * stamp + STAT_FWHM];

    double floor = skyEst + threshKernFit * fwhm;
    double dfrac = 0.9;
    
    long2 stampCoords = stampsCoords[stamp];
    long2 stampSize =  stampsSizes[stamp];

    int sstampCounter = 0;
    while(sstampCounter < maxSStamps) {
        double lowestPSFLim = max(floor, skyEst + (threshHigh - skyEst) * dfrac);
        for(long y = 0; y < fStampWidth; y++) {
            long absy = y + stampCoords.y;
            for(long x = 0; x < fStampWidth; x++) {
                long absx = x + stampCoords.x;
                long absCoords  = absx + (absy * imgW);
                
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
                    long startX = max(absx - hSStampWidth, stampCoords.x);
                    long startY = max(absy - hSStampWidth, stampCoords.y);
                    long endX   = min(absx + hSStampWidth, stampCoords.x + fStampWidth - 1);
                    long endY   = min(absy + hSStampWidth, stampCoords.y + fStampWidth - 1);
                    
                    for(long ky = startY; ky <= endY; ky++) {
                        for(long kx = startX; kx <= endX; kx++) {
                            
                            long kCoords = kx + (ky * imgW);
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
    
                    long startX2 = max((long)(maxCoords.x - hSStampWidth), stampCoords.x);
                    long startY2 = max((long)(maxCoords.y - hSStampWidth), stampCoords.y);
                    long endX2 = min((long)(maxCoords.x + hSStampWidth), stampCoords.x + stampSize.x - 1);
                    long endY2 = min((long)(maxCoords.y + hSStampWidth), stampCoords.y + stampSize.y - 1);

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

void kernel removeEmptyStamps(global const long2 *stampCoords, global const long2 *stampSizes,
                              global const double *stampStats, global const int *subStampCounts,
                              global const int2 *subStampCoords, global const double *subStampValues,
                              global long2 *filteredStampCoords, global long2 *filteredStampSizes,
                              global double *filteredStampStats, global int *filteredSubStampCounts,
                              global int2 *filteredSubStampCoords, global double *filteredSubStampValues,
                              global const int *keepIndeces, global const int *keepCounter, const int maxKSStamps) {
    int stamp = get_global_id(0);
    if(stamp >= *keepCounter) return;
    
    int index = keepIndeces[stamp];
    filteredStampCoords[stamp] = stampCoords[index];
    filteredStampSizes[stamp] = stampSizes[index];
    filteredStampStats[STATS_SIZE * stamp + STAT_SKY_EST] = stampStats[STATS_SIZE * index + STAT_SKY_EST];
    filteredStampStats[STATS_SIZE * stamp + STAT_FWHM] = stampStats[STATS_SIZE * index + STAT_FWHM];
    filteredStampStats[STATS_SIZE * stamp + STAT_NORM] = stampStats[STATS_SIZE * index + STAT_NORM];
    filteredStampStats[STATS_SIZE * stamp + STAT_DIFF] = stampStats[STATS_SIZE * index + STAT_DIFF];
    filteredStampStats[STATS_SIZE * stamp + STAT_CHI2] = stampStats[STATS_SIZE * index + STAT_CHI2];
    
    filteredSubStampCounts[stamp] = subStampCounts[index];
    
    int sstampCount = subStampCounts[index];
    int maxSStamps = 2 * maxKSStamps;
    for(int i = 0; i < sstampCount; i++){
        filteredSubStampCoords[maxSStamps*stamp + i] = subStampCoords[maxSStamps*index + i];
        filteredSubStampValues[maxSStamps*stamp + i] = subStampValues[maxSStamps*index + i];
    }
} 