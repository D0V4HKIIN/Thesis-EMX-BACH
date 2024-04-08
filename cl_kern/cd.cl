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

void kernel createTestVec(global const double* b,
                          global double *vec,
                          const long bCount) {
    int id = get_global_id(0);
    int stampId = get_global_id(1);

    double v = 0.0;

    if (id > 0) {
        v = b[stampId * bCount + id];
    }

    vec[stampId * bCount + id] = v;
}

void kernel createTestMat(global const double *q,
                          global double *matrix,
                          const long qCount) {
    int j = get_global_id(0);
    int i = get_global_id(1);
    int stampId = get_global_id(2);

    double v = 0.0;

    if (i > 0 && j > 0) {
        v = q[stampId * qCount * qCount + max(i, j) * qCount + min(i, j)];
    }

    matrix[stampId * qCount * qCount + i * qCount + j] = v;
}

void kernel saveKernelSums(global const double *vec,
                           global double *kernelSums,
                           const int matrixSize) {
    int stampId = get_global_id(0);

    kernelSums[stampId] = vec[stampId * matrixSize + 1];
}

void kernel genCdTestStamps(global const double *kernelSums,
                            global int *testStampIndices, global int *testStampCount,
                            const double kernelMean, const double kernelStdev, const double sigKernFit, const int stampCount) {
    int stampId = get_global_id(0);
    int lstampId = get_local_id(0);

    local int localTestStampCount;

    if (lstampId == 0) {
        localTestStampCount = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    bool keepStamp = false;
    int localIndex = 0;

    if (stampId < stampCount) {
        double diff = fabs((kernelSums[stampId] - kernelMean) / kernelStdev);
        keepStamp = diff < sigKernFit;

        if (keepStamp) {
            localIndex = atomic_inc(&localTestStampCount);
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    local int firstGlobalId;

    if (lstampId == 0) {
        firstGlobalId = atomic_add(testStampCount, localTestStampCount);
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);

    if (keepStamp) {
        int index = firstGlobalId + localIndex;
        testStampIndices[index] = stampId;
    }
}

void kernel copyTestSubStampsCoords(global const int2 *in, global const int *testStampIndices,
                                    global int2 *out,
                                    const long maxSubStamps) {
    int subStampId = get_global_id(0);
    int dstStampId = get_global_id(1);

    int srcStampId = testStampIndices[dstStampId];

    out[dstStampId * maxSubStamps + subStampId] = in[srcStampId * maxSubStamps + subStampId];
}

void kernel copyTestSubStampsCounts(global const int *inCurrents, global const int *inCounts, global const int *testStampIndices,
                                    global int *outCurrents, global int *outCounts) {
    int dstStampId = get_global_id(0);

    int srcStampId = testStampIndices[dstStampId];

    outCurrents[dstStampId] = inCurrents[srcStampId];
    outCounts[dstStampId] = inCounts[srcStampId];
}

void kernel copyTestStampsW(global const double *in, global const int *testStampIndices,
                            global double *out,
                            const long wRows, const long wColumns) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    int dstStampId = get_global_id(2);

    int srcStampId = testStampIndices[dstStampId];

    out[dstStampId * wColumns * wRows + j * wColumns + i] = in[srcStampId * wColumns * wRows + j * wColumns + i];
}

void kernel copyTestStampsQ(global const double *in, global const int *testStampIndices,
                            global double *out,
                            const long qCount) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    int dstStampId = get_global_id(2);

    int srcStampId = testStampIndices[dstStampId];

    out[dstStampId * qCount * qCount + j * qCount + i] = in[srcStampId * qCount * qCount + j * qCount + i];
}

void kernel copyTestStampsB(global const double *in, global const int *testStampIndices,
                            global double *out,
                            const long bCount) {
    int i = get_global_id(0);
    int dstStampId = get_global_id(1);

    int srcStampId = testStampIndices[dstStampId];

    out[dstStampId * bCount + i] = in[srcStampId * bCount + i];
}

void kernel createMatrixWeights(global const int2 *subStampCoords, global const int *currentSubStamps, global const int *subStampCounts, global const int2 *xy,
                                global double *weights,
                                const long width, const long height, const long maxSubStamps, const long count) {
    int k = get_global_id(0);
    int stampId = get_global_id(1);

    int ssIndex = currentSubStamps[stampId];
    int ssCount = subStampCounts[stampId];

    double a = 0.0;

    if (ssIndex < ssCount) {
        int i = xy[k].x;
        int j = xy[k].y;

        int x = subStampCoords[stampId * maxSubStamps + ssIndex].x;
        int y = subStampCoords[stampId * maxSubStamps + ssIndex].y;

        double xf = (x - (width * 0.5f)) / (width * 0.5f);
        double yf = (y - (height * 0.5f)) / (height * 0.5f);

        double a1 = pown(xf, i);
        double a2 = pown(yf, j);

        a = a1 * a2;
    }

    weights[stampId * count + k] = a;
}

void kernel createMatrix(global const double *weights, global const double *w, global const double *q,
                         global double *matrix,
                         const long stampCount, const long matrixSize, const long pixStamp, const long nComp1, const long nComp2,
                         const long wRows, const long wColumns, const long qCount) {
    int column = get_global_id(0);
    int row = get_global_id(1);

    int nComp = nComp1 * nComp2;

    double m0 = 0.0;

    if (column == 1 && row == 1) {
        for (int stampId = 0; stampId < stampCount; stampId++) {
            double q0 = q[stampId * qCount * qCount + 1 * qCount + 1];
            m0 += q0;
        }
    }
    else if (row > nComp + 1 && column > nComp + 1) {
        int j = max(row, column);
        int i = min(row, column);

        int iBg = j - (nComp + 2);
        int jBg = i - (nComp + 2);
        int iVecBg = nComp1 + iBg + 1;

        for (int stampId = 0; stampId < stampCount; stampId++) {
            double q = 0.0;

            for (int k = 0; k < pixStamp; k++) {
                double w0 = w[stampId * wRows * wColumns + iVecBg * wColumns + k];
                double w1 = w[stampId * wRows * wColumns + (nComp1 + jBg + 1) * wColumns + k];
                q += w0 * w1;
            }

            m0 += q;
        }
    }
    else if (min(column, row) == 1 && max(column, row) > nComp + 1) {
        int j = max(row, column);
        int i = min(row, column);
        
        int iBg = j - (nComp + 2);
        int iVecBg = nComp1 + iBg + 1;

        for (int stampId = 0; stampId < stampCount; stampId++) {
            double p0 = 0.0;

            for (int k = 0; k < pixStamp; k++) {
                double w0 = w[stampId * wRows * wColumns + 0 * wColumns + k];
                double w1 = w[stampId * wRows * wColumns + iVecBg * wColumns + k];
                p0 += w0 * w1;
            }

            m0 += p0;
        }
    }
    else if (max(column, row) > nComp + 1 && min(column, row) >= 2) {
        int j = max(row, column);
        int i = min(row, column);

        int iBg = j - (nComp + 2);
        int iVecBg = nComp1 + iBg + 1;

        int jj = i - 1;

        int i1 = ((jj - 1) / nComp2) + 1;
        int i2 = (jj - 1) % nComp2;

        for (int stampId = 0; stampId < stampCount; stampId++) {
            double p0 = 0.0;
            
            for (int k = 0; k < pixStamp; k++) {
                double w0 = w[stampId * wRows * wColumns + i1 * wColumns + k];
                double w1 = w[stampId * wRows * wColumns + iVecBg * wColumns + k];
                p0 += w0 * w1;
            }
            
            m0 += p0 * weights[stampId * nComp2 + i2];
        }
    }
    else if (min(column, row) == 1 && max(column, row) >= 2) {
        int i = max(row, column) - 2;
        
        int i1 = i / nComp2;
        int i2 = i - i1 * nComp2;

        for (int stampId = 0; stampId < stampCount; stampId++) {
            double w0 = weights[stampId * nComp2 + i2];
            double q0 = q[stampId * qCount * qCount + (i1 + 2) * qCount + 1];
            m0 += w0 * q0;
        }
    }
    else if (row >= 2 && column >= 2) {
        int j = column - 2;
        int i = row - 2;
        
        int ti = max(i, j);
        int tj = min(i, j);
        
        int i1 = ti / nComp2;
        int i2 = ti - i1 * nComp2;
        int j1 = tj / nComp2;
        int j2 = tj - j1 * nComp2;

        for (int stampId = 0; stampId < stampCount; stampId++) {
            double w0 = weights[stampId * nComp2 + i2];
            double w1 = weights[stampId * nComp2 + j2];
            double q0 = q[stampId * qCount * qCount + (i1 + 2) * qCount + (j1 + 2)];
            m0 += w0 * w1 * q0;
        }
    }

    matrix[row * matrixSize + column] = m0;
}

void kernel createScProd(const global double *img, const global double *weights, const global double *b, const global double *w,
                         const global int2 *subStampCoords, const global int *currentSubStamps, const global int *subStampCounts,
                         global double *res,
                         const long width, const long stampCount, const long nComp1, const long nComp2, const long nBGComp,
                         const long bCount, const long wRows, const long wColumns, const long subStampWidth, const long maxSubStamps) {
    int id = get_global_id(0);

    long nComp = nComp1 * nComp2;
    long halfSubStampWidth = subStampWidth / 2;
                        
    double r0 = 0.0;
    if (id >= 2 && id < nComp + 2) {
        int i = (id - 2) / nComp2 + 1;
        int j = (id - 2) % nComp2;

        for (int stampId = 0; stampId < stampCount; stampId++) {
            double p0 = b[stampId * bCount + i + 1];
            double w0 = weights[stampId * nComp2 + j];
            r0 += p0 * w0;
        }
    }
    else if (id >= nComp + 2 && id < nComp + 2 + nBGComp) {
        int bgIndex = id - (nComp + 2);

        for (int stampId = 0; stampId < stampCount; stampId++) {            
            int ssIndex = currentSubStamps[stampId];
            int ssCount = subStampCounts[stampId];

            if (ssIndex < ssCount) {
                double q = 0.0;

                int ssx = subStampCoords[stampId * maxSubStamps + ssIndex].x;
                int ssy = subStampCoords[stampId * maxSubStamps + ssIndex].y;

                for (int y = -halfSubStampWidth; y <= halfSubStampWidth; y++) {
                    for (int x = -halfSubStampWidth; x <= halfSubStampWidth; x++) {
                        int index = x + halfSubStampWidth + subStampWidth * (y + halfSubStampWidth);

                        double w0 = w[stampId * wRows * wColumns + (nComp1 + bgIndex + 1) * wColumns + index];
                        double i0 = img[x + ssx + (y + ssy) * width];
                        q += w0 * i0;
                    }
                }

                r0 += q;
            }
        }
    }
    else if (id == 1) {
        for (int stampId = 0; stampId < stampCount; stampId++) {
            r0 += b[stampId * bCount + 1];
        }
    }

    res[id] = r0;
}

void kernel makeKernelCoeffs(const global double *kernSol,
                             global double *coeffs,
                             const long kernelOrder, const long kernXyCount, const double xf, const double yf) {
    int i = get_global_id(0);

    double c0 = 0.0;

    if (i == 0) {
        c0 = kernSol[1];
    }
    else {
        int k = 2 + (i - 1) * kernXyCount;
        double aX = 1.0;

        for (int x = 0; x <= kernelOrder; x++) {
            double aY = 1.0;

            for (int y = 0; y <= kernelOrder - x; y++) {
                double s0 = kernSol[k++];
                c0 += s0 * aX * aY;

                aY *= yf;
            }

            aX *= xf;
        }
    }

    coeffs[i] = c0;
}

void kernel makeKernel(const global double *kernCoeffs, const global double *kernVec,
                       global double *kern, local double *localCoeffs,
                       const int nPsf, const int kernelWidth) {
    int i = get_global_id(0);

    int li = get_local_id(0);
    int lsize = get_local_size(0);

    int count = kernelWidth * kernelWidth;

    double k0 = 0.0;

    for (int p0 = 0; p0 < nPsf; p0 += lsize) {
        if (p0 + li < nPsf) {
            localCoeffs[li] = kernCoeffs[p0 + li];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (i < count) {
            int pCount = min(lsize, nPsf - p0);

            for (int pi = 0; pi < pCount; pi++) {
                int p = p0 + pi;
                k0 += localCoeffs[pi] * kernVec[p * count + i];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    kern[i] = k0;
}

void kernel sumKernel(const global double *in,
                      global double *out,
                      const long count) {
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    
    int outid = count / 32;

    local double localKern[32];

    if (gid < count) {
        localKern[lid] = in[gid];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        int localCount = min(32, (int)count - gid);
        double sum = 0.0;

        for (int i = 0; i < localCount; i++) {
            sum += localKern[i];
        }

        out[gid / 32] = sum;
    }
}

double getBackground(const long x, const long y, global const double *sol, const long width, const long height,
                     const long bgOrder, const long nBgComp) {
    double xf = (x - 0.5 * width) / (0.5 * width);
    double yf = (y - 0.5 * height) / (0.5 * height);

    double bg = 0.0;
    int k = nBgComp + 1;

    double ax = 1.0;

    for (int i = 0; i <= bgOrder; i++) {
        double ay = 1.0;

        for (int j = 0; j <= bgOrder - i; j++) {
            bg += sol[k++] * ax * ay;

            ay *= yf;
        }

        ax *= xf;
    }

    return bg;
}

void kernel calcSigBg(global const double *sol, global const int2 *subStampCoords, global const int *currentSubStamps, global const int *subStampCounts,
                      global double *bgs,
                      const long maxSubStamps, const long width, const long height, const long bgOrder, const long nBgComp) {
    int stampId = get_global_id(0);

    int ssIndex = currentSubStamps[stampId];
    int ssCount = subStampCounts[stampId];

    double bg = 0.0;

    if (ssIndex < ssCount) {
        int ssx = subStampCoords[stampId * maxSubStamps + ssIndex].x;
        int ssy = subStampCoords[stampId * maxSubStamps + ssIndex].y;

        bg = getBackground(ssx, ssy, sol, width, height, bgOrder, nBgComp);
    }

    bgs[stampId] = bg;
}

void kernel makeModel(global const double *w, global const double *kernSol, global const int2 *subStampCoords, global const int *currentSubStamps, global const int *subStampCounts,
                      global float *model,
                      const long nPsf, const long kernelOrder, const long wRows, const long wColumns, const long maxSubStamps,
                      const long width, const long height, const long modelSize) {
    int j = get_global_id(0);
    int stampId = get_global_id(1);

    int ssIndex = currentSubStamps[stampId];
    int ssCount = subStampCounts[stampId];

    float m0 = 0.0;

    if (ssIndex < ssCount) {
        int ssx = subStampCoords[stampId * maxSubStamps + ssIndex].x;
        int ssy = subStampCoords[stampId * maxSubStamps + ssIndex].y;

        double xf = (ssx - width * 0.5) / (width * 0.5);
        double yf = (ssy - height * 0.5) / (height * 0.5);

        m0 = kernSol[1] * w[stampId * wRows * wColumns + j];

        for (int i = 1, k = 2; i < nPsf; i++) {
            double coeff = 0.0;
            double ax = 1.0;

            for (int x = 0; x <= kernelOrder; x++) {
                double ay = 1.0;

                for (int y = 0; y <= kernelOrder - x; y++) {
                    double s0 = kernSol[k++]; 
                    coeff += s0 * ax * ay;
                    ay *= yf;
                }

                ax *= xf;
            }

            m0 += coeff * w[stampId * wRows * wColumns + i * wColumns + j];
        }
    }    

    model[stampId * modelSize + j] = m0;
}

void kernel calcSig(global const float *model, global const double *bg, global const double *tImg, global const double *sImg,
                    global const int2 *subStampCoords, global const int *currentSubStamps, global const int *subStampCounts,
                    global double *sig, global int *sigCount, global ushort *mask,
                    const long width, const long subStampWidth, const long maxSubStamps, const long modelSize, const long reduceCount) {
    int gi = get_global_id(0);
    int stampId = get_global_id(1);

    int li = get_local_id(0);

    int ssIndex = currentSubStamps[stampId];
    int ssCount = subStampCounts[stampId];

    int gx = gi % subStampWidth;
    int gy = gi / subStampWidth;

    local double localSig[32];
    local int localSigCount;

    if (li == 0) {
        localSigCount = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    double s0 = 0.0;

    if (ssIndex < ssCount && gx < subStampWidth && gy < subStampWidth) {
        int ssx = subStampCoords[stampId * maxSubStamps + ssIndex].x;
        int ssy = subStampCoords[stampId * maxSubStamps + ssIndex].y;

        int absX = gx - subStampWidth / 2 + ssx;
        int absY = gy - subStampWidth / 2 + ssy;
        
        int intIndex = gx + gy * subStampWidth;
        int absIndex = absX + absY * width;

        double tDat = model[stampId * modelSize + intIndex];
        double sDat = sImg[absIndex];
        double diff = tDat - sDat + bg[stampId];

        if ((mask[absIndex] & MASK_BAD_INPUT) == 0 && fabs(sDat) > 1e-10) {
            if (isnan(tDat) || isnan(sDat)) {
                mask[absIndex] |= MASK_NAN_PIXEL;
            }
            else {
                atomic_inc(&localSigCount);
                s0 = diff * diff / (fabs(tImg[absIndex]) + fabs(sDat));
            }
        }
    }

    localSig[li] = s0;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (li == 0) {
        double localSum = 0.0;

        for (int i = 0; i < 32; i++) {
            localSum += localSig[i];
        }

        int outId = stampId * reduceCount + gi / get_local_size(0);        
        sig[outId] = localSum;
        sigCount[outId] = localSigCount;
    }
}

void kernel reduceSig(global const double *in, global const int *inCount,
                      global double *out, global int *outCount,
                      const long count, const long nextCount) {
    int lid = get_local_id(0);
    int gid = get_global_id(0);
    int groupId = get_group_id(0);
    int stampId = get_global_id(1);

    local double localSig[32];
    local int localCount;

    if (lid == 0) {
        localCount = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (gid < count) {
        int inId = stampId * count + gid;
        localSig[lid] = in[inId];
        atomic_add(&localCount, inCount[inId]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        double sum = 0.0;
        int c = min(32, (int)count - gid);

        for (int i = 0; i < c; i++) {
            sum += localSig[i];
        }

        int outId = stampId * nextCount + groupId;

        if (nextCount == 1) {
            // Calculate average from sum
            if (localCount == 0) {
                sum = -1.0;
            }
            else {
                sum /= localCount;

                if (sum >= 1e10) {
                    sum = -1.0;
                }
            }

            out[outId] = sum;
        }
        else
        {
            out[outId] = sum;
            outCount[outId] = localCount;
        }
    }
}

void kernel removeBadSigs(global const double *in,
                          global double *out, global int *sigCounter,
                          const long inCount) {
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int groupId = get_group_id(0);

    local double localSigs[16];
    local int localCount;
    local int firstOutId;

    if (lid == 0) {
        localCount = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    int localOutId = -1;

    if (gid < inCount) {
        double sig = in[gid];

        if (sig >= 0.0) {
            int localOutId = atomic_inc(&localCount);
            localSigs[localOutId] = sig;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        firstOutId = atomic_add(sigCounter, localCount);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid < localCount) {
        out[firstOutId + lid] = localSigs[lid];
    }
}
