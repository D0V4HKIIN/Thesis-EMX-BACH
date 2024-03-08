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
                            const double kernelMean, const double kernelStdev, const double sigKernFit) {
    int stampId = get_global_id(0);

    double diff = fabs((kernelSums[stampId] - kernelMean) / kernelStdev);

    if (diff < sigKernFit) {
        int index = atomic_inc(testStampCount);
        testStampIndices[index] = stampId;
    }
}

void kernel copyTestSubStampsCoords(global const int *in, global const int *testStampIndices,
                                    global int *out,
                                    const long maxSubStamps) {
    int coordId = get_global_id(0);
    int subStampId = get_global_id(1);
    int dstStampId = get_global_id(2);

    int srcStampId = testStampIndices[dstStampId];

    out[2 * (dstStampId * maxSubStamps + subStampId) + coordId] = in[2 * (srcStampId * maxSubStamps + subStampId) + coordId];
}

void kernel copyTestSubStampsCounts(global const int *in, global const int *testStampIndices,
                                    global int *out) {
    int dstStampId = get_global_id(0);

    int srcStampId = testStampIndices[dstStampId];

    out[dstStampId] = in[srcStampId];
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

void kernel createMatrixWeights(global const int2 *subStampCoords, global const int2 *xy,
                                global double *weights,
                                const long width, const long height, const long maxSubStamps, const long count) {
    int k = get_global_id(0);
    int stampId = get_global_id(1);

    int i = xy[k].x;
    int j = xy[k].y;

    int x = subStampCoords[stampId * maxSubStamps].x;
    int y = subStampCoords[stampId * maxSubStamps].y;

    double xf = (x - (width * 0.5f)) / (width * 0.5f);
    double yf = (y - (height * 0.5f)) / (height * 0.5f);

    double a1 = pown(xf, i);
    double a2 = pown(yf, j);

    weights[stampId * count + k] = a1 * a2;
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

void kernel createScProd(const global double *img, const global double *weights, const global double *b, const global double *w, const global int2 *subStampCoords,
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
            double q = 0.0;

            int ssx = subStampCoords[stampId * maxSubStamps].x;
            int ssy = subStampCoords[stampId * maxSubStamps].y;

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
                       global double *kern,
                       const long nPsf, const long kernelWidth) {
    int i = get_global_id(0);

    int count = kernelWidth * kernelWidth;

    if (i < count) {
        double k0 = 0.0;

        // TODO: improve performance with local memory
        for (int p = 0; p < nPsf; p++) {
            k0 += kernCoeffs[p] * kernVec[p * count + i];
        }

        kern[i] = k0;
    }
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
