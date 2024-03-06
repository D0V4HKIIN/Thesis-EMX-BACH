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

void kernel createMatrixWeights(global const int *subStampCoords, global const int *xy,
                                global double *weights,
                                const long width, const long height, const long maxSubStamps, const long count) {
    int k = get_global_id(0);
    int stampId = get_global_id(1);

    int i = xy[2 * k + 0];
    int j = xy[2 * k + 1];

    int x = subStampCoords[2 * stampId * maxSubStamps + 0];
    int y = subStampCoords[2 * stampId * maxSubStamps + 1];

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
