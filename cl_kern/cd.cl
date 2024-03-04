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
