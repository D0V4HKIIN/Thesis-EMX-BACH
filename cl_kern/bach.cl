void kernel ludcmpBig(global const double *matrix,
                      global double *vv,
                      const int matrixSize) {
    int i = get_global_id(0);
    int stampId = get_global_id(1);

    double v = 0.0;

    if (i > 0) {
        double big = 0.0;

        for (int j = 1; j < matrixSize; j++) {
            double m = fabs(matrix[stampId * matrixSize * matrixSize + i * matrixSize + j]);

            if (m > big) {
                big = m;
            }
        }

        v = 1.0 / big;
    }

    vv[stampId * matrixSize + i] = v;
}

void kernel ludcmpRest(global double *vv, global double *matrix, global int *index,
                       const int matrixSize) {
    int stampId = get_global_id(0);

    int firstMtxId = stampId * matrixSize * matrixSize;
    int firstVId = stampId * matrixSize;
    int firstIId = stampId * matrixSize;
    int maxI = 0;

    // Terrible implementation
    // Should be parallelized a lot better
    for (int j = 1; j < matrixSize; j++) {
        for (int i = 1; i < j; i++) {
            double sum = matrix[firstMtxId + i * matrixSize + j];

            for (int k = 1; k < i; k++) {
                sum -= matrix[firstMtxId + i * matrixSize + k] * matrix[firstMtxId + k * matrixSize + j];
            }

            matrix[firstMtxId + i * matrixSize + j] = sum;
        }

        double big = 0.0;

        for (int i = j; i < matrixSize; i++) {
            double sum = matrix[firstMtxId + i * matrixSize + j];
            
            for(int k = 1; k < j; k++) {
                sum -= matrix[firstMtxId + i * matrixSize + k] * matrix[firstMtxId + k * matrixSize + j];
            }

            matrix[firstMtxId + i * matrixSize + j] = sum;
            double dum = vv[firstVId + i] * fabs(sum);

            if (dum >= big) {
                big = dum;
                maxI = i;
            }
        }
        
        if (j != maxI) {
            for (int k = 1; k < matrixSize; k++) {
                double dum = matrix[firstMtxId + maxI * matrixSize + k];
                matrix[firstMtxId + maxI * matrixSize + k] = matrix[firstMtxId + j * matrixSize + k];
                matrix[firstMtxId + j * matrixSize + k] = dum;
            }
            
            vv[firstVId + maxI] = vv[firstVId + j];
        }

        index[firstIId + j] = maxI;
        matrix[firstMtxId + j * matrixSize + j] = matrix[firstMtxId + j * matrixSize + j] == 0.0 ? 1.0e-20 : matrix[firstMtxId + j * matrixSize + j];

        if(j != matrixSize - 1) {
            double dum = 1.0 / matrix[firstMtxId + j * matrixSize + j];

            for(int i = j + 1; i < matrixSize; i++) {
                matrix[firstMtxId + i * matrixSize + j] *= dum;
            }
        }
    }
}

void kernel lubksb(const global double *matrix, const global int *index,
                   global double *result,
                   const int matrixSize) {
    int stampId = get_global_id(0);

    int firstMtxId = stampId * matrixSize * matrixSize;
    int firstIId = stampId * matrixSize;
    int firstResId = stampId * matrixSize;

    int ii = 0;

    // Also terrible implementation, too little parallelism
    for(int i = 1; i < matrixSize; i++) {
        int ip = index[firstIId + i];
        double sum = result[firstResId + ip];
        result[firstResId + ip] = result[firstResId + i];

        if (ii != 0) {
            for(int j = ii; j <= i - 1; j++) {
                sum -= matrix[firstMtxId + i * matrixSize + j] * result[firstResId + j];
            }
        }
        else if (sum != 0.0) {
            ii = i;
        }

        result[firstResId + i] = sum;
    }

    for(int i = matrixSize - 1; i >= 1; i--) {
        double sum = result[firstResId + i];

        for(int j = i + 1; j < matrixSize; j++) {
            sum -= matrix[firstMtxId + i * matrixSize + j] * result[firstResId + j];
        }

        result[firstResId + i] = sum / matrix[firstMtxId + i * matrixSize + i];
    }
}

void kernel sigmaClipInitMask(global uchar *mask) {
    int id = get_global_id(0);

    mask[id - get_global_offset(0)] = 0;
}

void kernel sigmaClipCalc(global double *sum, global double *sum2,
                          global const double *data, global const uchar *mask,
                          const int count) {
    int gid = get_global_id(0);
    int gidNoOffset = gid - get_global_offset(0);
    
    int lid = get_local_id(0);
    int groupId = get_group_id(0);

    local double localD[32];

    if (gidNoOffset < count) {
        double d = select(0.0, data[gid], (ulong)(mask[gidNoOffset] == 0));

        localD[lid] = d;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        double s = 0.0;
        double s2 = 0.0;

        int localCount = min((int)get_local_size(0), count - gidNoOffset);

        for (int i = 0; i < localCount; i++) {
            double d = localD[i];

            s += d;
            s2 += d * d;
        }

        sum[groupId] = s;
        sum2[groupId] = s2;
    }
}

void kernel sigmaClipMask(global uchar *mask, global int *clipCount,
                          global const double *data,
                          const double invStdDev, const double mean, const double sigClipAlpha) {
    int dataId = get_global_id(0);
    int maskId = dataId - get_global_offset(0);

    if (mask[maskId] == 0) {
        if (fabs(data[dataId] - mean) * invStdDev > sigClipAlpha) {
            mask[maskId] = 1;
            atomic_inc(clipCount);
        }
    }
}
