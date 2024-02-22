void kernel createKernelFilter(global const int *gauss,
                               global const int *kernelX, global const int *kernelY,
                               global const float *bg,
                               global double *filterX, global double *filterY,
                               const long kernelWidth) {
    int id = get_global_id(0);

    int firstFilterId = id * kernelWidth;

    double sumX = 0.0;
    double sumY = 0.0;

    float b = bg[gauss[id]];
    int kx = kernelX[id];
    int ky = kernelY[id];

    int dx = (kx / 2) * 2 - kx;
    int dy = (ky / 2) * 2 - ky;

    for (int i = 0; i < kernelWidth; i++) {
        double x = i - kernelWidth / 2;
        double qe = exp(-x * x * b);

        double fx = qe * pow(x, kx);
        double fy = qe * pow(x, ky);

        filterX[firstFilterId + i] = fx;
        filterY[firstFilterId + i] = fy;

        sumX += fx;
        sumY += fy;
    }

    if (dx == 0 && dy == 0) {
        double invSumX = 1.0 / sumX;
        double invSumY = 1.0 / sumY;

        for (int i = 0; i < kernelWidth; i++) {
            filterX[firstFilterId + i] *= invSumX;
            filterY[firstFilterId + i] *= invSumY;
        }
    }
}

void kernel createKernelVector(global const int *kernelX, global const int *kernelY,
                               global const double *filterX, global const double *filterY,
                               global double *vec,
                               const long kernelWidth) {
    int id = get_global_id(0);

    int n = id / (kernelWidth * kernelWidth);
    int u = (id % (kernelWidth * kernelWidth)) % kernelWidth;
    int v = (id % (kernelWidth * kernelWidth)) / kernelWidth;

    int kx = kernelX[n];
    int ky = kernelY[n];

    int dx = (kx / 2) * 2 - kx;
    int dy = (ky / 2) * 2 - ky;

    double vv = filterX[n * kernelWidth + u] * filterY[n * kernelWidth + v];
    
    if (dx == 0 && dy == 0 && n > 0) {
        // Subtract vec[u + v * kernelWidth], however, it is not guaranteed
        // that the value is available, so the expression is inlined
        vv -= filterX[u] * filterY[v];
    }
    
    vec[id] = vv;
}

void kernel convStampY(global const double *img, global const int *subStampCoords,
                       global const double *filterY,
                       global float *tmp,
                       const long kernelWidth, const long subStampWidth,
                       const long width, const long gaussCount, const long maxSubStamps) {
    int stampId = get_global_id(0);
    int n = get_global_id(1);
    int pixel = get_global_id(2);

    int halfKernWidth = kernelWidth / 2;
    int halfSubStampWidth = subStampWidth / 2;

    int ssx = subStampCoords[2 * stampId * maxSubStamps + 0];
    int ssy = subStampCoords[2 * stampId * maxSubStamps + 1];

    int xHalfSize = halfSubStampWidth + halfKernWidth;
    int yHalfSize = halfSubStampWidth;
    int pixelCount = (2 * xHalfSize + 1) * (2 * yHalfSize + 1);

    int j = ssy + pixel / (2 * xHalfSize + 1) - yHalfSize;
    int i = ssx + pixel % (2 * xHalfSize + 1) - xHalfSize;

    float v = 0.0;

    for (int y = -halfKernWidth; y <= halfKernWidth; y++) {
        int imgIndex = i + (j + y) * width;
        float imgV = img[imgIndex];
        v += imgV * filterY[n * kernelWidth + halfKernWidth - y];
    }

    tmp[stampId * gaussCount * pixelCount + n * pixelCount + pixel] = v;
}

void kernel convStampX(global const float *tmp, global const double *filterX,
                       global double *w, 
                       const long kernelWidth, const long subStampWidth,
                       const long wRows, const long wColumns,
                       const long gaussCount) {
    int stampId = get_global_id(0);
    int n = get_global_id(1);
    int pixel = get_global_id(2);

    int halfKernWidth = kernelWidth / 2;
    int halfSubStampWidth = subStampWidth / 2;

    int j = pixel / subStampWidth - halfSubStampWidth;
    int i = pixel % subStampWidth - halfSubStampWidth;

    int subWidth = kernelWidth + subStampWidth - 1;
    double w0 = 0.0f;

    int tmpPixelStride = (2 * (halfSubStampWidth + halfKernWidth) + 1) * (2 * halfSubStampWidth + 1);
    int firstTmpId = stampId * gaussCount * tmpPixelStride + n * tmpPixelStride;

    for(int x = -halfKernWidth; x <= halfKernWidth; x++) {
        int index = (i + x) + subWidth / 2 + (j + halfSubStampWidth) * subWidth;
        w0 += tmp[firstTmpId + index] * filterX[n * kernelWidth + halfKernWidth - x];
    }

    w[stampId * wRows * wColumns + n * wColumns + pixel] = w0;
}

void kernel convStampOdd(global const int *kernelX, global const int *kernelY,
                         global double *w,
                         const long wRows, const long wColumns) {
    int stampId = get_global_id(0);
    int n = get_global_id(1);
    int pixel = get_global_id(2);

    int x = kernelX[n];
    int y = kernelY[n];
    
    int dx = (x / 2) * 2 - x;
    int dy = (y / 2) * 2 - y;

    // Not optimal, a lot of threads will do nothing
    if (n > 0 && dx == 0 && dy == 0) {
        w[stampId * wRows * wColumns + n * wColumns + pixel] -= w[stampId * wRows * wColumns + pixel];
    }
}

void kernel createQ(global const double *w,
                    global double *q,
                    const long wRows, const long wColumns,
                    const long qRows, const long qColumns,
                    const long subStampWidth) {
    int stampId = get_global_id(0);
    int i = get_global_id(1);
    int j = get_global_id(2);

    double q0 = 0.0;

    // Can be optimized as ~50% of the threads are idle and outputs 0.0
    if (i > 0 && j > 0 && j <= i) {
        for(int k = 0; k < subStampWidth * subStampWidth; k++) {
          double w0 = w[stampId * wRows * wColumns + (i - 1) * wColumns + k];
          double w1 = w[stampId * wRows * wColumns + (j - 1) * wColumns + k];
          q0 += w0 * w1;
        }
    }

    q[stampId * qRows * qColumns + i * qColumns + j] = q0;
}

void kernel createB(global const int *subStampCoords,
                    global const double *img, global const double *w,
                    global double *b,
                    const long wRows, const long wColumns, const long bCount,
                    const long subStampWidth, const long maxSubStamps,
                    const long width) {
    int stampId = get_global_id(0);
    int i = get_global_id(1);

    double p0 = 0.0;

    if (i > 0) {
        int halfSubStampWidth = subStampWidth / 2;

        int ssx = subStampCoords[2 * stampId * maxSubStamps + 0];
        int ssy = subStampCoords[2 * stampId * maxSubStamps + 1];

        for(int x = -halfSubStampWidth; x <= halfSubStampWidth; x++) {
            for(int y = -halfSubStampWidth; y <= halfSubStampWidth; y++) {
                int k = x + halfSubStampWidth + subStampWidth * (y + halfSubStampWidth);
                int imgIndex = x + ssx + (y + ssy) * width;

                p0 += w[stampId * wRows * wColumns + (i - 1) * wColumns + k] * img[imgIndex];
            }
        }
    }

    b[stampId * bCount + i] = p0;
}
