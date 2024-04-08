void kernel createKernelFilter(global const int *gauss,
                               global const int2 *kernelXy,
                               global const float *bg,
                               global double *filterX, global double *filterY,
                               const int kernelWidth) {
    int id = get_global_id(0);

    int firstFilterId = id * kernelWidth;

    double sumX = 0.0;
    double sumY = 0.0;

    float b = bg[gauss[id]];
    int kx = kernelXy[id].x;
    int ky = kernelXy[id].y;

    int dx = (kx / 2) * 2 - kx;
    int dy = (ky / 2) * 2 - ky;

    for (int i = 0; i < kernelWidth; i++) {
        double x = i - kernelWidth / 2;
        double qe = exp(-x * x * b);

        double fx = qe * pown(x, kx);
        double fy = qe * pown(x, ky);

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

void kernel createKernelVector(global const int2 *kernelXy,
                               global const double *filterX, global const double *filterY,
                               global double *vec,
                               const int kernelWidth) {
    int u = get_global_id(0);
    int v = get_global_id(1);
    int n = get_global_id(2);

    int kx = kernelXy[n].x;
    int ky = kernelXy[n].y;

    int dx = (kx / 2) * 2 - kx;
    int dy = (ky / 2) * 2 - ky;

    double vv = filterX[n * kernelWidth + u] * filterY[n * kernelWidth + v];
    
    if (dx == 0 && dy == 0 && n > 0) {
        // Subtract vec[u + v * kernelWidth], however, it is not guaranteed
        // that the value is available, so the expression is inlined
        vv -= filterX[u] * filterY[v];
    }
    
    vec[n * kernelWidth * kernelWidth + v * kernelWidth + u] = vv;
}

void kernel convStampY(global const double *img, global const int2 *subStampCoords, global const int *currentSubStamps, global const int *subStampCounts,
                       global const double *filterY,
                       global float *tmp,
                       const int kernelWidth, const int subStampWidth,
                       const int width, const int gaussCount, const int maxSubStamps) {
    int pixel = get_global_id(0);
    int n = get_global_id(1);
    int stampId = get_global_id(2);

    int ssIndex = currentSubStamps[stampId];
    int ssCount = subStampCounts[stampId];

    int halfKernWidth = kernelWidth / 2;
    int halfSubStampWidth = subStampWidth / 2;

    int xHalfSize = halfSubStampWidth + halfKernWidth;
    int yHalfSize = halfSubStampWidth;
    int pixelCount = (2 * xHalfSize + 1) * (2 * yHalfSize + 1);

    float v = 0.0;

    if (ssIndex < ssCount) {
        int ssx = subStampCoords[stampId * maxSubStamps + ssIndex].x;
        int ssy = subStampCoords[stampId * maxSubStamps + ssIndex].y;

        int j = ssy + pixel / (2 * xHalfSize + 1) - yHalfSize;
        int i = ssx + pixel % (2 * xHalfSize + 1) - xHalfSize;

        for (int y = -halfKernWidth; y <= halfKernWidth; y++) {
            int imgIndex = i + (j + y) * width;
            float imgV = img[imgIndex];
            v += imgV * filterY[n * kernelWidth + halfKernWidth - y];
        }
    }

    tmp[stampId * gaussCount * pixelCount + n * pixelCount + pixel] = v;
}

void kernel convStampX(global const float *tmp, global const double *filterX,
                       global double *w, 
                       const int kernelWidth, const int subStampWidth,
                       const int wRows, const int wColumns,
                       const int gaussCount) {
    int pixel = get_global_id(0);
    int n = get_global_id(1);
    int stampId = get_global_id(2);

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

void kernel convStampOdd(global const int2 *kernelXy,
                         global double *w,
                         const int wRows, const int wColumns) {
    int pixel = get_global_id(0);
    int n = get_global_id(1);
    int stampId = get_global_id(2);

    int x = kernelXy[n].x;
    int y = kernelXy[n].y;
    
    int dx = (x / 2) * 2 - x;
    int dy = (y / 2) * 2 - y;

    // Not optimal, a lot of threads will do nothing
    if (dx == 0 && dy == 0) {
        w[stampId * wRows * wColumns + n * wColumns + pixel] -= w[stampId * wRows * wColumns + pixel];
    }
}

void kernel convStampBg(global const int2 *subStampCoords, global const int *currentSubStamps, global const int *subStampCounts, global const int2 *bgXY,
                        global double *w,
                        const int width, const int height,
                        const int subStampWidth,
                        const int wRows, const int wColumns,
                        const int gaussCount, const int maxSubStamps) {
    int pixel = get_global_id(0);
    int bgId = get_global_id(1);
    int stampId = get_global_id(2);

    int ssIndex = currentSubStamps[stampId];
    int ssCount = subStampCounts[stampId];

    double a = 0.0;

    if (ssIndex < ssCount) {
        int ssx = subStampCoords[stampId * maxSubStamps + ssIndex].x;
        int ssy = subStampCoords[stampId * maxSubStamps + ssIndex].y;

        int halfSubStampWidth = subStampWidth / 2;

        int x = ssx - halfSubStampWidth + pixel % subStampWidth;
        int y = ssy - halfSubStampWidth + pixel / subStampWidth;

        double xf = (x - (width * 0.5f)) / (width * 0.5f);
        double yf = (y - (height * 0.5f)) / (height * 0.5f);

        int j = bgXY[bgId].x;
        int k = bgXY[bgId].y;

        double ax = pown(xf, j);
        double ay = pown(yf, k);

        a = ax * ay;
    }

    w[stampId * wRows * wColumns + (bgId + gaussCount) * wColumns + pixel] = a;
}

void kernel createQ(global const double *w,
                    global double *q,
                    const int wRows, const int wColumns,
                    const int qRows, const int qColumns,
                    const int subStampWidth) {
    int j = get_global_id(0);
    int i = get_global_id(1);
    int stampId = get_global_id(2);

    double q0 = 0.0;

    if (i > 0 && j > 0 && j <= i) {
        for(int k = 0; k < subStampWidth * subStampWidth; k++) {
          double w0 = w[stampId * wRows * wColumns + (i - 1) * wColumns + k];
          double w1 = w[stampId * wRows * wColumns + (j - 1) * wColumns + k];
          q0 += w0 * w1;
        }
    }

    q[stampId * qRows * qColumns + i * qColumns + j] = q0;
}

void kernel createB(global const int2 *subStampCoords, global const int *currentSubStamps, global const int *subStampCounts,
                    global const double *img, global const double *w,
                    global double *b,
                    const int wRows, const int wColumns, const int bCount,
                    const int subStampWidth, const int maxSubStamps,
                    const int width) {
    int i = get_global_id(0);
    int stampId = get_global_id(1);

    int ssIndex = currentSubStamps[stampId];
    int ssCount = subStampCounts[stampId];

    double p0 = 0.0;

    if (ssIndex < ssCount && i > 0) {
        int halfSubStampWidth = subStampWidth / 2;

        int ssx = subStampCoords[stampId * maxSubStamps + ssIndex].x;
        int ssy = subStampCoords[stampId * maxSubStamps + ssIndex].y;

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
