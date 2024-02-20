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
