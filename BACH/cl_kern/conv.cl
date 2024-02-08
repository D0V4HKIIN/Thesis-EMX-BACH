void kernel conv(global const double *convKern, const long convWidth,
                 global const double *image, global double *outimg,
                 const long w, const long h) {
  int id = get_global_id(0);
  double acc = 0.0;
  long x = id % w;
  long y = id / w;

  if(x >= convWidth / 2 && x < w - convWidth / 2 && y >= convWidth / 2 &&
     y < h - convWidth / 2) {
    int xSteps = ceil((double)w / (double)convWidth);

    int xS = (x - convWidth / 2) / convWidth;
    int yS = (y - convWidth / 2) / convWidth;

    int convOffset = (xS + yS * xSteps) * convWidth * convWidth;

    for(long j = y - (convWidth / 2); j <= y + convWidth / 2; j++) {
      int jk = y - j + (convWidth / 2);
      for(long i = x - (convWidth / 2); i <= x + convWidth / 2; i++) {
        int ik = x - i + (convWidth / 2);
        long convIndex = ik + jk * convWidth;
        convIndex += convOffset;
        long imgIndex = i + w * j;
        acc += convKern[convIndex] * image[imgIndex];
      }
    }

    outimg[id] = acc;
  } else {
    outimg[id] = 1e-30;
  }
}
