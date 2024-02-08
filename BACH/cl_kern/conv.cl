void kernel conv(global const double *convKern, const long convWidth, const long xSteps,
                 global const double *image, global double *outimg,
                 const long w, const long h) {
  int id = get_global_id(0);
  double acc = 0.0;
  long x = id % w;
  long y = id / w;

  long halfConvWidth = convWidth / 2;

  if(x >= halfConvWidth && x < w - halfConvWidth && y >= halfConvWidth &&
     y < h - halfConvWidth) {

    int xS = (x - halfConvWidth) / convWidth;
    int yS = (y - halfConvWidth) / convWidth;

    int convOffset = (xS + yS * xSteps) * convWidth * convWidth;

    for(long j = y - (halfConvWidth); j <= y + halfConvWidth; j++) {
      int jk = y - j + (halfConvWidth);
      for(long i = x - (halfConvWidth); i <= x + halfConvWidth; i++) {
        int ik = x - i + (halfConvWidth);
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
