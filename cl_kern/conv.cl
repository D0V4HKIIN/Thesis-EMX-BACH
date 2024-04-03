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

void kernel createConvMask(global const double *img, global ushort *mask,
                           const int w, const double threshHigh, const double threshLow) {
  int x = get_global_id(0);
  int y = get_global_id(1);

  int id = x + y * w;
  double t = img[id];
  ushort m = 0;

  m |= select(0, MASK_BAD_INPUT | MASK_BAD_PIX_VAL, t == 0.0);
  m |= select(0, MASK_BAD_INPUT | MASK_SAT_PIXEL, t >= threshHigh);
  m |= select(0, MASK_BAD_INPUT | MASK_LOW_PIXEL, t <= threshLow);

  mask[id] = m;
}

void kernel conv(global const double *convKern, const long convWidth, const long xSteps,
                 global const double *image, global double *outimg,
                 global const ushort *convMask, global ushort *outMask, global const double *kernSolution,
                 const long w, const long h, const long bgOrder, const long nBgComp, const double invKernMult) {
  const int id = get_global_id(0);
  double acc = 0.0;
  const long x = id % w;
  const long y = id / w;

  long halfConvWidth = convWidth / 2;

  if(x >= halfConvWidth && x < w - halfConvWidth && y >= halfConvWidth &&
     y < h - halfConvWidth) {

    int xS = (x - halfConvWidth) / convWidth;
    int yS = (y - halfConvWidth) / convWidth;

    int convOffset = (xS + yS * xSteps) * convWidth * convWidth;

    int maskAcc = 0;
    double aks = 0.0;
    double uks = 0.0;

    for(long j = y - halfConvWidth; j <= y + halfConvWidth; j++) {
      int jk = y - j + halfConvWidth;
      for(long i = x - halfConvWidth; i <= x + halfConvWidth; i++) {
        int ik = x - i + halfConvWidth;
        long convIndex = ik + jk * convWidth;
        convIndex += convOffset;
        long imgIndex = i + w * j;

        double kk = convKern[convIndex];
        acc += kk * image[imgIndex];
        maskAcc |= convMask[imgIndex];
        aks += fabs(kk);

        if ((convMask[imgIndex] & MASK_BAD_INPUT) == 0) {
          uks += fabs(kk);
        }
      }
    }

    acc += getBackground(x, y, kernSolution, w, h, bgOrder, nBgComp);
    acc *= invKernMult;

    outimg[id] = acc;

    ushort newMask = convMask[id];

    if ((convMask[id] & MASK_BAD_INPUT) != 0) {
      newMask |= MASK_BAD_OUTPUT;
    }

    if (maskAcc != 0) {
      if ((uks / aks) < 0.99f) {
        newMask |= MASK_BAD_OUTPUT | MASK_BAD_CONV;
      }
      else {
        newMask |= MASK_OK_CONV;
      }
    }
    
    outMask[id] |= newMask;
  } else {
    outimg[id] = 1e-30;
  }
}

void kernel maskAfterConv(global const double *img, global ushort *mask,
                          const int w, const double threshHigh, const double threshLow) {
  int x = get_global_id(0);
  int y = get_global_id(1);

  int id = x + y * w;
  double t = img[id];
  ushort m = 0;

  m |= select(0, MASK_BAD_OUTPUT | MASK_BAD_INPUT | MASK_BAD_PIX_VAL, t == 0.0);
  m |= select(0, MASK_BAD_OUTPUT | MASK_BAD_INPUT | MASK_SAT_PIXEL, t >= threshHigh);
  m |= select(0, MASK_BAD_OUTPUT | MASK_BAD_INPUT | MASK_LOW_PIXEL, t <= threshLow);

  mask[id] |= m;
}
