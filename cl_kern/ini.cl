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

void kernel maskInput(global const double *tmplImg, global const double *sciImg, global ushort *mask,
                      const long w, const long h, const long borderSize,
                      const double threshHigh, const double threshLow) {
  const int id = get_global_id(0);
  const long x = id % w;
  const long y = id / w;

  if (x >= 0 && x < w && y >= 0 && y < h) {
    ushort m = 0;

    double t = tmplImg[id];
    double s = sciImg[id];

    m |= (t == 0.0 || s == 0.0) * (MASK_BAD_INPUT | MASK_BAD_PIX_VAL);
    m |= (t >= threshHigh || s >= threshHigh) * (MASK_BAD_INPUT | MASK_SAT_PIXEL);
    m |= (t <= threshLow || s <= threshLow) * (MASK_BAD_INPUT | MASK_LOW_PIXEL);
    m |= (x < borderSize || x >= w - borderSize || y < borderSize || y >= h - borderSize) *
      (MASK_BAD_PIXEL_S | MASK_BAD_PIXEL_T);

    mask[id] = m;
  }
}

void kernel spreadMask(global ushort *mask, const long w, const long h, const long spreadWidth) {
  const int id = get_global_id(0);
  const long x = id % w;
  const long y = id / w;

  if (x >= 0 && x < w && y >= 0 && y < h) {
    if ((mask[id] & MASK_BAD_INPUT) != 0) {
      long w2 = spreadWidth / 2;

      long sx = max(x - w2, 0l);
      long sy = max(y - w2, 0l);
      long ex = min(x + w2, w - 1);
      long ey = min(y + w2, h - 1);

      for (int y2 = sy; y2 <= ey; y2++) {
        for (int x2 = sx; x2 <= ex; x2++) {
          long id2 = y2 * w + x2;

          if ((mask[id2] & MASK_BAD_INPUT) == 0) {
            mask[id2] |= MASK_OK_CONV;
          }
        }
      }
    }
  }
}
