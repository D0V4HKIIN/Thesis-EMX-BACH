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
                      const int w, const int h, const int borderSize,
                      const double threshHigh, const double threshLow) {
  const int id = get_global_id(0);
  const int x = id % w;
  const int y = id / w;

  if (x < w && y < h) {
    ushort m = 0;

    double t = tmplImg[id];
    double s = sciImg[id];
    
    m |= select(0, MASK_BAD_INPUT | MASK_BAD_PIX_VAL, t == 0.0 || s == 0.0);
    m |= select(0, MASK_BAD_INPUT | MASK_SAT_PIXEL, t >= threshHigh || s >= threshHigh);
    m |= select(0, MASK_BAD_INPUT | MASK_LOW_PIXEL, t <= threshLow || s <= threshLow);
    m |= select(0, MASK_BAD_PIXEL_S | MASK_BAD_PIXEL_T, x < borderSize || x >= w - borderSize || y < borderSize || y >= h - borderSize);

    mask[id] = m;
  }
}

void kernel spreadMask(global ushort *mask, const int w, const int h, const int spreadWidth) {
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int id = x + w * y;

  int w2 = spreadWidth / 2;

  if (x < w && y < h && (mask[id] & MASK_BAD_INPUT) == 0) {
    int sx = max(x - w2, 0);
    int sy = max(y - w2, 0);
    int ex = min(x + w2, w - 1);
    int ey = min(y + w2, h - 1);

    bool isOk = false;

    for (int y2 = sy; y2 <= ey; y2++) {
      for (int x2 = sx; x2 <= ex; x2++) {
        int id2 = y2 * w + x2;

        if ((mask[id2] & MASK_BAD_INPUT) != 0) {
          isOk = true;
          break;
        }
      }
    }

    if (isOk) {
      mask[id] |= MASK_OK_CONV;
    }
  }
}
