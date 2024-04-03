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

void kernel sub(global const double *S, global const double *I,
                global const ushort *mask, global double *D,
                const long convWidth, const long w, const long h,
                const double convFactor, const double finalFactor) {
  const int id = get_global_id(0);
  const long x = id % w;
  const long y = id / w;

  int halfConvWidth = convWidth / 2;
  double d = 1e-30;

  if(x >= halfConvWidth && x < w - halfConvWidth && y >= halfConvWidth && y < h - halfConvWidth) {
    if ((mask[id] & MASK_BAD_OUTPUT) == 0) {
      d = (I[id] * convFactor - S[id]) * finalFactor;
    }
  }

  D[id] = d;
}