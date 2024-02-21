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

//TODO: Fix swizzling for stamp data

void kernel createStamps(global long *stampsCoords, global long *stampsSizes,
                          const int nStampsX, const int nStampsY, const long fullStampWidth, 
                          const long w, const long h) {
  const int id = get_global_id(0);
  const int stampx = id % nStampsX;
  const int stampy = id / nStampsX;


  long startx = stampx * w / nStampsX;
  long starty = stampy * h / nStampsY;
  
  long stopx = min(startx + fullStampWidth, w);
  long stopy = min(starty + fullStampWidth, h);

  long stampw = stopx - startx;
  long stamph = stopy - starty;

  stampsCoords[2*id + 0] = startx;
  stampsCoords[2*id + 1] = starty;
  stampsSizes[2*id + 0] = stampw;
  stampsSizes[2*id + 1] = stamph;
}