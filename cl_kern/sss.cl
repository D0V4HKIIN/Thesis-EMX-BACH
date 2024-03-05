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

#define ZEROVAL ((double)1e-10)

//TODO: Fix swizzling for stampId data

void kernel createStampBounds(global long *stampsCoords, global long *stampsSizes,
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

// void kernel sampleStamp(global const double *img, global const ushort *mask,
//                         global const double *rng, 
//                         global const long *stampsCoords, global const long *stampsSizes,
//                         global double *values, volatile global int *valueCounters,
//                         const int nValues, const int paddedNValues, const long w) {

//   int sample = get_global_id(0);
//   int stampId =  get_global_id(1);

//   int randX = (int)floor(rng[2*sample + 0] * (int)stampsSizes[2*stampId + 0]);
//   int randY = (int)floor(rng[2*sample + 1] * (int)stampsSizes[2*stampId + 1]);
  
//   // Random pixel in stampId in Image coords.
//   int xI = randX + stampsCoords[2*stampId+0];
//   int yI = randY + stampsCoords[2*stampId+1];
//   int indexI = xI + yI * w;
  
//   if(mask[indexI] == 0 && fabs(img[indexI]) > ZEROVAL) {
    
    
//     long stamp_offset = stampId*paddedNValues;
//     int sample_offset = atomic_inc(&valueCounters[stampId]);
//     if (sample_offset < nValues) values[stamp_offset + sample_offset] = img[indexI];
//     else atomic_dec(&valueCounters[stampId]);
//   }
// }

void kernel sampleStamp(global const double *img, global const ushort *mask,
                        global int *stampCounter, global const double *rng, 
                        global const long *stampsCoords, global const long *stampsSizes,
                        global double *values, volatile global int *valueCounters,
                        const int stamps, const int nValues, const int paddedNValues, const long w) {
  int id = get_global_id(0);

  // Stop after randomly having selected a pixel numPix times.
  do {
    int stampId = atomic_inc(stampCounter);
    if (stampId >= stamps) return;
    int numPix = stampsSizes[2*stampId + 0] * stampsSizes[2*stampId + 1];
    int valuesCount = 0;
    for(int iter = 0; valuesCount < nValues && iter < numPix; iter++) {
      int randX = (int)floor(rng[2*iter + 0] * (int)stampsSizes[2*stampId + 0]);
      int randY = (int)floor(rng[2*iter + 1] * (int)stampsSizes[2*stampId + 1]);
      
      // Random pixel in stampId in stampId coords.
      // Random pixel in stampId in Image coords.
      int xI = randX + stampsCoords[2*stampId + 0];
      int yI = randY + stampsCoords[2*stampId + 1];
      int indexI = xI + yI * w;

      if(mask[indexI] > 0 || fabs(img[indexI]) <= ZEROVAL) {
        continue;
      }

      values[stampId*paddedNValues+valuesCount++] = img[indexI];
    }
    valueCounters[stampId] = valuesCount;
  } while (*stampCounter < stamps);
}

static void exchange(global double *i, global double *j)
{
	double k;
	k = *i;
	*i = *j;
	*j = k;
}

void kernel sortSamples(const long j, const long k,
                        global double* values, const long paddedNValues) {
  int id = get_global_id(0);                        
 
  const long i = id % paddedNValues;
  const long stampId = id / paddedNValues;
  int ixj=i^j; // Calculate indexing!
  if ((ixj)>i)
  {
    if ((i&k)==0 && values[id] > values[stampId * paddedNValues + ixj]) {
      exchange(&values[id], &values[stampId * paddedNValues + ixj]);
    }
    if ((i&k)!=0 && values[id] < values[stampId * paddedNValues + ixj]) {
      exchange(&values[id], &values[stampId * paddedNValues + ixj]);
    }
  }
}


void kernel maskSamples(global const double *img, global ushort *mask, 
                        global const long *stampsCoords, global const long *stampsSizes,
                        global double *goodPixels, global int *goodPixelCounters,
                        const long w){
  int id = get_global_id(0);
  int stampId = get_global_id(1);
  long stampPixelCount = get_global_size(0);
  
  int x = id % stampsSizes[2*stampId + 0];
  int y = id / stampsSizes[2*stampId + 0];
  
  int xI = x + stampsCoords[2*stampId + 0];
  int yI = x + stampsCoords[2*stampId + 1];
  int indexI = xI + yI*w;
  
  if (isnan(img[indexI])) {
    mask[indexI] |= (MASK_NAN_PIXEL | MASK_BAD_INPUT);
  }

  if (mask[indexI] == 0 && img[indexI] > ZEROVAL) {
    long stamp_offset = stampId*stampPixelCount;
    int sample_offset = atomic_inc(&goodPixelCounters[stampId]);
    goodPixels[stamp_offset + sample_offset] = img[indexI];
  }
}
