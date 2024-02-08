void kernel sub(global const double *S, global const double *I,
                global double *D,
                const long convWidth, const long w, const long h) {
  int id = get_global_id(0);
  long x = id % w;
  long y = id / w;

  if(x >= convWidth / 2 && x < w - convWidth / 2 && y >= convWidth / 2 &&
     y < h - convWidth / 2) {
    D[id] = (I[id] - S[id]);
  } else {
    D[id] = 1e-30;
  }
}