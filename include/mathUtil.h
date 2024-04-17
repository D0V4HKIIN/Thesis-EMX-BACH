#pragma once

constexpr int triNum(int t) {
    return t * (t + 1) / 2;
}

constexpr int roundUpToMultiple(int num, int multiple) {
    int remainder = num % multiple;

    if (remainder == 0) {
        return num;
    }

    return num + multiple - remainder;
}

constexpr cl_int leastGreaterPow2(cl_int n) {
  if (n < 0) return 0;
  --n;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  return n+1;
}