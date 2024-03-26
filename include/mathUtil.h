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
