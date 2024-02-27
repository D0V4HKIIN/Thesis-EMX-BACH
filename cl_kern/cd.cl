void kernel createTestVec(global const double* b,
                          global double *vec,
                          const long bCount) {
    int id = get_global_id(0);
    int stampId = get_global_id(1);

    double v = 0.0;

    if (id > 0) {
        v = b[stampId * bCount + id];
    }

    vec[stampId * bCount + id] = v;
}

void kernel createTestMat(global const double *q,
                          global double *matrix,
                          const long qCount) {
    int j = get_global_id(0);
    int i = get_global_id(1);
    int stampId = get_global_id(2);

    double v = 0.0;

    if (i > 0 && j > 0) {
        v = q[stampId * qCount * qCount + max(i, j) * qCount + min(i, j)];
    }

    matrix[stampId * qCount * qCount + i * qCount + j] = v;
}
