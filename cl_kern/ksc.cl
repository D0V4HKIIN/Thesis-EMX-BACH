void kernel checkBadSubStamps(global const double *in, global const int *subStampCounts,
                              global double *out, global uchar *invalidated, global int *currentSubStamps, global int *chi2Counter, local double *localSigs,
                              const int count) {
    int gid = get_global_id(0);
    int lid = get_local_id(0);

    local int localCount;
    local int firstChi2Id;

    if (lid == 0) {
        localCount = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (gid < count) {
        int ssIndex = currentSubStamps[gid];
        int ssCount = subStampCounts[gid];
        bool invalidate = false;

        if (ssIndex < ssCount) {
            double sig = in[gid];
            invalidate = sig == -1.0;

            if (invalidate) {
                // Move to next sub-stamp
                currentSubStamps[gid]++;
            }
            else {            
                int localSigId = atomic_inc(&localCount);
                localSigs[localSigId] = sig;
            }
        }
        
        invalidated[gid] = select(0, 1, invalidate);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        firstChi2Id = atomic_add(chi2Counter, localCount);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid < localCount) {
        out[firstChi2Id + lid] = localSigs[lid];
    }
}

void kernel checkBadSubStampsFromSigmaClip(global const double *in, global const int *subStampCounts,
                                           global uchar *invalidated, global int *currentSubStamps,
                                           const double mean, const double stdDev, const double sigKernFit) {
    int id = get_global_id(0);

    int ssIndex = currentSubStamps[id];
    int ssCount = subStampCounts[id];

    bool invalidate = false;

    if (ssIndex < ssCount) {
        double chi2 = in[id];
        invalidate = (chi2 - mean) > sigKernFit * stdDev;

        if (invalidate) {
            // Move to next sub-stamp
            currentSubStamps[id]++;
        }
    }

    invalidated[id] = select(0, 1, invalidate);
}
