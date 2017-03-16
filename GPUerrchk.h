#ifndef GPUERRCHK_H
#define GPUERRCHK_H

#include <stdio.h>

inline void gpuAssert(cudaError_t code, char * file, int line, bool Abort = true)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %d %s %s %d\n", code, cudaGetErrorString(code),file,line);
        if (Abort) exit(code);
    }       
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

#endif
