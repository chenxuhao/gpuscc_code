#ifndef TIMING_H
#define TIMING_H

#include <sys/time.h>
#include <cuda_runtime_api.h>
#include <stdlib.h>   

#include "GPUerrchk.h"

void startTimer(timeval *timer);

int stopTimer(timeval *timer);

#define STOPWATCH(fnc, var) {timeval t; startTimer(&t); fnc; gpuErrchk( cudaDeviceSynchronize() ); var += stopTimer(&t); }

#endif
