#include <stdio.h> 

#include "timing.h"

void startTimer(timeval *timer){
        gettimeofday(timer, NULL);
}

int stopTimer(timeval *timer){
        timeval tmp;
        gettimeofday(&tmp, NULL);
        tmp.tv_sec -= timer->tv_sec;
        tmp.tv_usec -= timer->tv_usec;
        if (tmp.tv_usec < 0){
                tmp.tv_usec+=1000000;
                tmp.tv_sec--;
        }
        return int(tmp.tv_usec + tmp.tv_sec*1000000);
}
