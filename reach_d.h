#ifndef REACH_D_H
#define REACH_D_H

#include "worklistc.h"

//#define VERIFY

unsigned bwd_reach1_d(unsigned *nodes, unsigned *edges, unsigned char *states, unsigned countOfNodes, Worklist2 &wl);

#ifdef VERIFY
unsigned fwd_reach1_d(unsigned *nodes, unsigned *edges, unsigned char *states, unsigned countOfNodes, Worklist2 &wl, unsigned *scc_root);
#else
unsigned fwd_reach1_d(unsigned *nodes, unsigned *edges, unsigned char *states, unsigned countOfNodes, Worklist2 &wl);
#endif

unsigned bwd_reach_d(unsigned *nodes, unsigned *edges, unsigned *ranges, unsigned char *states, unsigned countOfNodes, Worklist2 &wl);

#ifdef VERIFY
unsigned fwd_reach_d(unsigned *nodes, unsigned *edges, unsigned *ranges, unsigned char *states, unsigned countOfNodes, Worklist2 &wl, unsigned *scc_root);
#else
unsigned fwd_reach_d(unsigned *nodes, unsigned *edges, unsigned *ranges, unsigned char *states, unsigned countOfNodes, Worklist2 &wl);
#endif


#endif
