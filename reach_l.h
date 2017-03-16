#ifndef REACH_L_H
#define REACH_L_H

#include "worklistc.h"

//#define VERIFY

unsigned bwd_reach1_l(unsigned *nodes, unsigned *edges, unsigned char *states, unsigned countOfNodes, Worklist2 &wl);

#ifdef VERIFY
unsigned fwd_reach1_l(unsigned *nodes, unsigned *edges, unsigned char *states, unsigned countOfNodes, Worklist2 &wl, unsigned *scc_root);
#else
unsigned fwd_reach1_l(unsigned *nodes, unsigned *edges, unsigned char *states, unsigned countOfNodes, Worklist2 &wl);
#endif

unsigned bwd_reach_l(unsigned *nodes, unsigned *edges, unsigned *ranges, unsigned char *states, unsigned countOfNodes, Worklist2 &wl);

#ifdef VERIFY
unsigned fwd_reach_l(unsigned *nodes, unsigned *edges, unsigned *ranges, unsigned char *states, unsigned countOfNodes, Worklist2 &wl, unsigned *scc_root);
#else
unsigned fwd_reach_l(unsigned *nodes, unsigned *edges, unsigned *ranges, unsigned char *states, unsigned countOfNodes, Worklist2 &wl);
#endif


#endif
