#ifndef WCC_H
#define WCC_H

#include "scc.h"
#include "worklistc.h"

#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/count.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

#define BLOCKSIZE	256

//#define VERIFY

__global__ void update_ranges1(unsigned *ranges, unsigned char *states, unsigned countOfNodes);

#ifdef VERIFY
unsigned update_wcc(unsigned *d_nodes, unsigned *d_edges, unsigned *d_ranges, unsigned char *d_states, unsigned countOfNodes, bool *hasPivot, unsigned *d_minRange, unsigned *scc_root);
#else
unsigned update_wcc(unsigned *d_nodes, unsigned *d_edges, unsigned *d_ranges, unsigned char *d_states, unsigned countOfNodes, bool *hasPivot, unsigned *d_minRange);
#endif

#ifdef VERIFY
unsigned update_wcc(unsigned *d_nodes, unsigned *d_edges, unsigned *d_ranges, unsigned char *d_states, unsigned countOfNodes, Worklist2 *wlptr, unsigned *d_minRange, unsigned *scc_root);
#else
unsigned update_wcc(unsigned *d_nodes, unsigned *d_edges, unsigned *d_ranges, unsigned char *d_states, unsigned countOfNodes, Worklist2 *wlptr, unsigned *d_minRange);
#endif

#endif
