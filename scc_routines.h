#ifndef SCC_ROUTINES_H
#define SCC_ROUTINES_H

#include "worklistc.h"

//#define VERIFY

void unsetPivots(unsigned char *states, unsigned countOfNodes);

void update_ranges(unsigned *ranges, unsigned char *states, unsigned countOfNodes);

#ifdef VERIFY
void selectPivots(unsigned *ranges, unsigned char *states, unsigned countOfNodes, unsigned *pivots, Worklist2 *wlptr, unsigned *scc_root);
#else
void selectPivots(unsigned *ranges, unsigned char *states, unsigned countOfNodes, unsigned *pivots, Worklist2 *wlptr);
#endif

#ifdef VERIFY
void selectPivots(unsigned *ranges, unsigned char *states, unsigned countOfNodes, unsigned *pivots, bool *hasPivot, unsigned *scc_root);
#else
void selectPivots(unsigned *ranges, unsigned char *states, unsigned countOfNodes, unsigned *pivots, bool *hasPivot);
#endif

#ifdef VERIFY
void update(unsigned *ranges, unsigned char *states, unsigned countOfNodes, unsigned *pivots, Worklist2 *wlptr, unsigned *scc_root);
#else
void update(unsigned *ranges, unsigned char *states, unsigned countOfNodes, unsigned *pivots, Worklist2 *wlptr);
#endif

#ifdef VERIFY
void update(unsigned *ranges, unsigned char *states, unsigned countOfNodes, unsigned *pivots, bool *hasPivot, unsigned *scc_root);
#else
void update(unsigned *ranges, unsigned char *states, unsigned countOfNodes, unsigned *pivots, bool *hasPivot);
#endif

unsigned trim_first(unsigned *d_nodes, unsigned *d_nodesT, unsigned countOfNodes, unsigned *d_edges, unsigned *d_edgesT, unsigned countOfEdges, unsigned char *d_states);

#ifdef VERIFY
unsigned trim(unsigned *d_nodes, unsigned *d_nodesT, unsigned countOfNodes, unsigned *d_edges, unsigned *d_edgesT, unsigned countOfEdges, unsigned *d_ranges, unsigned char *d_states, unsigned *scc_root);
#else
unsigned trim(unsigned *d_nodes, unsigned *d_nodesT, unsigned countOfNodes, unsigned *d_edges, unsigned *d_edgesT, unsigned countOfEdges, unsigned *d_ranges, unsigned char *d_states);
#endif

#ifdef VERIFY
unsigned trim1(unsigned *d_nodes, unsigned *d_nodesT, unsigned countOfNodes, unsigned *d_edges, unsigned *d_edgesT, unsigned countOfEdges, unsigned char *d_states, unsigned *scc_root);
#else
unsigned trim1(unsigned *d_nodes, unsigned *d_nodesT, unsigned countOfNodes, unsigned *d_edges, unsigned *d_edgesT, unsigned countOfEdges, unsigned char *d_states);
#endif

unsigned trim2(unsigned *d_nodes, unsigned *d_nodesT, unsigned countOfNodes, unsigned *d_edges, unsigned *d_edgesT, unsigned countOfEdges, unsigned *d_ranges, unsigned char *d_states);

#endif
