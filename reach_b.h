#ifndef REACH_B_H
#define REACH_B_H

//#define VERIFY

//#ifdef VERIFY
//unsigned fwd_reach1_b(unsigned *nodes, unsigned countOfNodes, unsigned *edges, unsigned countOfEdges, unsigned char *states, alg_opt opt, unsigned *scc_root);
//#else
//unsigned fwd_reach1_b(unsigned *nodes, unsigned countOfNodes, unsigned *edges, unsigned countOfEdges, unsigned char *states, alg_opt opt);
unsigned fwd_reach1_b(unsigned *nodes, unsigned *nodesT, unsigned countOfNodes, unsigned *edges, unsigned *edgesT, unsigned countOfEdges, unsigned char *states, alg_opt opt, int *h_degree, int source);
//#endif
				
unsigned bwd_reach1_b(unsigned *nodes, unsigned countOfNodes, unsigned *edges, unsigned countOfEdges, unsigned char *states, alg_opt opt);

//#ifdef VERIFY
//unsigned fwd_reach_b(unsigned *nodes, unsigned countOfNodes, unsigned *edges, unsigned countOfEdges, unsigned *ranges, unsigned char *states, alg_opt opt, unsigned *scc_root);
//#else
unsigned fwd_reach_b(unsigned *nodes, unsigned countOfNodes, unsigned *edges, unsigned countOfEdges, unsigned *ranges, unsigned char *states, alg_opt opt);
//#endif
				
unsigned bwd_reach_b(unsigned *nodes, unsigned countOfNodes, unsigned *edges, unsigned countOfEdges, unsigned *ranges, unsigned char *states, alg_opt opt);

#endif
