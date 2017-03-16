#ifndef REACH_T_H
#define REACH_T_H

//#define VERIFY

#ifdef VERIFY
unsigned fwd_reach1_t(unsigned *nodes, unsigned countOfNodes, unsigned *edges, unsigned countOfEdges, unsigned char *states, alg_opt opt, unsigned *scc_root);
#else
unsigned fwd_reach1_t(unsigned *nodes, unsigned countOfNodes, unsigned *edges, unsigned countOfEdges, unsigned char *states, alg_opt opt);
#endif
				
unsigned bwd_reach1_t(unsigned *nodes, unsigned countOfNodes, unsigned *edges, unsigned countOfEdges, unsigned char *states, alg_opt opt);

#ifdef VERIFY
unsigned fwd_reach_t(unsigned *nodes, unsigned countOfNodes, unsigned *edges, unsigned countOfEdges, unsigned *ranges, unsigned char *states, alg_opt opt, unsigned *scc_root);
#else
unsigned fwd_reach_t(unsigned *nodes, unsigned countOfNodes, unsigned *edges, unsigned countOfEdges, unsigned *ranges, unsigned char *states, alg_opt opt);
#endif
				
unsigned bwd_reach_t(unsigned *nodes, unsigned countOfNodes, unsigned *edges, unsigned countOfEdges, unsigned *ranges, unsigned char *states, alg_opt opt);

#endif
