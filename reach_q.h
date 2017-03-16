#ifndef REACH_Q_H
#define REACH_Q_H

//#define VERIFY

/* firstly, all nodes are of the same range, 1 */
#ifdef VERIFY
unsigned fwd_reach1_q(unsigned *nodes, unsigned countOfNodes, unsigned *edges, unsigned countOfEdges, unsigned char *states, alg_opt opt, unsigned *scc_root);
#else
unsigned fwd_reach1_q(unsigned *nodes, unsigned countOfNodes, unsigned *edges, unsigned countOfEdges, unsigned char *states, alg_opt opt);
#endif
				
unsigned bwd_reach1_q(unsigned *nodes, unsigned countOfNodes, unsigned *edges, unsigned countOfEdges, unsigned char *states, alg_opt opt);

#ifdef VERIFY
unsigned fwd_reach_q(unsigned *nodes, unsigned countOfNodes, unsigned *edges, unsigned countOfEdges, unsigned *ranges, unsigned char *states, alg_opt opt, unsigned *scc_root);
#else
unsigned fwd_reach_q(unsigned *nodes, unsigned countOfNodes, unsigned *edges, unsigned countOfEdges, unsigned *ranges, unsigned char *states, alg_opt opt);
#endif
				
unsigned bwd_reach_q(unsigned *nodes, unsigned countOfNodes, unsigned *edges, unsigned countOfEdges, unsigned *ranges, unsigned char *states, alg_opt opt);

#endif
