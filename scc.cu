#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <utility>
#include <cuda_runtime_api.h>

//#define VERIFY

#include "load.h"
#include "tarjan.h"
#include "scc.h"
#include "timing.h"
#include "GPUerrchk.h"
#include "reach_q.h"
#include "reach_t.h"
#include "reach_b.h"

#include "reach_d.h"
#include "reach_l.h"
#include "worklistc.h"

#include "wcc.h"
#include "scc_routines.h"

#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/count.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

#include <set>
#include <vector>

#define TEST
//#define WCC
#define TRIM_FIRST
//#define TRIM_2


__global__ void countElimmed(unsigned char *states, unsigned countOfNodes, unsigned *nElimmed) {

	int id = blockDim.x * blockIdx.x + threadIdx.x + 1;
	if (id <= countOfNodes) {
		//if (isElim(&states[id])) {
		if ((states[id] & 12) == 4) {
			atomicAdd(nElimmed, 1);
		}
	}
}

using namespace std;

static pair <unsigned, float> computeSCC(unsigned countOfNodes, unsigned countOfEdges, unsigned *nodes, unsigned *edges, unsigned *nodesT, unsigned *edgesT, alg_opt opt);

int main(int argc, char **argv) {
	
	unsigned countOfNodes;
	unsigned countOfEdges;
	
	unsigned *nodes = NULL;
	unsigned *edges = NULL;
	
	unsigned *nodesT = NULL;
	unsigned *edgesT = NULL; // this represents transposed graph

	unsigned *outDegree = NULL;
	unsigned *inDegree = NULL;
	
	bool header = false;
	bool tarjan = false;
	
	pair <unsigned, float> result;
	
	int c;
	char *file = NULL;
	alg_opt opt;
	
	opt.alg = TOPO;
	opt.warp_size = 32;
	opt.block_size = 256; 
	opt.trim_strat = 0;
	opt.trim_switch_limit = 20;
	opt.trim_both = true;
	opt.lin_cull_strat = 1;	//default values
	
	while((c = getopt(argc, argv, "htf:a:w:b:")) != -1) {
		switch(c) 
		{
			case 'h': header = true; break;
			case 't': tarjan = true; break;
			case 'f': file = optarg; break;
			case 'a': switch(optarg[0])
					  {
						case 'q': opt.alg = QUADRATIC; break;
						case 't': opt.alg = TOPO; break;
						case 'd': opt.alg = DATA; break;
						case 'l': opt.alg = LINEAR; break;
						case 'h': opt.alg = HYBRID; break;
						case 'c': opt.alg = COMBINE; break;
						case 'b': opt.alg = BEST; break;
						case 'a': opt.alg = ALL; break;
					  }
					  break;	
			case 'w': opt.warp_size = strtoul(optarg, NULL, 0); break;
			case 'b': opt.block_size = strtoul(optarg, NULL, 0); break;
		}
	}
	
	printf("loading graph...\n");
	//if( !load(file, header, &countOfNodes, &countOfEdges, &nodes, &edges, &nodesT, &edgesT) ) 
	if( !load(file, header, &countOfNodes, &countOfEdges, &nodes, &edges, &nodesT, &edgesT, outDegree, inDegree) ) 
		return 1;
	
	printf("    ->nodes: %u\n", countOfNodes);
	printf("    ->edges: %u\n", countOfEdges);	
	
	if(tarjan) {
		printf("tarjan decomposition...\n");
	
		result = TR_Decomposition(countOfEdges, countOfNodes, edges, nodes);
	
		printf("    ->count of SCC: %d\n", result.first);
		printf("    ->time: %f\n", result.second);
	}
	
	if (opt.alg == ALL) {
		//for (int i = QUADRATIC; i <= HYBRID; i++) { 
		for (int i = QUADRATIC; i <= BEST; i++) { 
			opt.alg = (reach_strategy)i;	
			printf("computing SCC in parallel - alg: %d...\n", opt.alg);
	
			result = computeSCC(countOfNodes, countOfEdges, nodes, edges, nodesT, edgesT, opt);
			
			printf("    ->count of SCC: %d\n", result.first);
			printf("    ->time: %f\n", result.second);
		}
	}	
	else {	
		printf("computing SCC in parallel - alg: %d...\n", opt.alg);
	
		result = computeSCC(countOfNodes, countOfEdges, nodes, edges, nodesT, edgesT, opt);
			
		printf("    ->count of SCC: %d\n", result.first);
		printf("    ->time: %f\n", result.second);
	}
}

pair <unsigned, float> computeSCC(unsigned countOfNodes, unsigned countOfEdges, unsigned *nodes, unsigned *edges, unsigned *nodesT, unsigned *edgesT, alg_opt opt) {
	
	unsigned numberOfSCC = 0;
	unsigned numberOfTrivSCC = 0;
	unsigned numberOfNontrivSCC = 0;
	
	unsigned *d_nodes = NULL;
	unsigned *d_nodesT = NULL;
	unsigned *d_edges = NULL;
	unsigned *d_edgesT = NULL;
	unsigned *d_pivots = NULL;
	unsigned *d_ranges = NULL;
	unsigned char *d_states = NULL; 

	Worklist2 wl(countOfNodes);

	/* states bits of nodes TQ,FE,BE,P,T,E,F,B: TQ - has already been in trimming queue, FE - fwd expanded, BE - bwd expanded, P - pivot, T - trimmed, E - eliminated, F - fwd visited, B - bwd visited */
	timeval timer;
	
	#ifdef TEST
	unsigned time_fwd = 0;
	unsigned time_bwd = 0;	
	unsigned time_trim = 0;
	unsigned time_update = 0;
	
	unsigned fwdDepths = 0;
	unsigned bwdDepths = 0;
	unsigned trimIterations = 0;
	#endif
	
	#ifdef WCC
	unsigned t_fwd;
	unsigned t_bwd;
	bool wcc_done = false;
	unsigned min_range;
	unsigned *d_minRange;
	#endif

	#ifdef VERIFY
	unsigned *d_sccroot = NULL;
	gpuErrchk( cudaMalloc((void **)&d_sccroot, (1 + countOfNodes) * sizeof(*d_sccroot)) );
	//gpuErrchk( cudaMemset(d_sccroot, 0, (1 + countOfNodes) * sizeof(*d_sccroot)) );
	#endif
	
	unsigned n_elimmed = 0;
	unsigned *d_nElimmed;
	gpuErrchk( cudaMalloc((void **)&d_nElimmed, sizeof(*d_nElimmed)) );
	dim3 dimGrid;
	
	gpuErrchk( cudaSetDevice(0) );
	gpuErrchk( cudaMalloc((void**)&d_nodes, (countOfNodes + 2)*sizeof(*d_nodes)) );
	gpuErrchk( cudaMalloc((void**)&d_edges, (countOfEdges + 1)*sizeof(*d_edges)) );
	gpuErrchk( cudaMalloc((void**)&d_nodesT, (countOfNodes + 2)*sizeof(*d_nodesT)) );
	gpuErrchk( cudaMalloc((void**)&d_edgesT, (countOfEdges + 1)*sizeof(*d_edgesT)) );
	gpuErrchk( cudaMalloc((void**)&d_pivots, (PIVOT_HASH_CONST + 1)*sizeof(*d_pivots)) );
	gpuErrchk( cudaMalloc((void**)&d_ranges, (countOfNodes + 1)*sizeof(*d_ranges)) );
	gpuErrchk( cudaMalloc((void**)&d_states, (countOfNodes + 1)*sizeof(*d_states)) );
	
	gpuErrchk( cudaMemcpy(d_nodes, nodes, (countOfNodes + 2)*sizeof(*d_nodes), cudaMemcpyHostToDevice) );		
	gpuErrchk( cudaMemcpy(d_edges, edges, (countOfEdges + 1)*sizeof(*d_edges), cudaMemcpyHostToDevice) );	
	gpuErrchk( cudaMemcpy(d_nodesT, nodesT, (countOfNodes + 2)*sizeof(*d_nodesT), cudaMemcpyHostToDevice) );	
	gpuErrchk( cudaMemcpy(d_edgesT, edgesT, (countOfEdges + 1)*sizeof(*d_edgesT), cudaMemcpyHostToDevice) );	
	
	gpuErrchk( cudaMemset(d_pivots, 0, (PIVOT_HASH_CONST + 1)*sizeof(*d_pivots)) );
	thrust::fill(thrust::device, d_ranges, d_ranges + 1 + countOfNodes, 1);
	gpuErrchk( cudaMemset(d_states, 0, (countOfNodes + 1)*sizeof(*d_states)) );

	startTimer(&timer);	
	unsigned char *states = NULL;
	states = (unsigned char*)malloc((countOfNodes + 1) * sizeof(*states));
	unsigned firstPivot = 0;
	int *degree = (int *)malloc((countOfNodes + 1) * sizeof(int));
	int *degreeT = (int *)malloc((countOfNodes + 1) * sizeof(int));
	degree[0] = 0;
	degreeT[0] = 0;
	for (int i = 1; i < (countOfNodes + 1); i++) degree[i] = nodes[i + 1] - nodes[i]; 
	for (int i = 1; i < (countOfNodes + 1); i++) degreeT[i] = nodesT[i + 1] - nodesT[i]; 
		
#ifdef TRIM_FIRST
	#ifdef VERIFY
	STOPWATCH( trim1(d_nodes, d_nodesT, countOfNodes, d_edges, d_edgesT, countOfEdges, d_states, d_sccroot), time_trim );
	#else
	STOPWATCH( trim1(d_nodes, d_nodesT, countOfNodes, d_edges, d_edgesT, countOfEdges, d_states), time_trim );
	#endif
	gpuErrchk( cudaMemcpy(states, d_states, (countOfNodes + 1) * sizeof(*states), cudaMemcpyDeviceToHost) );
	for (int i = 1; i <= countOfNodes; i++) {
		if (!isElim(&states[i])) {
			printf("%d not elimmed, set as first pivot\n", i);
			gpuErrchk( cudaMemset(&d_states[i], 19, 1) );
			firstPivot = i;
		#ifdef VERIFY
			gpuErrchk( cudaMemcpy(&d_sccroot[i], &firstPivot, sizeof(unsigned), cudaMemcpyHostToDevice) );
		#endif
			break;
		}
	}
	
#else
	gpuErrchk( cudaMemset(&d_states[1], 19, 1) ); // first seed = node 1 - already marked as visited(forward & backward) and is pivot = 00010011
	firstPivot = 1;
	#ifdef VERIFY
		gpuErrchk( cudaMemcpy(&d_sccroot[1], &firstPivot, sizeof(unsigned), cudaMemcpyHostToDevice) );
	#endif
#endif
	
	unsigned step = 0;
	bool hasPivot = true;
	bool FALSE = false;
	bool *d_hasPivot;
	gpuErrchk( cudaMalloc((void **)&d_hasPivot, sizeof(*d_hasPivot)) );

#ifdef TRIM_FIRST
	if (firstPivot == 0)
		hasPivot = false;
#endif

	while (hasPivot) {

		if (opt.alg == DATA || opt.alg == LINEAR) {
			hasPivot = false;
		}
		else if (opt.alg == COMBINE) {
			if (n_elimmed < (countOfNodes / 100))
				hasPivot = false;
			else
				gpuErrchk( cudaMemcpy(d_hasPivot, &FALSE, sizeof(*d_hasPivot), cudaMemcpyHostToDevice) );
		}
		else {
			gpuErrchk( cudaMemcpy(d_hasPivot, &FALSE, sizeof(*d_hasPivot), cudaMemcpyHostToDevice) );
		}
		step++;

		unsigned temp;

		/*=== fwd & bwd reach */
		switch(opt.alg) {
			case QUADRATIC:
			printf("running QUADRATIC strategy\n");
			#ifdef WCC
				if (!wcc_done) 
					t_fwd = time_fwd;
			#else
				unsigned t_f_first;
				if (step == 1) {
					t_f_first = time_fwd;
				}
			#endif
			#ifdef VERIFY
				if (step == 1) {
					STOPWATCH( temp = fwd_reach1_q(d_nodes, countOfNodes, d_edges, countOfEdges, d_states, opt, d_sccroot), time_fwd );
				}
				else {
					STOPWATCH( temp = fwd_reach_q(d_nodes, countOfNodes, d_edges, countOfEdges, d_ranges, d_states, opt, d_sccroot), time_fwd );
				}
			#else
				if (step == 1) {
					STOPWATCH( temp = fwd_reach1_q(d_nodes, countOfNodes, d_edges, countOfEdges, d_states, opt), time_fwd );
				}
				else {
					STOPWATCH( temp = fwd_reach_q(d_nodes, countOfNodes, d_edges, countOfEdges, d_ranges, d_states, opt), time_fwd );
				}
			#endif
			#ifdef WCC
				if (!wcc_done) { 
					printf("fwd time of step %d is %f ms\n", step, (time_fwd - t_fwd) / 1000.0);
				}
			#else
				if (step == 1) {
					printf("t_f_first is %f ms\n", (time_fwd - t_f_first) / 1000.0);	
				}
			#endif
				fwdDepths += temp;
			
			#ifdef WCC
				if (!wcc_done) 
					t_bwd = time_bwd;
			#else 
				unsigned t_b_first;
				if (step == 1) {
					t_b_first = time_bwd;
				}
			#endif
				if (step == 1) {	
					STOPWATCH( temp = bwd_reach1_q(d_nodesT, countOfNodes, d_edgesT, countOfEdges, d_states, opt), time_bwd );
				}
				else {
					STOPWATCH( temp = bwd_reach_q(d_nodesT, countOfNodes, d_edgesT, countOfEdges, d_ranges, d_states, opt), time_bwd );
				}
			#ifdef WCC
				if (!wcc_done) { 
					printf("bwd time of step %d is %f ms\n", step, (time_bwd - t_bwd) / 1000.0);
				}
			#else
				if (step == 1) {
					printf("t_b_first is %f ms\n", (time_bwd - t_b_first) / 1000.0);	
				}
			#endif
				bwdDepths += temp;
				break;
			
			case TOPO:
			printf("running TOPO strategy\n");
			#ifdef WCC
				if (!wcc_done) 
					t_fwd = time_fwd;
			#endif
			#ifdef VERIFY
				if (step == 1) {
					STOPWATCH( temp = fwd_reach1_t(d_nodes, countOfNodes, d_edges, countOfEdges, d_states, opt, d_sccroot), time_fwd );
				}
				else {
					STOPWATCH( temp = fwd_reach_t(d_nodes, countOfNodes, d_edges, countOfEdges, d_ranges, d_states, opt, d_sccroot), time_fwd );
				}
			#else
				if (step == 1) {
					STOPWATCH( temp = fwd_reach1_t(d_nodes, countOfNodes, d_edges, countOfEdges, d_states, opt), time_fwd );
				}
				else {
					STOPWATCH( temp = fwd_reach_t(d_nodes, countOfNodes, d_edges, countOfEdges, d_ranges, d_states, opt), time_fwd );
				}
			#endif
			#ifdef WCC
				if (!wcc_done) { 
					printf("fwd time of step %d is %f ms\n", step, (time_fwd - t_fwd) / 1000.0);
				}
			#endif
				fwdDepths += temp;
			
			#ifdef WCC
				if (!wcc_done) 
					t_bwd = time_bwd;
			#endif
				if (step == 1) {	
					STOPWATCH( temp = bwd_reach1_t(d_nodesT, countOfNodes, d_edgesT, countOfEdges, d_states, opt), time_bwd );
				}
				else {
					STOPWATCH( temp = bwd_reach_t(d_nodesT, countOfNodes, d_edgesT, countOfEdges, d_ranges, d_states, opt), time_bwd );
				}
			#ifdef WCC
				if (!wcc_done) { 
					printf("bwd time of step %d is %f ms\n", step, (time_bwd - t_bwd) / 1000.0);
				}
			#endif
				bwdDepths += temp;
				break;

			case DATA:
			if (step == 1) {
				wl.pushHost(firstPivot);
			}
			#ifdef WCC
				if (!wcc_done) 
					t_fwd = time_fwd;
			#endif
			#ifdef VERIFY
				if (step == 1) {
					STOPWATCH( temp = fwd_reach1_d(d_nodes, d_edges, d_states, countOfNodes, wl, d_sccroot), time_fwd );
				}
				else {
					STOPWATCH( temp = fwd_reach_d(d_nodes, d_edges, d_ranges, d_states, countOfNodes, wl, d_sccroot), time_fwd );
				}
			#else
				if (step == 1) {
					STOPWATCH( temp = fwd_reach1_d(d_nodes, d_edges, d_states, countOfNodes, wl), time_fwd );
				}
				else {
					STOPWATCH( temp = fwd_reach_d(d_nodes, d_edges, d_ranges, d_states, countOfNodes, wl), time_fwd );
				}
			#endif
			#ifdef WCC
				if (!wcc_done) { 
					printf("fwd time of step %d is %f ms\n", step, (time_fwd - t_fwd) / 1000.0);
				}
			#endif
				fwdDepths += temp;
			
			#ifdef WCC
				if (!wcc_done) 
					t_bwd = time_bwd;
			#endif
				if (step == 1) {	
					STOPWATCH( temp = bwd_reach1_d(d_nodesT, d_edgesT, d_states, countOfNodes, wl), time_bwd );	
				}
				else {
					STOPWATCH( temp = bwd_reach_d(d_nodesT, d_edgesT, d_ranges, d_states, countOfNodes, wl), time_bwd );
				}
			#ifdef WCC
				if (!wcc_done) { 
					printf("bwd time of step %d is %f ms\n", step, (time_bwd - t_bwd) / 1000.0);
				}
			#endif
				bwdDepths += temp;
				break;

			case LINEAR:
			printf("running LINEAR strategy\n");
			if (step == 1) {
				wl.pushHost(firstPivot);
			}
			#ifdef WCC
				if (!wcc_done) 
					t_fwd = time_fwd;
			#endif
			#ifdef VERIFY
				if (step == 1) {
					STOPWATCH( temp = fwd_reach1_l(d_nodes, d_edges, d_states, countOfNodes, wl, d_sccroot), time_fwd );
				}
				else {
					STOPWATCH( temp = fwd_reach_l(d_nodes, d_edges, d_ranges, d_states, countOfNodes, wl, d_sccroot), time_fwd );
				}
			#else
				if (step == 1) {	
					STOPWATCH( temp = fwd_reach1_l(d_nodes, d_edges, d_states, countOfNodes, wl), time_fwd );
				}
				else {
					STOPWATCH( temp = fwd_reach_l(d_nodes, d_edges, d_ranges, d_states, countOfNodes, wl), time_fwd );	
				}
			#endif
			#ifdef WCC
				if (!wcc_done) { 
					printf("fwd time of step %d is %f ms\n", step, (time_fwd - t_fwd) / 1000.0);
				}
			#endif
				fwdDepths += temp;
			
			#ifdef WCC
				if (!wcc_done) 
					t_bwd = time_bwd;
			#endif
				if (step == 1) {	
					STOPWATCH( temp = bwd_reach1_l(d_nodesT, d_edgesT, d_states, countOfNodes, wl), time_bwd );	
				}
				else {
					STOPWATCH( temp = bwd_reach_l(d_nodesT, d_edgesT, d_ranges, d_states, countOfNodes, wl), time_bwd );	
				}
			#ifdef WCC
				if (!wcc_done) { 
					printf("bwd time of step %d is %f ms\n", step, (time_bwd - t_bwd) / 1000.0);
				}
			#endif
				bwdDepths += temp;
				break;

			case HYBRID:
			printf("running HYBRID strategy\n");
				#if 0
				if(step != 1 && n_elimmed < (countOfNodes / 100)) {
					setGridDimension(&dimGrid, countOfNodes, 1024);
					gpuErrchk( cudaMemset(d_nElimmed, 0, sizeof(*d_nElimmed)) );
					countElimmed<<<dimGrid, 1024>>>(d_states, countOfNodes, d_nElimmed);
					gpuErrchk( cudaMemcpy(&n_elimmed, d_nElimmed, sizeof(n_elimmed), cudaMemcpyDeviceToHost) );
					if (n_elimmed >= (countOfNodes / 100)) {
						printf("%d of %d elimmed, transfer in step %d\n", n_elimmed, countOfNodes, step);
					}
				}
				#endif
								
				if (n_elimmed < (countOfNodes / 100)) {
					printf("reach_t\n");
			#ifdef WCC
				if (!wcc_done) 
					t_fwd = time_fwd;
			#endif
				#ifdef VERIFY
					if (step == 1) {
						STOPWATCH( temp = fwd_reach1_t(d_nodes, countOfNodes, d_edges, countOfEdges, d_states, opt, d_sccroot), time_fwd );
					}
					else {
						STOPWATCH( temp = fwd_reach_t(d_nodes, countOfNodes, d_edges, countOfEdges, d_ranges, d_states, opt, d_sccroot), time_fwd );
					}
				#else
					if (step == 1) {
						STOPWATCH( temp = fwd_reach1_t(d_nodes, countOfNodes, d_edges, countOfEdges, d_states, opt), time_fwd );
					}
					else {
						STOPWATCH( temp = fwd_reach_t(d_nodes, countOfNodes, d_edges, countOfEdges, d_ranges, d_states, opt), time_fwd );
					}
				#endif
			#ifdef WCC
				if (!wcc_done) { 
					printf("fwd time of step %d is %f ms\n", step, (time_fwd - t_fwd) / 1000.0);
				}
			#endif
					fwdDepths += temp;		
					
			#ifdef WCC
				if (!wcc_done) 
					t_bwd = time_bwd;
			#endif
					if (step == 1) {
						STOPWATCH( temp = bwd_reach1_t(d_nodesT, countOfNodes, d_edgesT, countOfEdges, d_states, opt), time_bwd );
					}
					else {
						STOPWATCH( temp = bwd_reach_t(d_nodesT, countOfNodes, d_edgesT, countOfEdges, d_ranges, d_states, opt), time_bwd );
					}
			#ifdef WCC
				if (!wcc_done) { 
					printf("bwd time of step %d is %f ms\n", step, (time_bwd - t_bwd) / 1000.0);
				}
			#endif
					bwdDepths += temp;	
				}
				else {
				#ifdef VERIFY
					STOPWATCH( temp = fwd_reach_q(d_nodes, countOfNodes, d_edges, countOfEdges, d_ranges, d_states, opt, d_sccroot), time_fwd );	
				#else
					STOPWATCH( temp = fwd_reach_q(d_nodes, countOfNodes, d_edges, countOfEdges, d_ranges, d_states, opt), time_fwd );	
				#endif
					fwdDepths += temp;
					
					STOPWATCH( temp = bwd_reach_q(d_nodesT, countOfNodes, d_edgesT, countOfEdges, d_ranges, d_states, opt), time_bwd );
					bwdDepths += temp;					
				}
				break;						

			case COMBINE:
			printf("running COMBINE strategy\n");
			if (step == 1) {
				wl.pushHost(firstPivot);
			}
				#if 0
				if(step != 1 && n_elimmed < (countOfNodes / 100)) {
					setGridDimension(&dimGrid, countOfNodes, 1024);
					gpuErrchk( cudaMemset(d_nElimmed, 0, sizeof(*d_nElimmed)) );
					countElimmed<<<dimGrid, 1024>>>(d_states, countOfNodes, d_nElimmed);
					gpuErrchk( cudaMemcpy(&n_elimmed, d_nElimmed, sizeof(n_elimmed), cudaMemcpyDeviceToHost) );
					if (n_elimmed >= (countOfNodes / 100)) {
						printf("%d of %d elimmed, transfer in step %d\n", n_elimmed, countOfNodes, step);
					}
				}
				#endif
								
				if (n_elimmed < (countOfNodes / 100)) {
					printf("reach_l\n");
			#ifdef WCC
				if (!wcc_done) 
					t_fwd = time_fwd;
			#endif
				#ifdef VERIFY
					if (step == 1) {
						STOPWATCH( temp = fwd_reach1_l(d_nodes, d_edges, d_states, countOfNodes, wl, d_sccroot), time_fwd );
					}
					else {
						STOPWATCH( temp = fwd_reach_l(d_nodes, d_edges, d_ranges, d_states, countOfNodes, wl, d_sccroot), time_fwd );
					}
				#else
					if (step == 1) {
						STOPWATCH( temp = fwd_reach1_l(d_nodes, d_edges, d_states, countOfNodes, wl), time_fwd );
					}
					else {
						STOPWATCH( temp = fwd_reach_l(d_nodes, d_edges, d_ranges, d_states, countOfNodes, wl), time_fwd );
					}
				#endif
			#ifdef WCC
				if (!wcc_done) { 
					printf("fwd time of step %d is %f ms\n", step, (time_fwd - t_fwd) / 1000.0);
				}
			#endif
					fwdDepths += temp;		
					
			#ifdef WCC
				if (!wcc_done) 
					t_bwd = time_bwd;
			#endif
					if (step == 1) {
						STOPWATCH( temp = bwd_reach1_l(d_nodesT, d_edgesT, d_states, countOfNodes, wl), time_bwd );
					}
					else {
						STOPWATCH( temp = bwd_reach_l(d_nodesT, d_edgesT, d_ranges, d_states, countOfNodes, wl), time_bwd );
					}
			#ifdef WCC
				if (!wcc_done) { 
					printf("bwd time of step %d is %f ms\n", step, (time_bwd - t_bwd) / 1000.0);
				}
			#endif
					bwdDepths += temp;	
				}
				else {
				#ifdef VERIFY
					STOPWATCH( temp = fwd_reach_q(d_nodes, countOfNodes, d_edges, countOfEdges, d_ranges, d_states, opt, d_sccroot), time_fwd );	
				#else
					STOPWATCH( temp = fwd_reach_q(d_nodes, countOfNodes, d_edges, countOfEdges, d_ranges, d_states, opt), time_fwd );	
				#endif
					fwdDepths += temp;
					
					STOPWATCH( temp = bwd_reach_q(d_nodesT, countOfNodes, d_edgesT, countOfEdges, d_ranges, d_states, opt), time_bwd );
					bwdDepths += temp;					
				}
				break;						


			case BEST:
				printf("running BEST strategy\n");
				if (n_elimmed < (countOfNodes / 100)) {
					printf("best reach_t\n");
			#ifdef WCC
				if (!wcc_done) 
					t_fwd = time_fwd;
			#endif
				#ifdef VERIFY
				/*
					if (step == 1) {
						STOPWATCH( temp = fwd_reach1_b(d_nodes, countOfNodes, d_edges, countOfEdges, d_states, opt, d_sccroot), time_fwd );
					}
					else {
						STOPWATCH( temp = fwd_reach_b(d_nodes, countOfNodes, d_edges, countOfEdges, d_ranges, d_states, opt, d_sccroot), time_fwd );
					}
				*/
				#else
					if (step == 1) {
						STOPWATCH( temp = fwd_reach1_b(d_nodes, d_nodesT, countOfNodes, d_edges, d_edgesT, countOfEdges, d_states, opt, degree, firstPivot), time_fwd );
						//STOPWATCH( temp = fwd_reach1_b(d_nodes, countOfNodes, d_edges, countOfEdges, d_states, opt), time_fwd );
					}
					else {
						STOPWATCH( temp = fwd_reach_b(d_nodes, countOfNodes, d_edges, countOfEdges, d_ranges, d_states, opt), time_fwd );
					}
				#endif
			#ifdef WCC
				if (!wcc_done) { 
					printf("best fwd time of step %d is %f ms\n", step, (time_fwd - t_fwd) / 1000.0);
				}
			#endif
					fwdDepths += temp;		
					
			#ifdef WCC
				if (!wcc_done) 
					t_bwd = time_bwd;
			#endif
					if (step == 1) {
						STOPWATCH( temp = bwd_reach1_b(d_nodes, d_nodesT, countOfNodes, d_edges, d_edgesT, countOfEdges, d_states, opt, degreeT, firstPivot), time_bwd );
						//STOPWATCH( temp = bwd_reach1_b(d_nodesT, countOfNodes, d_edgesT, countOfEdges, d_states, opt), time_bwd );
					}
					else {
						STOPWATCH( temp = bwd_reach_b(d_nodesT, countOfNodes, d_edgesT, countOfEdges, d_ranges, d_states, opt), time_bwd );
					}
			#ifdef WCC
				if (!wcc_done) { 
					printf("best bwd time of step %d is %f ms\n", step, (time_bwd - t_bwd) / 1000.0);
				}
			#endif
					bwdDepths += temp;	
				}
				else {
				#ifdef VERIFY
					STOPWATCH( temp = fwd_reach_q(d_nodes, countOfNodes, d_edges, countOfEdges, d_ranges, d_states, opt, d_sccroot), time_fwd );	
				#else
					STOPWATCH( temp = fwd_reach_q(d_nodes, countOfNodes, d_edges, countOfEdges, d_ranges, d_states, opt), time_fwd );	
				#endif
					fwdDepths += temp;
					
					STOPWATCH( temp = bwd_reach_q(d_nodesT, countOfNodes, d_edgesT, countOfEdges, d_ranges, d_states, opt), time_bwd );
					bwdDepths += temp;					
				}
				break;
		}
		/*=== fwd & bwd reach */

		/*=== Trim ===*/
	//#ifndef TRIM_FIST
		#ifdef VERIFY
		if (step == 1) {
			STOPWATCH( temp = trim1(d_nodes, d_nodesT, countOfNodes, d_edges, d_edgesT, countOfEdges, d_states, d_sccroot), time_trim );
		}
		else {
			STOPWATCH( temp = trim(d_nodes, d_nodesT, countOfNodes, d_edges, d_edgesT, countOfEdges, d_ranges, d_states, d_sccroot), time_trim );
		}
		#else
		if (step == 1) {
			STOPWATCH( temp = trim1(d_nodes, d_nodesT, countOfNodes, d_edges, d_edgesT, countOfEdges, d_states), time_trim );
		}
		else {
			STOPWATCH( temp = trim(d_nodes, d_nodesT, countOfNodes, d_edges, d_edgesT, countOfEdges, d_ranges, d_states), time_trim );
		}
		#endif
	/*
	#else
		#ifdef VERIFY
			STOPWATCH( temp = trim(d_nodes, d_nodesT, countOfNodes, d_edges, d_edgesT, countOfEdges, d_ranges, d_states, d_sccroot), time_trim );
		#else
			STOPWATCH( temp = trim(d_nodes, d_nodesT, countOfNodes, d_edges, d_edgesT, countOfEdges, d_ranges, d_states), time_trim );
		#endif
	#endif
	*/
		trimIterations += temp;
		/*=== Trim ===*/

		/*=== update ===*/
		if (opt.alg == DATA || opt.alg == LINEAR) {
			wl.clearHost();
		}
		else if (opt.alg == COMBINE) {
			if (n_elimmed < (countOfNodes / 100))
				wl.clearHost();
		}

		#ifdef WCC
		if (!wcc_done) {

				gpuErrchk( cudaMemset(d_pivots, 0, (PIVOT_HASH_CONST + 1) * sizeof(*d_pivots)) );
			#ifdef VERIFY
				if (opt.alg == DATA || opt.alg == LINEAR) {
					STOPWATCH( update(d_ranges, d_states, countOfNodes, d_pivots, &wl, d_sccroot), time_update );
				}
				else if (opt.alg == COMBINE) {
					if (n_elimmed < (countOfNodes / 100)) {
						STOPWATCH( update(d_ranges, d_states, countOfNodes, d_pivots, &wl, d_sccroot), time_update );
					}
					else {
						STOPWATCH( update(d_ranges, d_states, countOfNodes, d_pivots, d_hasPivot, d_sccroot), time_update );
					}
				}
				else {
					STOPWATCH( update(d_ranges, d_states, countOfNodes, d_pivots, d_hasPivot, d_sccroot), time_update );
				}
			#else
				if (opt.alg == DATA || opt.alg == LINEAR) {
					STOPWATCH( update(d_ranges, d_states, countOfNodes, d_pivots, &wl), time_update );
				}
				else if (opt.alg == COMBINE) {	
					if (n_elimmed < (countOfNodes / 100)) {
						STOPWATCH( update(d_ranges, d_states, countOfNodes, d_pivots, &wl), time_update );
					}
					else {
						STOPWATCH( update(d_ranges, d_states, countOfNodes, d_pivots, d_hasPivot), time_update );
					}
				}
				else {
					STOPWATCH( update(d_ranges, d_states, countOfNodes, d_pivots, d_hasPivot), time_update );
				}
			#endif

				/* count nodes that are elimmed but not trimmed */
				setGridDimension(&dimGrid, countOfNodes, 1024);
				gpuErrchk( cudaMemset(d_nElimmed, 0, sizeof(*d_nElimmed)) );
				countElimmed<<<dimGrid, 1024>>>(d_states, countOfNodes, d_nElimmed);
				gpuErrchk( cudaMemcpy(&n_elimmed, d_nElimmed, sizeof(n_elimmed), cudaMemcpyDeviceToHost) );
				printf("%d of %d elimmed\n", n_elimmed, countOfNodes);

			if (n_elimmed >= (countOfNodes / 100)) {
				if (n_elimmed == countOfNodes)
					break;
				wcc_done = true;
				printf("wcc done in step %d\n", step);

				timeval timer_wcc;
				startTimer(&timer_wcc);
				STOPWATCH( unsetPivots(d_states, countOfNodes), time_update );
				//update_ranges1<<<dimGrid, 1024>>>(d_ranges, d_states, countOfNodes);
				//cudaDeviceSynchronize();
				#ifdef TRIM_2
				STOPWATCH( trim2(d_nodes, d_nodesT, countOfNodes, d_edges, d_edgesT, countOfEdges, d_ranges, d_states), time_trim );
				#endif
				min_range = thrust::reduce(thrust::device, d_ranges + 1, d_ranges + 1 + countOfNodes, 0, thrust::maximum<unsigned>());
				printf("min_range = %d\n", min_range);
				gpuErrchk( cudaMalloc((void **)&d_minRange, sizeof(*d_minRange)) );
				gpuErrchk( cudaMemcpy(d_minRange, &min_range, sizeof(*d_minRange), cudaMemcpyHostToDevice) );
			#ifdef VERIFY
				if (opt.alg == DATA || opt.alg == LINEAR) {
					STOPWATCH( update_wcc(d_nodes, d_edges, d_ranges, d_states, countOfNodes, &wl, d_minRange, d_sccroot), time_update );
				}
				else {
					STOPWATCH( update_wcc(d_nodes, d_edges, d_ranges, d_states, countOfNodes, d_hasPivot, d_minRange, d_sccroot), time_update );
				}
			#else
				if (opt.alg == DATA || opt.alg == LINEAR) {
					STOPWATCH( update_wcc(d_nodes, d_edges, d_ranges, d_states, countOfNodes, &wl, d_minRange), time_update );
				}
				else {
					STOPWATCH( update_wcc(d_nodes, d_edges, d_ranges, d_states, countOfNodes, d_hasPivot, d_minRange), time_update );
				}
			#endif
				printf("time of wcc: %f ms\n", stopTimer(&timer_wcc) / 1000.0);
			}
		}
		else {
			gpuErrchk( cudaMemset(d_pivots, 0, (PIVOT_HASH_CONST + 1) * sizeof(*d_pivots)) );
			#ifdef VERIFY
				if (opt.alg == DATA || opt.alg == LINEAR) {
					STOPWATCH( update(d_ranges, d_states, countOfNodes, d_pivots, &wl, d_sccroot), time_update );
				}
				else {
					STOPWATCH( update(d_ranges, d_states, countOfNodes, d_pivots, d_hasPivot, d_sccroot), time_update );
				}
			#else
				if (opt.alg == DATA || opt.alg == LINEAR) {
					STOPWATCH( update(d_ranges, d_states, countOfNodes, d_pivots, &wl), time_update );
				}
				else {
					STOPWATCH( update(d_ranges, d_states, countOfNodes, d_pivots, d_hasPivot), time_update );
				}
			#endif
		}
		#else
		gpuErrchk( cudaMemset(d_pivots, 0, (PIVOT_HASH_CONST + 1) * sizeof(*d_pivots)) );
			#ifdef VERIFY
				if (opt.alg == DATA || opt.alg == LINEAR) {
					STOPWATCH( update(d_ranges, d_states, countOfNodes, d_pivots, &wl, d_sccroot), time_update );
				}
				else if (opt.alg == COMBINE) {
					if (n_elimmed < (countOfNodes / 100)) {
						STOPWATCH( update(d_ranges, d_states, countOfNodes, d_pivots, &wl, d_sccroot), time_update );
					}
					else {
						STOPWATCH( update(d_ranges, d_states, countOfNodes, d_pivots, d_hasPivot, d_sccroot), time_update );
					}
				}
				else {
					STOPWATCH( update(d_ranges, d_states, countOfNodes, d_pivots, d_hasPivot, d_sccroot), time_update );
				}
			#else
				if (opt.alg == DATA || opt.alg == LINEAR) {
					STOPWATCH( update(d_ranges, d_states, countOfNodes, d_pivots, &wl), time_update );
				}
				else if (opt.alg == COMBINE) {
					if (n_elimmed < (countOfNodes / 100)) {
						STOPWATCH( update(d_ranges, d_states, countOfNodes, d_pivots, &wl), time_update );
					}
					else {
						STOPWATCH( update(d_ranges, d_states, countOfNodes, d_pivots, d_hasPivot), time_update );
					}
				}
				else {
					STOPWATCH( update(d_ranges, d_states, countOfNodes, d_pivots, d_hasPivot), time_update );
				}
			#endif
		#endif

		#ifndef WCC
		//if (opt.alg == HYBRID) {
		if (opt.alg == HYBRID || opt.alg == COMBINE || opt.alg == BEST) {
			//if(step != 1 && n_elimmed < (countOfNodes / 100)) {
			if(n_elimmed < (countOfNodes / 100)) {
				setGridDimension(&dimGrid, countOfNodes, 1024);
				gpuErrchk( cudaMemset(d_nElimmed, 0, sizeof(*d_nElimmed)) );
				countElimmed<<<dimGrid, 1024>>>(d_states, countOfNodes, d_nElimmed);
				gpuErrchk( cudaMemcpy(&n_elimmed, d_nElimmed, sizeof(n_elimmed), cudaMemcpyDeviceToHost) );
				if (n_elimmed >= (countOfNodes / 100)) {
					printf("%d of %d elimmed, transfer in step %d\n", n_elimmed, countOfNodes, step);
				}
			}
		}
		#endif

		if (opt.alg == DATA || opt.alg == LINEAR) {
			if (wl.getSize() > 0)
				hasPivot = true;
		}
		else if (opt.alg == COMBINE) {
			if (n_elimmed < (countOfNodes / 100)) {
				if (wl.getSize() > 0)
					hasPivot = true;
			}
			else {
				gpuErrchk( cudaMemcpy(&hasPivot, d_hasPivot, sizeof(hasPivot), cudaMemcpyDeviceToHost) );
			}
		}
		else {
			gpuErrchk( cudaMemcpy(&hasPivot, d_hasPivot, sizeof(hasPivot), cudaMemcpyDeviceToHost) );
		}
		/*=== update === */
	}

	gpuErrchk( cudaMemcpy(states, d_states, (countOfNodes + 1)*sizeof(*d_states), cudaMemcpyDeviceToHost) );
	printf("FB done!\n");

#ifndef VERIFY
	for (int i = 1; i <= countOfNodes; i++) {
		if (states[i] & 16)
			numberOfSCC++;
	}
#else
	FILE *fp_trivial = fopen("trivial.txt", "w");
	FILE *fp_dist = fopen("scc_dist.txt", "w");
	FILE *fp_scc = fopen("parallel_scc.txt", "w");

	FILE *fp_detail = fopen("state_range_root.txt", "w");

	unsigned *scc_root = NULL;
	scc_root = (unsigned *)malloc((1 + countOfNodes) * sizeof(*scc_root));
	gpuErrchk( cudaMemcpy(scc_root, d_sccroot, (1 + countOfNodes) * sizeof(*d_sccroot), cudaMemcpyDeviceToHost) );
	
	fprintf(fp_detail, "ID\t\tSTATE\tP  T  ROOT\t\tRANGE\tRANGE\n");

	unsigned *ranges = NULL;
	ranges = (unsigned *)malloc((countOfNodes + 1) * sizeof(unsigned));
	gpuErrchk( cudaMemcpy(ranges, d_ranges, (countOfNodes + 1) * sizeof(*d_ranges), cudaMemcpyDeviceToHost) );

	int *indices;
	//int *range_root;
	int *scc_dist;
	
	//map<unsigned, unsigned> range_pivot;	/* key:range value:pivot */

	indices = (int *)malloc((1 + countOfNodes) * sizeof(int));
	//range_root = (int *)malloc((1 + countOfNodes) * sizeof(int));
	scc_dist = (int *)malloc((1 + countOfNodes) * sizeof(int));
	memset(scc_dist, 0, (1 + countOfNodes) * sizeof(int));
	vector<set<int> > vs(1 + countOfNodes);
	printf("vector init done!\n");
	int number = 0;
	int sum = 0;
	for (int i = 1; i <= countOfNodes; i++) {
		fprintf(fp_detail, "%d\t\t%d\t%d  %d  %d\t\t%d\t%u\n", i, states[i], states[i] & 16, states[i] & 8, scc_root[i], ranges[i] & PIVOT_HASH_CONST, ranges[i]);
		vs[scc_root[i]].insert(i);
		//if (states[i] & 16) {	/* pivot */
		if (scc_root[i] == i) {	/* pivot */
			indices[++number] = i;
			//if ((states[i] & 8) == 0) {	/* not trimmed */
				//range_root[ranges[i] & PIVOT_HASH_CONST] = i;	/* & may cause collision */
				//range_pivot.insert(pair<unsigned, unsigned>(ranges[i], i));
			//}
		}
		sum++;
	}
	fclose(fp_detail);
	printf("number=%d\n", number);
	printf("sum=%d\n", sum);
	sum = 0;
	set<int>::iterator site;
	for (int i = 1; i <= number; i++) {
		if (vs[indices[i]].size() == 1) {
			numberOfTrivSCC++;
			fprintf(fp_trivial, "%d\n", *vs[indices[i]].begin());
		}
		else {
			numberOfNontrivSCC++;
		}
		sum += vs[indices[i]].size();
		scc_dist[vs[indices[i]].size()]++;

		site = vs[indices[i]].begin();
		while (site != vs[indices[i]].end()) {
			fprintf(fp_scc, "%d ", *site);
			site++;
		}
		fprintf(fp_scc, "\n");
	}
	for (int i = 1; i <= countOfNodes; i++) {
		if (scc_dist[i])
			fprintf(fp_dist, "%d %d\n", i, scc_dist[i]);
	}
	fclose(fp_scc);
	fclose(fp_dist);
	printf("sum=%d\n", sum);
	numberOfSCC = numberOfNontrivSCC + numberOfTrivSCC;
#endif
	
	printf("\n");
	printf("    ->steps: %d\n", step);
	printf("    ->sum of fwdDepths: %d and avg fwd depth: %.1f\n", fwdDepths, (fwdDepths * 1.0) / step );
	printf("    ->sum of bwdDepths: %d and avg bwd depth: %.1f\n", bwdDepths, (bwdDepths * 1.0) / step );
	printf("    ->sum of trimIterations: %d and avg of trimIterations: %.1f\n", trimIterations, (trimIterations * 1.0) / step );
	printf("    ->time spent by fwd_reach: %f\n", float(time_fwd)/1000000);
	printf("    ->time spent by bwd_reach: %f\n", float(time_bwd)/1000000);
	printf("    ->time spent by trimming:  %f\n", float(time_trim)/1000000);
	printf("    ->time spent by updating:  %f\n", float(time_update)/1000000);
#ifdef VERIFY
	printf("    ->count of nontrivial SCC: %d\n", numberOfNontrivSCC);
	printf("    ->count of trivial SCC:    %d\n", numberOfTrivSCC);
#endif
	
	float f = float(stopTimer(&timer))/1000000;
	
	gpuErrchk( cudaFree(d_nodes) );
	gpuErrchk( cudaFree(d_nodesT) );
	gpuErrchk( cudaFree(d_edges) );
	gpuErrchk( cudaFree(d_edgesT) );
	gpuErrchk( cudaFree(d_pivots) );
	gpuErrchk( cudaFree(d_ranges) );
	gpuErrchk( cudaFree(d_states) );
	
	return make_pair(numberOfSCC, f);
}

