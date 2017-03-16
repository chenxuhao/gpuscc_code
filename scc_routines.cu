#include <stdint.h>

#include "scc_routines.h"
#include "GPUerrchk.h"
#include "scc.h"

#define BLKSIZE 256


static __global__ void unsetPivots_kernel(unsigned char *states, unsigned countOfNodes)
{
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x + 1;
	if (id > countOfNodes) return;

	unsigned char my_state = states[id];
	if (my_state == 19) {	
		states[id] = 0;
	}
}

void unsetPivots(unsigned char *d_states, unsigned countOfNodes) {

	dim3 dimGrid;
	
	setGridDimension(&dimGrid, countOfNodes, BLKSIZE);
	unsetPivots_kernel<<<dimGrid, BLKSIZE>>>(d_states, countOfNodes);
}

static __global__ void update_ranges_kernel(unsigned *ranges, unsigned char *states, unsigned countOfNodes)
{
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x + 1;
	if (id > countOfNodes) return;

	unsigned new_subgraph;
	unsigned par_subgraph;
	unsigned char my_state = states[id];

	if (!isElim(&my_state)) {	
		//updating part
		new_subgraph = (isFVis(&my_state) ? 1 : 0) + (isBVis(&my_state) ? 2 : 0);	// F intersec B == 3, F/B == 1 B/F == 2 (V/F)/B == 0
		if(new_subgraph == 3) {
			setElim(&states[id]);
			return;
		}
		
		par_subgraph = ranges[id];
		ranges[id] = 3 * par_subgraph + new_subgraph;
		states[id] = 0;
	}
}

void update_ranges(unsigned *d_ranges, unsigned char *d_states, unsigned countOfNodes) {

	dim3 dimGrid;
	
	setGridDimension(&dimGrid, countOfNodes, BLKSIZE);
	update_ranges_kernel<<<dimGrid, BLKSIZE>>>(d_ranges, d_states, countOfNodes);
}

#ifdef VERIFY
static __global__ void selectPivots_kernel(unsigned *ranges, unsigned char *states, unsigned countOfNodes, unsigned *pivots, Worklist2 wl, unsigned *scc_root)
#else
static __global__ void selectPivots_kernel(unsigned *ranges, unsigned char *states, unsigned countOfNodes, unsigned *pivots, Worklist2 wl)
#endif
{
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x + 1;
	if (id > countOfNodes) return;

	unsigned char my_state = states[id];

	if (!isElim(&my_state)) {	

		//pivots selection part
		unsigned my_range = ranges[id];
		if(pivots[my_range & PIVOT_HASH_CONST] == 0) {
			if(atomicCAS(&pivots[my_range & PIVOT_HASH_CONST], 0, id) == 0) {
			#ifdef VERIFY
				scc_root[id] = id;
			#endif
				wl.push(id);
				states[id] = 19;	// set F B P bit
			}
		}
	}
}

#ifdef VERIFY
static __global__ void selectPivots_kernel(unsigned *ranges, unsigned char *states, unsigned countOfNodes, unsigned *pivots, bool *hasPivot, unsigned *scc_root)
#else
static __global__ void selectPivots_kernel(unsigned *ranges, unsigned char *states, unsigned countOfNodes, unsigned *pivots, bool *hasPivot)
#endif
{
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x + 1;
	if (id > countOfNodes) return;

	unsigned char my_state = states[id];

	if (!isElim(&my_state)) {	

		//pivots selection part
		unsigned my_range = ranges[id];
		if(pivots[my_range & PIVOT_HASH_CONST] == 0) {
			if(atomicCAS(&pivots[my_range & PIVOT_HASH_CONST], 0, id) == 0) {
			#ifdef VERIFY
				scc_root[id] = id;
			#endif
				*hasPivot = true;
				states[id] = 19;	// set F B P bit
			}
		}
	}
}

#ifdef VERIFY
void selectPivots(unsigned *d_ranges, unsigned char *d_states, unsigned countOfNodes, unsigned *d_pivots, Worklist2 *wlptr, unsigned *scc_root) {
#else
void selectPivots(unsigned *d_ranges, unsigned char *d_states, unsigned countOfNodes, unsigned *d_pivots, Worklist2 *wlptr) {
#endif

	dim3 dimGrid;
	
	setGridDimension(&dimGrid, countOfNodes, BLKSIZE);
	#ifdef VERIFY
	selectPivots_kernel<<<dimGrid, BLKSIZE>>>(d_ranges, d_states, countOfNodes, d_pivots, *wlptr, scc_root);
	#else
	selectPivots_kernel<<<dimGrid, BLKSIZE>>>(d_ranges, d_states, countOfNodes, d_pivots, *wlptr);
	#endif
}

#ifdef VERIFY
void selectPivots(unsigned *d_ranges, unsigned char *d_states, unsigned countOfNodes, unsigned *d_pivots, bool *hasPivot, unsigned *scc_root) {
#else
void selectPivots(unsigned *d_ranges, unsigned char *d_states, unsigned countOfNodes, unsigned *d_pivots, bool *hasPivot) {
#endif

	dim3 dimGrid;
	
	setGridDimension(&dimGrid, countOfNodes, BLKSIZE);
	#ifdef VERIFY
	selectPivots_kernel<<<dimGrid, BLKSIZE>>>(d_ranges, d_states, countOfNodes, d_pivots, hasPivot, scc_root);
	#else
	selectPivots_kernel<<<dimGrid, BLKSIZE>>>(d_ranges, d_states, countOfNodes, d_pivots, hasPivot);
	#endif
}

#ifdef VERIFY
static __global__ void update_kernel(unsigned *ranges, unsigned char *states, unsigned countOfNodes, unsigned *pivots, Worklist2 wl, unsigned *scc_root)
#else
static __global__ void update_kernel(unsigned *ranges, unsigned char *states, unsigned countOfNodes, unsigned *pivots, Worklist2 wl)
#endif
{
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x + 1;
	if (id > countOfNodes) return;

	unsigned new_subgraph;
	unsigned par_subgraph;
	unsigned char my_state = states[id];

	if (!isElim(&my_state)) {	
		//updating part
		new_subgraph = (isFVis(&my_state) ? 1 : 0) + (isBVis(&my_state) ? 2 : 0);	// F intersec B == 3, F/B == 1 B/F == 2 (V/F)/B == 0
		if(new_subgraph == 3) {
			setElim(&states[id]);
			return;
		}
		
		par_subgraph = ranges[id];
		unsigned new_range = 3 * par_subgraph + new_subgraph;
		ranges[id] = new_range;
		//states[id] &= 12; //10001100 clear pivot, visitation and expension bits
		states[id] = 0;

		//pivots selection part
		if(pivots[new_range & PIVOT_HASH_CONST] == 0) {
			if(atomicCAS(&pivots[new_range & PIVOT_HASH_CONST], 0, id) == 0) {
			#ifdef VERIFY
				scc_root[id] = id;
			#endif
				wl.push((unsigned)id);
				states[id] = 19;	// set F B P bit
			}
		}
	}
}

#ifdef VERIFY
static __global__ void update_kernel(unsigned *ranges, unsigned char *states, unsigned countOfNodes, unsigned *pivots, bool *hasPivot, unsigned *scc_root)
#else
static __global__ void update_kernel(unsigned *ranges, unsigned char *states, unsigned countOfNodes, unsigned *pivots, bool *hasPivot)
#endif
{
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x + 1;
	if (id > countOfNodes) return;

	unsigned new_subgraph;
	unsigned par_subgraph;
	unsigned char my_state = states[id];

	if (!isElim(&my_state)) {	
		//updating part
		new_subgraph = (isFVis(&my_state) ? 1 : 0) + (isBVis(&my_state) ? 2 : 0);	// F intersec B == 3, F/B == 1 B/F == 2 (V/F)/B == 0
		if(new_subgraph == 3) {
			setElim(&states[id]);
			return;
		}
		
		par_subgraph = ranges[id];
		unsigned new_range = 3 * par_subgraph + new_subgraph;
		ranges[id] = new_range;
		//states[id] &= 12; //10001100 clear pivot, visitation and expension bits
		states[id] = 0;

		//pivots selection part
		if(pivots[new_range & PIVOT_HASH_CONST] == 0) {
			if(atomicCAS(&pivots[new_range & PIVOT_HASH_CONST], 0, id) == 0) {
			#ifdef VERIFY
				scc_root[id] = id;
			#endif
				*hasPivot = true;
				states[id] = 19;	// set F B P bit
			}
		}
	}
}

static __global__ void trim_first_kernel(unsigned *nodes, unsigned *nodesT, unsigned *edges, unsigned *edgesT, unsigned char *states, unsigned countOfNodes, bool volatile *end) {
	
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x + 1;
	if (id > countOfNodes) return;

	unsigned my_start;
	unsigned my_end;
	unsigned nbi;
	
	bool isActive = false;

	if(!isElim(&states[id])) {
		my_start = nodes[id];
		my_end = nodes[id + 1];

		for(unsigned i = my_start; i < my_end; i++) {
			nbi = edges[i];
			if(!isElim(&states[nbi]) && id != nbi) {
				isActive = true;
				break;
			}
		}		
		if(!isActive) {
			/*setElim(&states[id]); setTrim(&states[id]);*/
			states[id] = 28;	//states[id] = 12;
			*end = false;
			return;
		}
		
		isActive = false;
		my_start = nodesT[id];
		my_end = nodesT[id + 1];

		for(unsigned i = my_start; i < my_end; i++) {
			nbi = edgesT[i];
			if(!isElim(&states[nbi]) && id != nbi) {
				isActive = true;
				break;
			}
		}	
		if(!isActive) {
			/*setElim(&states[id]); setTrim(&states[id]);*/
			states[id] = 28;	//states[id] = 12;
			*end = false;
			return;
		}	
	}
}

#ifdef VERIFY
static __global__ void trim1_kernel(unsigned *nodes, unsigned *nodesT, unsigned *edges, unsigned *edgesT, unsigned char *states, unsigned countOfNodes, bool volatile *end, unsigned *scc_root) {
#else
static __global__ void trim1_kernel(unsigned *nodes, unsigned *nodesT, unsigned *edges, unsigned *edgesT, unsigned char *states, unsigned countOfNodes, bool volatile *end) {
#endif
	
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x + 1;
	if (id > countOfNodes) return;

	unsigned my_start;
	unsigned my_end;
	unsigned nbi;
	
	bool isActive = false;

	if(!isElim(&states[id])) {
		my_start = nodes[id];
		my_end = nodes[id + 1];

		for(unsigned i = my_start; i < my_end; i++) {
			nbi = edges[i];
			if(!isElim(&states[nbi]) && id != nbi) {
				isActive = true;
				break;
			}
		}		
		if(!isActive) {
		#ifdef VERIFY
			scc_root[id] = id;
		#endif
			/*setElim(&states[id]); setTrim(&states[id]);*/
			states[id] = 28;	//states[id] = 12;
			*end = false;
			return;
		}
		
		isActive = false;
		my_start = nodesT[id];
		my_end = nodesT[id + 1];

		for(unsigned i = my_start; i < my_end; i++) {
			nbi = edgesT[i];
			if(!isElim(&states[nbi]) && id != nbi) {
				isActive = true;
				break;
			}
		}	
		if(!isActive) {
		#ifdef VERIFY
			scc_root[id] = id;
		#endif
			/*setElim(&states[id]); setTrim(&states[id]);*/
			states[id] = 28;	//states[id] = 12;
			*end = false;
			return;
		}	
	}
}


#ifdef VERIFY
static __global__ void trim_kernel(unsigned *nodes, unsigned *nodesT, unsigned *edges, unsigned *edgesT, unsigned *ranges, unsigned char *states, unsigned countOfNodes, bool volatile *end, unsigned *scc_root) {
#else
static __global__ void trim_kernel(unsigned *nodes, unsigned *nodesT, unsigned *edges, unsigned *edgesT, unsigned *ranges, unsigned char *states, unsigned countOfNodes, bool volatile *end) {
#endif
	
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x + 1;
	if (id > countOfNodes) return;

	unsigned my_range;
	unsigned my_start;
	unsigned my_end;
	unsigned nbi;
	
	bool isActive = false;

	if(!isElim(&states[id])) {
		my_range = ranges[id];
		my_start = nodes[id];
		my_end = nodes[id + 1];

		for(unsigned i = my_start; i < my_end; i++) {
			nbi = edges[i];
			if(my_range == ranges[nbi] && !isElim(&states[nbi]) && id != nbi) {
				isActive = true;
				break;
			}
		}		
		if(!isActive) {
		#ifdef VERIFY
			scc_root[id] = id;
		#endif
			/*setElim(&states[id]); setTrim(&states[id]);*/
			states[id] = 28;	//states[id] = 12;
			*end = false;
			return;
		}
		
		isActive = false;
		my_start = nodesT[id];
		my_end = nodesT[id + 1];

		for(unsigned i = my_start; i < my_end; i++) {
			nbi = edgesT[i];
			if(my_range == ranges[nbi] && !isElim(&states[nbi]) && id != nbi) {
				isActive = true;
				break;
			}
		}	
		if(!isActive) {
		#ifdef VERIFY
			scc_root[id] = id;
		#endif
			/*setElim(&states[id]); setTrim(&states[id]);*/
			states[id] = 28;	//states[id] = 12;
			*end = false;
			return;
		}	
	}
}

//static __global__ void trim2_kernel(unsigned *nodes, unsigned *nodesT, unsigned *edges, unsigned *edgesT, unsigned *ranges, unsigned char *states, unsigned countOfNodes, bool volatile *end) {
static __global__ void trim2_kernel(unsigned *nodes, unsigned *nodesT, unsigned *edges, unsigned *edgesT, unsigned *ranges, unsigned char *states, unsigned countOfNodes) {

	unsigned id = blockIdx.x * blockDim.x + threadIdx.x + 1;
	if (id > countOfNodes) return;

	unsigned my_start;
	unsigned my_end;
	unsigned my_range;
	unsigned char my_state;
	bool isActive = false;

	unsigned nbi;
	unsigned actualNb;
	unsigned actualNbs = 0;

	my_range = ranges[id];
	my_state = states[id];

	if (!isElim(&my_state)) {
		// outgoing edges
		my_start = nodes[id];
		my_end = nodes[id + 1];
	
		for (unsigned i = my_start; i < my_end; i++) {
			nbi = edges[i];
			if (id != nbi && my_range == ranges[nbi] && !isElim(&states[nbi])) {
				actualNbs++;
				if (actualNbs > 1) {
					//isActive = true;
					break;
				}
				actualNb = nbi;
			}
		}
		if (actualNbs == 1) {
			my_start = nodes[actualNb];
			my_end = nodes[actualNb + 1];
			actualNbs = 0;
			
			for (unsigned i = my_start; i < my_end; i++) {
				nbi = edges[i];
				if (actualNb != nbi && my_range == ranges[nbi] && !isElim(&states[nbi])) {
					if (nbi != id) {
						isActive = true;
						break;
					}
					actualNbs++;
					/*
					if (actualNbs > 1) {
						//isActive = true;
						break;
					}
					*/
				}
			}

			if (!isActive && actualNbs == 1) {
				//printf("fwd:id=%d actualNb=%d both Elimmed\n", id, actualNb);
				if (id < actualNb) {
					states[id] = 20;
					states[actualNb] = 4;
				}
				else {
					states[id] = 4;
					states[actualNb] = 20;
				}
				//*end = false;
				return;
			}	
		}

		// incoming edges
		my_start = nodesT[id];
		my_end = nodesT[id + 1];
		actualNbs = 0;
		isActive = false;

		for (unsigned i = my_start; i < my_end; i++) {
			nbi = edgesT[i];
			if (id != nbi && my_range == ranges[nbi] && !isElim(&states[nbi])) {
				actualNbs++;
				if (actualNbs > 1) {
					//isActive = true;
					break;
				}
				actualNb = nbi;
			}
		}
		if (actualNbs == 1) {
			my_start = nodesT[actualNb];
			my_end = nodesT[actualNb + 1];
			actualNbs = 0;

			for (unsigned i = my_start; i < my_end; i++) {
				nbi = edgesT[i];
				if (actualNb != nbi && my_range == ranges[nbi] && !isElim(&states[nbi])) {
					if (nbi != id) {
						isActive = true;
						break;
					}
					actualNbs++;
					/*
					if (actualNbs > 1) {
						isActive = true;
						break;
					}
					*/
				}
			}

			if (!isActive && actualNbs == 1) {
				//printf("bwd:id=%d actualNb=%d both Elimmed\n", id, actualNb);
				if (id < actualNb) {
					states[id] = 20;
					states[actualNb] = 4;
				}
				else {
					states[id] = 4;
					states[actualNb] = 20;
				}
				//*end = false;
				return;
			}
			
		}
	}
}

#ifdef VERIFY
void update(unsigned *d_ranges, unsigned char *d_states, unsigned countOfNodes, unsigned *d_pivots, Worklist2 *wlptr, unsigned *scc_root) {
#else
void update(unsigned *d_ranges, unsigned char *d_states, unsigned countOfNodes, unsigned *d_pivots, Worklist2 *wlptr) {
#endif

	dim3 dimGrid;
	
	setGridDimension(&dimGrid, countOfNodes, BLKSIZE);
	#ifdef VERIFY
	update_kernel<<<dimGrid, BLKSIZE>>>(d_ranges, d_states, countOfNodes, d_pivots, *wlptr, scc_root);
	#else
	update_kernel<<<dimGrid, BLKSIZE>>>(d_ranges, d_states, countOfNodes, d_pivots, *wlptr);
	#endif
}

#ifdef VERIFY
void update(unsigned *d_ranges, unsigned char *d_states, unsigned countOfNodes, unsigned *d_pivots, bool *hasPivot, unsigned *scc_root) {
#else
void update(unsigned *d_ranges, unsigned char *d_states, unsigned countOfNodes, unsigned *d_pivots, bool *hasPivot) {
#endif

	dim3 dimGrid;
	
	setGridDimension(&dimGrid, countOfNodes, BLKSIZE);
	#ifdef VERIFY
	update_kernel<<<dimGrid, BLKSIZE>>>(d_ranges, d_states, countOfNodes, d_pivots, hasPivot, scc_root);
	#else
	update_kernel<<<dimGrid, BLKSIZE>>>(d_ranges, d_states, countOfNodes, d_pivots, hasPivot);
	#endif
}

unsigned trim_first(unsigned *d_nodes, unsigned *d_nodesT, unsigned countOfNodes, unsigned *d_edges, unsigned *d_edgesT, unsigned countOfEdges, unsigned char *d_states) {
	
	bool volatile *d_end;
	bool end = false;
	dim3 dimGrid;
	
	unsigned nmbOfIterations = 0;
	
	setGridDimension(&dimGrid, countOfNodes, BLKSIZE);	
	
	gpuErrchk( cudaMalloc((void**)&d_end, sizeof(*d_end)) );

	while(!end) {
		nmbOfIterations++;
		gpuErrchk( cudaMemset((void *)d_end, true, sizeof(*d_end)) );
		trim_first_kernel<<<dimGrid, BLKSIZE>>>(d_nodes, d_nodesT, d_edges, d_edgesT, d_states, countOfNodes, d_end);
		gpuErrchk( cudaMemcpy(&end, (void *)d_end, sizeof(*d_end), cudaMemcpyDeviceToHost) );	
	}	
	
	gpuErrchk( cudaFree((void *) d_end) );
	
	return nmbOfIterations;
}

#ifdef VERIFY
unsigned trim1(unsigned *d_nodes, unsigned *d_nodesT, unsigned countOfNodes, unsigned *d_edges, unsigned *d_edgesT, unsigned countOfEdges, unsigned char *d_states, unsigned *scc_root) {
#else
unsigned trim1(unsigned *d_nodes, unsigned *d_nodesT, unsigned countOfNodes, unsigned *d_edges, unsigned *d_edgesT, unsigned countOfEdges, unsigned char *d_states) {
#endif
	
	bool volatile *d_end;
	bool end = false;
	dim3 dimGrid;
	
	unsigned nmbOfIterations = 0;
	
	setGridDimension(&dimGrid, countOfNodes, BLKSIZE);	
	
	gpuErrchk( cudaMalloc((void**)&d_end, sizeof(*d_end)) );

	while(!end) {
		gpuErrchk( cudaMemset((void *)d_end, true, sizeof(*d_end)) );
	#ifdef VERIFY
		trim1_kernel<<<dimGrid, BLKSIZE>>>(d_nodes, d_nodesT, d_edges, d_edgesT, d_states, countOfNodes, d_end, scc_root);
	#else
		trim1_kernel<<<dimGrid, BLKSIZE>>>(d_nodes, d_nodesT, d_edges, d_edgesT, d_states, countOfNodes, d_end);
	#endif
		gpuErrchk( cudaMemcpy(&end, (void *)d_end, sizeof(*d_end), cudaMemcpyDeviceToHost) );		
		nmbOfIterations++;
	}	
	
	gpuErrchk( cudaFree((void *) d_end) );
	
	return nmbOfIterations;
}

#ifdef VERIFY
unsigned trim(unsigned *d_nodes, unsigned *d_nodesT, unsigned countOfNodes, unsigned *d_edges, unsigned *d_edgesT, unsigned countOfEdges, unsigned *d_ranges, unsigned char *d_states, unsigned *scc_root) {
#else
unsigned trim(unsigned *d_nodes, unsigned *d_nodesT, unsigned countOfNodes, unsigned *d_edges, unsigned *d_edgesT, unsigned countOfEdges, unsigned *d_ranges, unsigned char *d_states) {
#endif
	
	bool volatile *d_end;
	bool end = false;
	dim3 dimGrid;
	
	unsigned nmbOfIterations = 0;
	
	setGridDimension(&dimGrid, countOfNodes, BLKSIZE);	
	
	gpuErrchk( cudaMalloc((void**)&d_end, sizeof(*d_end)) );

	while(!end) {
		gpuErrchk( cudaMemset((void *)d_end, true, sizeof(*d_end)) );
	#ifdef VERIFY
		trim_kernel<<<dimGrid, BLKSIZE>>>(d_nodes, d_nodesT, d_edges, d_edgesT, d_ranges, d_states, countOfNodes, d_end, scc_root);
	#else
		trim_kernel<<<dimGrid, BLKSIZE>>>(d_nodes, d_nodesT, d_edges, d_edgesT, d_ranges, d_states, countOfNodes, d_end);
	#endif
		gpuErrchk( cudaMemcpy(&end, (void *)d_end, sizeof(*d_end), cudaMemcpyDeviceToHost) );		
		nmbOfIterations++;
	}	
	
	gpuErrchk( cudaFree((void *) d_end) );
	
	return nmbOfIterations;
}

unsigned trim2(unsigned *d_nodes, unsigned *d_nodesT, unsigned countOfNodes, unsigned *d_edges, unsigned *d_edgesT, unsigned countOfEdges, unsigned *d_ranges, unsigned char *d_states) {

	//bool volatile *d_end;
	//bool end = false;
	dim3 dimGrid;

	unsigned nmbOfIterations = 0;

	setGridDimension(&dimGrid, countOfNodes, BLKSIZE);

	//gpuErrchk( cudaMalloc((void **)&d_end, sizeof(*d_end)) );

	//while (!end) {
		//gpuErrchk( cudaMemset((void *)d_end, true, sizeof(*d_end)) );
		//trim2_kernel<<<dimGrid, BLKSIZE>>>(d_nodes, d_nodesT, d_edges, d_edgesT, d_ranges, d_states, countOfNodes, d_end);
		trim2_kernel<<<dimGrid, BLKSIZE>>>(d_nodes, d_nodesT, d_edges, d_edgesT, d_ranges, d_states, countOfNodes);
		//gpuErrchk( cudaMemcpy(&end, (void *)d_end, sizeof(*d_end), cudaMemcpyDeviceToHost) );
		//nmbOfIterations++;
	//}

	//gpuErrchk( cudaFree((void *)d_end) );

	return nmbOfIterations;
}

#undef BLKSIZE
