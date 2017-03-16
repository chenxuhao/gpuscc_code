#include <stdint.h>

#include "wcc.h"
#include "GPUerrchk.h"

	
static __global__ void computeWCC(unsigned *nodes, unsigned *edges, unsigned *ranges, unsigned char *states, unsigned countOfNodes, bool volatile *end, unsigned *wcc) {

	unsigned id = blockDim.x * blockIdx.x + threadIdx.x + 1;
	if (id > countOfNodes) return;

	unsigned my_start;
	unsigned my_end;
	unsigned nbi;
	unsigned min_wcc;
	unsigned my_range;

	bool update = false;

	if(!isElim(&states[id])) {
		my_start = nodes[id];
		my_end = nodes[id + 1];
		my_range = ranges[id];
		min_wcc = wcc[id];
		unsigned wcc_nbi;	

		for(unsigned i = my_start; i < my_end; i++) {
			nbi = edges[i];
			if(my_range == ranges[nbi] && !isElim(&states[nbi])) {
				wcc_nbi = wcc[nbi];
				if (wcc_nbi < min_wcc) {
					min_wcc = wcc_nbi;
					update = true;
				}
			}
		}
		if (update) 
			wcc[id] = min_wcc;

		__syncthreads();

		unsigned  wcc_k = wcc[min_wcc];
		if (min_wcc != id && min_wcc != wcc_k) {
			wcc[id] = wcc_k;
			update = true;
		}

		__syncthreads();
		if (update)
			*end = false;
	}
}

__global__ void update_ranges1(unsigned *ranges, unsigned char *states, unsigned countOfNodes) {

	unsigned id = blockDim.x * blockIdx.x + threadIdx.x + 1;
	if (id > countOfNodes) return;

	unsigned new_subgraph;
	unsigned par_subgraph;
	unsigned char my_state = states[id];

	if (!isElim(&my_state)) {
		new_subgraph = (isFVis(&my_state) ? 1 : 0) + (isBVis(&my_state) ? 2 : 0);

		if (new_subgraph == 3) {
			setElim(&states[id]);
			return;
		}

		states[id] = 0;

		par_subgraph = ranges[id];
		ranges[id] = 3 * par_subgraph + new_subgraph;
	}
}

#ifdef VERIFY
static __global__ void update_ranges2(unsigned *ranges, unsigned char *states, unsigned countOfNodes, unsigned *wcc, Worklist2 wl, unsigned *min_range, unsigned *scc_root) {
#else
static __global__ void update_ranges2(unsigned *ranges, unsigned char *states, unsigned countOfNodes, unsigned *wcc, Worklist2 wl, unsigned *min_range) {
#endif
	
	unsigned id = blockDim.x * blockIdx.x + threadIdx.x + 1;
	if (id > countOfNodes) return;

	unsigned wcc_id;
	unsigned new_range;
	
	if (!isElim(&states[id])) {
		wcc_id = wcc[id];
		if (wcc_id == id) {
			new_range = atomicAdd(min_range, 1);

			ranges[id] = new_range;
			__syncthreads();
		#ifdef VERIFY
			scc_root[id] = id;
		#endif
			states[id] = 19;
			wl.push(id);
		}
	}
}

#ifdef VERIFY
static __global__ void update_ranges2(unsigned *ranges, unsigned char *states, unsigned countOfNodes, unsigned *wcc, bool *hasPivot, unsigned *min_range, unsigned *scc_root) {
#else
static __global__ void update_ranges2(unsigned *ranges, unsigned char *states, unsigned countOfNodes, unsigned *wcc, bool *hasPivot, unsigned *min_range) {
#endif
	
	unsigned id = blockDim.x * blockIdx.x + threadIdx.x + 1;
	if (id > countOfNodes) return;

	unsigned wcc_id;
	unsigned new_range;
	
	if (!isElim(&states[id])) {
		wcc_id = wcc[id];
		if (wcc_id == id) {
			new_range = atomicAdd(min_range, 1);

			ranges[id] = new_range;
			__syncthreads();
		#ifdef VERIFY
			scc_root[id] = id;
		#endif
			states[id] = 19;
			*hasPivot = true;
		}
	}
}

static __global__ void update_ranges3(unsigned *ranges, unsigned char *states, unsigned countOfNodes, unsigned *wcc) {

	unsigned id = blockIdx.x * blockDim.x + threadIdx.x + 1;
	if (id > countOfNodes) return;

	if (!isElim(&states[id])) {
		unsigned wcc_id = wcc[id];
		if (wcc_id != id)
			ranges[id] = ranges[wcc_id];
	}
}

#ifdef VERIFY	
unsigned update_wcc(unsigned *d_nodes, unsigned *d_edges, unsigned *d_ranges, unsigned char *d_states, unsigned countOfNodes, Worklist2 *wlptr, unsigned *d_minRange, unsigned *scc_root) {
#else
unsigned update_wcc(unsigned *d_nodes, unsigned *d_edges, unsigned *d_ranges, unsigned char *d_states, unsigned countOfNodes, Worklist2 *wlptr, unsigned *d_minRange) {
#endif
				
	bool volatile *d_end;
	bool end = false;
	
	unsigned *d_wcc;
	unsigned depth = 0;
	
	dim3 dimGrid;
	setGridDimension(&dimGrid, countOfNodes, BLOCKSIZE);

	gpuErrchk( cudaMalloc((void**)&d_end, sizeof(*d_end)) );
	gpuErrchk( cudaMalloc((void **)&d_wcc, (1 + countOfNodes) * sizeof(*d_wcc)) );
	/* d_wcc[i] = i */
	thrust::sequence(thrust::device, d_wcc, d_wcc + 1 + countOfNodes);
	
	while(!end) {
		gpuErrchk( cudaMemset((void *)d_end, true, sizeof(*d_end)) ); 
		computeWCC<<<dimGrid, BLOCKSIZE>>>(d_nodes, d_edges, d_ranges, d_states, countOfNodes, d_end, d_wcc);
		gpuErrchk( cudaMemcpy(&end, (const void *)d_end, sizeof(*d_end), cudaMemcpyDeviceToHost) );
		depth++;
	}

#ifdef VERIFY	
	update_ranges2<<<dimGrid, BLOCKSIZE>>>(d_ranges, d_states, countOfNodes, d_wcc, *wlptr, d_minRange, scc_root);
#else
	update_ranges2<<<dimGrid, BLOCKSIZE>>>(d_ranges, d_states, countOfNodes, d_wcc, *wlptr, d_minRange);
#endif
	update_ranges3<<<dimGrid, BLOCKSIZE>>>(d_ranges, d_states, countOfNodes, d_wcc);
	//cudaDeviceSynchronize();

	gpuErrchk( cudaFree((void *)d_end) );
	gpuErrchk( cudaFree((void *)d_wcc) );
	
	return depth;
}

#ifdef VERIFY	
unsigned update_wcc(unsigned *d_nodes, unsigned *d_edges, unsigned *d_ranges, unsigned char *d_states, unsigned countOfNodes, bool *hasPivot, unsigned *d_minRange, unsigned *scc_root) {
#else
unsigned update_wcc(unsigned *d_nodes, unsigned *d_edges, unsigned *d_ranges, unsigned char *d_states, unsigned countOfNodes, bool *hasPivot, unsigned *d_minRange) {
#endif
				
	bool volatile *d_end;
	bool end = false;
	
	unsigned *d_wcc;
	unsigned depth = 0;
	
	dim3 dimGrid;
	setGridDimension(&dimGrid, countOfNodes, BLOCKSIZE);

	gpuErrchk( cudaMalloc((void**)&d_end, sizeof(*d_end)) );
	gpuErrchk( cudaMalloc((void **)&d_wcc, (1 + countOfNodes) * sizeof(*d_wcc)) );
	/* d_wcc[i] = i */
	thrust::sequence(thrust::device, d_wcc, d_wcc + 1 + countOfNodes);
	
	while(!end) {
		gpuErrchk( cudaMemset((void *)d_end, true, sizeof(*d_end)) ); 
		computeWCC<<<dimGrid, BLOCKSIZE>>>(d_nodes, d_edges, d_ranges, d_states, countOfNodes, d_end, d_wcc);
		gpuErrchk( cudaMemcpy(&end, (const void *)d_end, sizeof(*d_end), cudaMemcpyDeviceToHost) );
		depth++;
	}

#ifdef VERIFY	
	update_ranges2<<<dimGrid, BLOCKSIZE>>>(d_ranges, d_states, countOfNodes, d_wcc, hasPivot, d_minRange, scc_root);
#else
	update_ranges2<<<dimGrid, BLOCKSIZE>>>(d_ranges, d_states, countOfNodes, d_wcc, hasPivot, d_minRange);
#endif
	update_ranges3<<<dimGrid, BLOCKSIZE>>>(d_ranges, d_states, countOfNodes, d_wcc);
	//cudaDeviceSynchronize();

	gpuErrchk( cudaFree((void *)d_end) );
	gpuErrchk( cudaFree((void *)d_wcc) );
	
	return depth;
}
