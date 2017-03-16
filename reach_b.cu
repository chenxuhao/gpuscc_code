#include <stdint.h>

#include "scc.h"
#include "reach_b.h"
#include "GPUerrchk.h"
#include "timing.h"
#include "cub/cub.cuh"
#include "worklistc.h"

//#define DEBUG
//#define EXPAND_BY_CTA
#define BLKSIZE 256

static __global__ void bwd_step(unsigned *nodes, unsigned *edges, unsigned *ranges, unsigned char *states, unsigned countOfNodes, bool volatile *end) {

	unsigned id = blockDim.x * blockIdx.x + threadIdx.x + 1;
	typedef cub::BlockScan<unsigned, BLKSIZE> BlockScan;
	__shared__ BlockScan::TempStorage temp_storage;
	__shared__ unsigned gather_offsets[BLKSIZE];
	__shared__ unsigned srcsrc[BLKSIZE];
	__shared__ unsigned sh_ranges[BLKSIZE];
	__shared__ bool sh_end;

	gather_offsets[threadIdx.x] = 0;
	if (id <= countOfNodes)
		sh_ranges[threadIdx.x] = ranges[id];
	__syncthreads();
	if (threadIdx.x == 0)
		sh_end = true;
	__syncthreads();

	unsigned neighboroffset = 0;
	unsigned neighborsize = 0;
	unsigned scratch_offset = 0;
	unsigned total_edges = 0;

	if(id <= countOfNodes && (states[id] & 37) == 1) {
		setBExt(&states[id]);
		neighboroffset = nodes[id];
		neighborsize = nodes[id + 1] - neighboroffset;
	}
	__syncthreads();

	BlockScan(temp_storage).ExclusiveSum(neighborsize, scratch_offset, total_edges);

	unsigned done = 0;
	unsigned neighborsdone = 0;
	while ((int)total_edges > 0) {
		__syncthreads();
		unsigned i;
		unsigned index;
		for (i = 0; neighborsdone + i < neighborsize && (index = scratch_offset + i - done) < BLKSIZE; i++) {
			//gather_offsets[scratch_offset + i - done] = neighboroffset + neighborsdone + i;
			//srcsrc[scratch_offset + i - done] = id;
			gather_offsets[index] = neighboroffset + neighborsdone + i;
			srcsrc[index] = id;
		}
		neighborsdone += i;
		scratch_offset += i;
		__syncthreads();

		unsigned src;
		unsigned dst;

		if (threadIdx.x < total_edges) {
			src = srcsrc[threadIdx.x];
			dst = edges[gather_offsets[threadIdx.x]];
			//if (src != dst) {
			if ((states[dst] & 5) == 0 && sh_ranges[src - blockDim.x * blockIdx.x - 1] == ranges[dst]) {
				setBVis(&states[dst]);
				sh_end = false;
			}
			//}
		}
		done += BLKSIZE;
		total_edges -= BLKSIZE;
	}
	__syncthreads();
	if (threadIdx.x == 0 && sh_end == false)
		*end = false;
	__syncthreads();
}

static __global__ void fwd_step(unsigned *nodes, unsigned *edges, unsigned *ranges, unsigned char *states, unsigned countOfNodes, bool volatile *end) {
	unsigned id = blockDim.x * blockIdx.x + threadIdx.x + 1;
	typedef cub::BlockScan<unsigned, BLKSIZE> BlockScan;
	__shared__ BlockScan::TempStorage temp_storage;
	__shared__ unsigned gather_offsets[BLKSIZE];
	__shared__ unsigned srcsrc[BLKSIZE];
	__shared__ unsigned sh_ranges[BLKSIZE];
	__shared__ bool sh_end;
	gather_offsets[threadIdx.x] = 0;
	if (id <= countOfNodes)
		sh_ranges[threadIdx.x] = ranges[id];
	__syncthreads();
	if (threadIdx.x == 0)
		sh_end = true;
	__syncthreads();
	unsigned neighboroffset = 0;
	unsigned neighborsize = 0;
	unsigned scratch_offset = 0;
	unsigned total_edges = 0;
	if(id <= countOfNodes && (states[id] & 70) == 2) {
		setFExt(&states[id]);
		neighboroffset = nodes[id];
		neighborsize = nodes[id + 1] - neighboroffset;
	}
	__syncthreads();
	BlockScan(temp_storage).ExclusiveSum(neighborsize, scratch_offset, total_edges);
	unsigned done = 0;
	unsigned neighborsdone = 0;
	while ((int)total_edges > 0) {
		__syncthreads();
		unsigned i;
		unsigned index;
		for (i = 0; neighborsdone + i < neighborsize && (index = scratch_offset + i - done) < BLKSIZE; i++) {
			//gather_offsets[scratch_offset + i - done] = neighboroffset + neighborsdone + i;
			//srcsrc[scratch_offset + i - done] = id;
			gather_offsets[index] = neighboroffset + neighborsdone + i;
			srcsrc[index] = id;
		}
		neighborsdone += i;
		scratch_offset += i;
		__syncthreads();

		unsigned src;
		unsigned dst;

		if (threadIdx.x < total_edges) {
			src = srcsrc[threadIdx.x];
			dst = edges[gather_offsets[threadIdx.x]];
			if ((states[dst] & 6) == 0 && sh_ranges[src - blockDim.x * blockIdx.x - 1] == ranges[dst]) {
				setFVis(&states[dst]);
				sh_end = false;
			}
		}
		done += BLKSIZE;
		total_edges -= BLKSIZE;
	}
	//__syncthreads();
	if (threadIdx.x == 0 && sh_end == false)
		*end = false;
	__syncthreads();
}

unsigned bwd_reach_b(unsigned *d_nodes, unsigned countOfNodes, unsigned *d_edges, unsigned countOfEdges, unsigned *d_ranges, unsigned char *d_states, alg_opt opt) {
	bool volatile *d_end;
	bool end = false;
	unsigned depth = 0;
	dim3 dimGrid;
	setGridDimension(&dimGrid, countOfNodes, opt.block_size);
	gpuErrchk( cudaMalloc((void**)&d_end, sizeof(*d_end)) );
#ifdef DEBUG
	timeval timer;
#endif
	while(!end) {
		gpuErrchk( cudaMemset((void *)d_end, true, sizeof(*d_end)) ); // prerobit <<< >>> aj na mnoho vrcholov
#ifdef DEBUG
		startTimer(&timer);
#endif
		bwd_step<<<dimGrid, opt.block_size>>>(d_nodes, d_edges, d_ranges, d_states, countOfNodes, d_end);
		gpuErrchk( cudaMemcpy(&end, (void *)d_end, sizeof(*d_end), cudaMemcpyDeviceToHost) );
#ifdef DEBUG
		printf("bwd iteration %d time:%f ms\n", depth + 1, stopTimer(&timer) / 1000.0);
#endif
		depth++;
	}
	gpuErrchk( cudaFree((void *) d_end) );
	return depth;
}

unsigned fwd_reach_b(unsigned *d_nodes, unsigned countOfNodes, unsigned *d_edges, unsigned countOfEdges, unsigned *d_ranges, unsigned char *d_states, alg_opt opt) {
	bool volatile *d_end;
	bool end = false;
	unsigned depth = 0;
	dim3 dimGrid;
	setGridDimension(&dimGrid, countOfNodes, opt.block_size);
	gpuErrchk( cudaMalloc((void**)&d_end, sizeof(*d_end)) );
	while(!end) {
		gpuErrchk( cudaMemset((void *)d_end, true, sizeof(*d_end)) ); 
#ifdef VERIFY
		fwd_step<<<dimGrid, opt.block_size>>>(d_nodes, d_edges, d_ranges, d_states, countOfNodes, d_end, scc_root);
#else
		fwd_step<<<dimGrid, opt.block_size>>>(d_nodes, d_edges, d_ranges, d_states, countOfNodes, d_end);
#endif
		gpuErrchk( cudaMemcpy(&end, (void *)d_end, sizeof(*d_end), cudaMemcpyDeviceToHost) );		
		depth++;
	}
	gpuErrchk( cudaFree((void *) d_end) );
	return depth;
}

static __global__ void bwd_step1(unsigned *nodes, unsigned *edges, unsigned char *states, unsigned countOfNodes, bool volatile *end) {
	unsigned id = blockDim.x * blockIdx.x + threadIdx.x + 1;
	typedef cub::BlockScan<unsigned, BLKSIZE> BlockScan;
	__shared__ BlockScan::TempStorage temp_storage;
	__shared__ unsigned gather_offsets[BLKSIZE];
	__shared__ bool sh_end;

	gather_offsets[threadIdx.x] = 0;
	if (threadIdx.x == 0)
		sh_end = true;
	__syncthreads();

	unsigned neighboroffset = 0;
	unsigned neighborsize = 0;
	unsigned scratch_offset = 0;
	unsigned total_edges = 0;

	if(id <= countOfNodes && (states[id] & 37) == 1) {
		setBExt(&states[id]);
		neighboroffset = nodes[id];
		neighborsize = nodes[id + 1] - neighboroffset;
	}
	__syncthreads();
	BlockScan(temp_storage).ExclusiveSum(neighborsize, scratch_offset, total_edges);
	unsigned done = 0;
	unsigned neighborsdone = 0;
	while ((int)total_edges > 0) {
		__syncthreads();
		unsigned i;
		unsigned index;
		for (i = 0; neighborsdone + i < neighborsize && (index = scratch_offset + i - done) < BLKSIZE; i++) {
			//gather_offsets[scratch_offset + i - done] = neighboroffset + neighborsdone + i;
			//srcsrc[scratch_offset + i - done] = id;
			gather_offsets[index] = neighboroffset + neighborsdone + i;
			//srcsrc[index] = id;
		}
		neighborsdone += i;
		scratch_offset += i;
		__syncthreads();
		//unsigned src;
		unsigned dst;
		if (threadIdx.x < total_edges) {
			dst = edges[gather_offsets[threadIdx.x]];
			if ((states[dst] & 5) == 0) {
				setBVis(&states[dst]);
				sh_end = false;
			}
		}
		done += BLKSIZE;
		total_edges -= BLKSIZE;
	}
	__syncthreads();
	if (threadIdx.x == 0 && sh_end == false)
		*end = false;
	__syncthreads();
}


static __global__ void fwd_step1(unsigned *nodes, unsigned *edges, unsigned char *states, unsigned countOfNodes, int *frontier_size) {
	unsigned id = blockDim.x * blockIdx.x + threadIdx.x + 1;
	typedef cub::BlockScan<unsigned, BLKSIZE> BlockScan;
	__shared__ BlockScan::TempStorage temp_storage;
	__shared__ unsigned gather_offsets[BLKSIZE];
	//__shared__ bool sh_end;

	gather_offsets[threadIdx.x] = 0;
	//if (threadIdx.x == 0) sh_end = true;
	__syncthreads();

	unsigned neighboroffset = 0;
	unsigned neighborsize = 0;
	unsigned scratch_offset = 0;
	unsigned total_edges = 0;

	//if(id <= countOfNodes && !isElim(&my_state) && !isFExt(&my_state) && isFVis(&my_state)) {
	if(id <= countOfNodes && (states[id] & 70) == 2) {
		setFExt(&states[id]);
		neighboroffset = nodes[id];
		neighborsize = nodes[id + 1] - neighboroffset;
	}
	__syncthreads();

	BlockScan(temp_storage).ExclusiveSum(neighborsize, scratch_offset, total_edges);

	unsigned done = 0;
	unsigned neighborsdone = 0;

	while ((int)total_edges > 0) {
		__syncthreads();
		unsigned i;
		unsigned index;
		for (i = 0; neighborsdone + i < neighborsize && (index = scratch_offset + i - done) < BLKSIZE; i++) {
			//gather_offsets[scratch_offset + i - done] = neighboroffset + neighborsdone + i;
			//srcsrc[scratch_offset + i - done] = id;
			gather_offsets[index] = neighboroffset + neighborsdone + i;
		}
		neighborsdone += i;
		scratch_offset += i;
		__syncthreads();
		unsigned dst;
		if (threadIdx.x < total_edges) {
			dst = edges[gather_offsets[threadIdx.x]];
			if ((states[dst] & 6) == 0) {
				setFVis(&states[dst]);
				atomicAdd(frontier_size,1);
				//sh_end = false;
			}
		}
		done += BLKSIZE;
		total_edges -= BLKSIZE;
	}
	//__syncthreads();
	//if (threadIdx.x == 0 && sh_end == false)
	//	*end = false;
	//__syncthreads();
}

unsigned bwd_reach1_b(unsigned *d_nodes, unsigned countOfNodes, unsigned *d_edges, unsigned countOfEdges, unsigned char *d_states, alg_opt opt) {
	bool volatile *d_end;
	bool end = false;
	unsigned depth = 0;
	dim3 dimGrid;
	setGridDimension(&dimGrid, countOfNodes, opt.block_size);
	gpuErrchk( cudaMalloc((void**)&d_end, sizeof(*d_end)) );
	while(!end) {
		gpuErrchk( cudaMemset((void *)d_end, true, sizeof(*d_end)) ); // prerobit <<< >>> aj na mnoho vrcholov
		bwd_step1<<<dimGrid, opt.block_size>>>(d_nodes, d_edges, d_states, countOfNodes, d_end);
		gpuErrchk( cudaMemcpy(&end, (void *)d_end, sizeof(*d_end), cudaMemcpyDeviceToHost) );
		depth++;
	}
	gpuErrchk( cudaFree((void *) d_end) );
	return depth;
}

static __global__ void fwd_step1_linear(unsigned *nodes, unsigned *edges, unsigned char *states, unsigned countOfNodes, int *scout_count, int *degree, Worklist2 inwl, Worklist2 outwl) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned nn; 
	if (!inwl.pop_id(id, nn))
		return;
	unsigned my_start = nodes[nn];
	unsigned my_end = nodes[nn + 1]; 
	//unsigned my_range = ranges[nn];
	unsigned dst;
	for (unsigned ii = my_start; ii < my_end; ++ii) {
		dst = edges[ii];
		if ((states[dst] & 6) == 0) {// && my_range == ranges[dst]) {
			setFVis(&states[dst]);
			//scc_root[dst] = scc_root[nn];
			outwl.push(dst);
			atomicAdd(scout_count, degree[dst]);
		}
	}   
}

__global__ void insert(int source, Worklist2 queue) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id == 0) queue.push(source);
	return;
}

//hybrid BFS only for phase-1 (to identify the single giant SCC)
unsigned fwd_reach1_b(unsigned *d_nodes, unsigned countOfNodes, unsigned *d_edges, unsigned countOfEdges, unsigned char *d_states, alg_opt opt, int *h_degree, int source) {
	//bool volatile *d_end;
	//bool end = false;
	unsigned depth = 0;
	dim3 dimGrid;
	//gpuErrchk( cudaMalloc((void**)&d_end, sizeof(*d_end)) );

	Worklist2 queue1(countOfEdges), queue2(countOfEdges);
	Worklist2 *in_frontier = &queue1, *out_frontier = &queue2;
	int alpha = 15, beta = 18;
	int nitems = 1;
	int edges_to_check = countOfNodes;
	int scout_count = h_degree[source];
	int *d_scout_count, *d_frontier_size;
	int *d_degree;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_scout_count, sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_frontier_size, sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_degree, countOfNodes * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_degree, h_degree, countOfNodes * sizeof(int), cudaMemcpyHostToDevice));
	insert<<<1, 32>>>(source, *in_frontier);

	do {
		if(scout_count > edges_to_check / alpha) {
			int awake_count, old_awake_count;
			awake_count = nitems;
			do {
				depth++;
				old_awake_count = awake_count;
				setGridDimension(&dimGrid, countOfNodes, opt.block_size);
				fwd_step1<<<dimGrid, opt.block_size>>>(d_nodes, d_edges, d_states, countOfNodes, d_frontier_size);
				gpuErrchk( cudaMemcpy(&awake_count, (void *)d_frontier_size, sizeof(int), cudaMemcpyDeviceToHost) );
			} while((awake_count >= old_awake_count) || (awake_count > countOfNodes / beta));
		} else{ 
			depth++;
			edges_to_check -= scout_count;
			setGridDimension(&dimGrid, nitems, opt.block_size);
			CUDA_SAFE_CALL(cudaMemcpy(d_scout_count, &zero, sizeof(int), cudaMemcpyHostToDevice));
			fwd_step1_linear<<<dimGrid, BLKSIZE>>>(d_nodes, d_edges, d_states, countOfNodes, d_scout_count, d_degree, *in_frontier, *out_frontier);
			CUDA_SAFE_CALL(cudaMemcpy(&scout_count, d_scout_count, sizeof(int), cudaMemcpyDeviceToHost));
			//nitems = out_frontier->nitems();
			nitems = out_frontier->getSize();
			Worklist2 *tmp = in_frontier;
			in_frontier = out_frontier;
			out_frontier = tmp;
			//out_frontier->reset();
			out_frontier->clearHost();;
		}
	} while (nitems > 0);
	gpuErrchk(cudaFree(d_scout_count));
	//gpuErrchk( cudaFree((void *) d_end) );
	return depth;
}
/*
unsigned fwd_reach1_t(unsigned *d_nodes, unsigned countOfNodes, unsigned *d_edges, unsigned countOfEdges, unsigned char *d_states, alg_opt opt) {
	bool volatile *d_end;
	bool end = false;
	unsigned depth = 0;
	dim3 dimGrid;
	setGridDimension(&dimGrid, countOfNodes, opt.block_size);
	gpuErrchk( cudaMalloc((void**)&d_end, sizeof(*d_end)) );
	while(!end) {
		gpuErrchk( cudaMemset((void *)d_end, true, sizeof(*d_end)) ); 
		fwd_step1<<<dimGrid, opt.block_size>>>(d_nodes, d_edges, d_states, countOfNodes, d_end);
		gpuErrchk( cudaMemcpy(&end, (void *)d_end, sizeof(*d_end), cudaMemcpyDeviceToHost) );		
		depth++;
	}
	gpuErrchk( cudaFree((void *) d_end) );
	return depth;
}
*/
