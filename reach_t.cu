#include <stdint.h>

#include "scc.h"
#include "reach_t.h"
#include "GPUerrchk.h"
#include "timing.h"
#include "cub/cub.cuh"

//#include <assert.h>

//#define DEBUG
//#define EXPAND_BY_CTA

#define BLKSIZE 256


static __global__ void bwd_step(unsigned *nodes, unsigned *edges, unsigned *ranges, unsigned char *states, unsigned countOfNodes, bool volatile *end) {

	unsigned id = blockDim.x * blockIdx.x + threadIdx.x + 1;
	//if (id > countOfNodes) return;

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

	//if(id <= countOfNodes && !isElim(&my_state) && !isBExt(&my_state) && isBVis(&my_state)) {
	if(id <= countOfNodes && (states[id] & 37) == 1) {
		setBExt(&states[id]);
		neighboroffset = nodes[id];
		neighborsize = nodes[id + 1] - neighboroffset;
	}
	__syncthreads();

//#ifdef EXPAND_BY_CTA
#if 0
	__shared__ int owner;
	__shared__ unsigned nn;
	
	if (threadIdx.x == 1)
		owner = -1;
	__syncthreads();

	while (true) {
		if (neighborsize >= BLKSIZE) 
			owner = threadIdx.x;
		__syncthreads;

		if (owner == -1) 
			break;

		if (owner == threadIdx.x) {
			nn = id;
			neighborsize = 0;
			owner = -1;
		}
		__syncthreads();

	#if 0
		unsigned offset = nodes[nn];
		unsigned size = nodes[nn + 1] - offset;
		unsigned range = sh_ranges[nn - blockDim.x * blockIdx.x - 1];

		unsigned num = ((size + blockDim.x - 1) / blockDim.x) * blockDim.x;
		unsigned dst;

		for (unsigned i = threadIdx.x; i < num; i += blockDim.x) {
			if (i < size) {
				dst = edges[offset + i];
				if ((states[dst] & 5) == 0 && ranges[dst] == range) {
					setBVis(&states[dst]);
					sh_end = false;
				}
			}
		}
	#endif
	}
	__syncthreads();
#endif

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
				//assert(src >= blockDim.x * blockIdx.x + 1 && src <= blockDim.x * blockIdx.x + blockDim.x);
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


#ifdef VERIFY					
static __global__ void fwd_step(unsigned *nodes, unsigned *edges, unsigned *ranges, unsigned char *states, unsigned countOfNodes, bool volatile *end, unsigned *scc_root) {
#else
static __global__ void fwd_step(unsigned *nodes, unsigned *edges, unsigned *ranges, unsigned char *states, unsigned countOfNodes, bool volatile *end) {
#endif

	unsigned id = blockDim.x * blockIdx.x + threadIdx.x + 1;
	//if (id > countOfNodes) return;

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

	//if(id <= countOfNodes && !isElim(&my_state) && !isFExt(&my_state) && isFVis(&my_state)) {
	if(id <= countOfNodes && (states[id] & 70) == 2) {
		setFExt(&states[id]);
		neighboroffset = nodes[id];
		neighborsize = nodes[id + 1] - neighboroffset;
	}
	__syncthreads();

//#ifdef EXPAND_BY_CTA
#if 0
	__shared__ int owner;
	__shared__ unsigned nn;
	
	if (threadIdx.x == 1)
		owner = -1;
	__syncthreads();

	while (true) {
		if (neighborsize >= BLKSIZE) 
			owner = threadIdx.x;
		__syncthreads;

		if (owner == -1) 
			break;

		if (owner == threadIdx.x) {
			nn = id;
			neighborsize = 0;
			owner = -1;
		}
		__syncthreads();

		unsigned offset = nodes[nn];
		unsigned size = nodes[nn + 1] - offset;
		unsigned range = sh_ranges[nn - blockDim.x * blockIdx.x - 1];

		unsigned num = ((size + blockDim.x - 1) / blockDim.x) * blockDim.x;
		unsigned dst;

		for (unsigned i = threadIdx.x; i < num; i += blockDim.x) {
			if (i < size) {
				dst = edges[offset + i];
				if ((states[dst] & 6) == 0 && ranges[dst] == range) {
				#ifdef VERIFY
					scc_root[dst] = scc_root[nn];
				#endif
					setFVis(&states[dst]);
					sh_end = false;
				}
			}
		}
	}
	__syncthreads();
#endif

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
				//assert(src >= blockDim.x * blockIdx.x + 1 && src <= blockDim.x * blockIdx.x + blockDim.x);
				if ((states[dst] & 6) == 0 && sh_ranges[src - blockDim.x * blockIdx.x - 1] == ranges[dst]) {
				#ifdef VERIFY
					scc_root[dst] = scc_root[src];
				#endif
					setFVis(&states[dst]);
					sh_end = false;
				}
			//}
		}
		done += BLKSIZE;
		total_edges -= BLKSIZE;
	}
	//__syncthreads();
	if (threadIdx.x == 0 && sh_end == false)
		*end = false;
	__syncthreads();
}
					
unsigned bwd_reach_t(unsigned *d_nodes, unsigned countOfNodes, unsigned *d_edges, unsigned countOfEdges, unsigned *d_ranges, unsigned char *d_states, alg_opt opt) {
				
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

#ifdef VERIFY
unsigned fwd_reach_t(unsigned *d_nodes, unsigned countOfNodes, unsigned *d_edges, unsigned countOfEdges, unsigned *d_ranges, unsigned char *d_states, alg_opt opt, unsigned *scc_root) {
#else
unsigned fwd_reach_t(unsigned *d_nodes, unsigned countOfNodes, unsigned *d_edges, unsigned countOfEdges, unsigned *d_ranges, unsigned char *d_states, alg_opt opt) {
#endif
				
	bool volatile *d_end;
	bool end = false;
	
	unsigned depth = 0;
	
	dim3 dimGrid;

#ifdef DEBUG
	timeval timer;
#endif
	
	setGridDimension(&dimGrid, countOfNodes, opt.block_size);
	
	gpuErrchk( cudaMalloc((void**)&d_end, sizeof(*d_end)) );
	
	while(!end) {
		gpuErrchk( cudaMemset((void *)d_end, true, sizeof(*d_end)) ); 
	#ifdef DEBUG
		startTimer(&timer);
	#endif
	#ifdef VERIFY
		fwd_step<<<dimGrid, opt.block_size>>>(d_nodes, d_edges, d_ranges, d_states, countOfNodes, d_end, scc_root);
	#else
		fwd_step<<<dimGrid, opt.block_size>>>(d_nodes, d_edges, d_ranges, d_states, countOfNodes, d_end);
	#endif
		gpuErrchk( cudaMemcpy(&end, (void *)d_end, sizeof(*d_end), cudaMemcpyDeviceToHost) );		
	#ifdef DEBUG
		printf("fwd iteration %d time:%f ms\n", depth + 1, stopTimer(&timer) / 1000.0);
	#endif
		depth++;
	}
	
	gpuErrchk( cudaFree((void *) d_end) );
	
	return depth;
}

//static __global__ void bwd_step(unsigned *nodes, unsigned *edges, unsigned *ranges, unsigned char *states, unsigned countOfNodes, bool volatile *end) {
static __global__ void bwd_step1(unsigned *nodes, unsigned *edges, unsigned char *states, unsigned countOfNodes, bool volatile *end) {

	unsigned id = blockDim.x * blockIdx.x + threadIdx.x + 1;
	//if (id > countOfNodes) return;

	typedef cub::BlockScan<unsigned, BLKSIZE> BlockScan;
	
	__shared__ BlockScan::TempStorage temp_storage;
	__shared__ unsigned gather_offsets[BLKSIZE];
	//__shared__ unsigned srcsrc[BLKSIZE];
	//__shared__ unsigned sh_ranges[BLKSIZE];
	__shared__ bool sh_end;

	gather_offsets[threadIdx.x] = 0;
	//if (id <= countOfNodes)
		//sh_ranges[threadIdx.x] = ranges[id];
	//__syncthreads();
	if (threadIdx.x == 0)
		sh_end = true;
	__syncthreads();

	unsigned neighboroffset = 0;
	unsigned neighborsize = 0;
	unsigned scratch_offset = 0;
	unsigned total_edges = 0;

	//if(id <= countOfNodes && !isElim(&my_state) && !isBExt(&my_state) && isBVis(&my_state)) {
	if(id <= countOfNodes && (states[id] & 37) == 1) {
		setBExt(&states[id]);
		neighboroffset = nodes[id];
		neighborsize = nodes[id + 1] - neighboroffset;
	}
	__syncthreads();

//#ifdef EXPAND_BY_CTA
#if 0
	__shared__ int owner;
	__shared__ unsigned nn;
	
	if (threadIdx.x == 1)
		owner = -1;
	__syncthreads();

	while (true) {
		if (neighborsize >= BLKSIZE) 
			owner = threadIdx.x;
		__syncthreads;

		if (owner == -1) 
			break;

		if (owner == threadIdx.x) {
			nn = id;
			neighborsize = 0;
			owner = -1;
		}
		__syncthreads();

	#if 0
		unsigned offset = nodes[nn];
		unsigned size = nodes[nn + 1] - offset;
		unsigned range = sh_ranges[nn - blockDim.x * blockIdx.x - 1];

		unsigned num = ((size + blockDim.x - 1) / blockDim.x) * blockDim.x;
		unsigned dst;

		for (unsigned i = threadIdx.x; i < num; i += blockDim.x) {
			if (i < size) {
				dst = edges[offset + i];
				if ((states[dst] & 5) == 0 && ranges[dst] == range) {
					setBVis(&states[dst]);
					sh_end = false;
				}
			}
		}
	#endif
	}
	__syncthreads();
#endif

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
			//src = srcsrc[threadIdx.x];
			dst = edges[gather_offsets[threadIdx.x]];

			//if (src != dst) {
				//assert(src >= blockDim.x * blockIdx.x + 1 && src <= blockDim.x * blockIdx.x + blockDim.x);
				//if ((states[dst] & 5) == 0 && sh_ranges[src - blockDim.x * blockIdx.x - 1] == ranges[dst]) {
				if ((states[dst] & 5) == 0) {
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


#ifdef VERIFY					
//static __global__ void fwd_step(unsigned *nodes, unsigned *edges, unsigned *ranges, unsigned char *states, unsigned countOfNodes, bool volatile *end, unsigned *scc_root) {
static __global__ void fwd_step1(unsigned *nodes, unsigned *edges, unsigned char *states, unsigned countOfNodes, bool volatile *end, unsigned *scc_root) {
#else
//static __global__ void fwd_step(unsigned *nodes, unsigned *edges, unsigned *ranges, unsigned char *states, unsigned countOfNodes, bool volatile *end) {
static __global__ void fwd_step1(unsigned *nodes, unsigned *edges, unsigned char *states, unsigned countOfNodes, bool volatile *end) {
#endif

	unsigned id = blockDim.x * blockIdx.x + threadIdx.x + 1;
	//if (id > countOfNodes) return;

	typedef cub::BlockScan<unsigned, BLKSIZE> BlockScan;
	
	__shared__ BlockScan::TempStorage temp_storage;
	__shared__ unsigned gather_offsets[BLKSIZE];
	#ifdef VERIFY
	__shared__ unsigned srcsrc[BLKSIZE];
	#endif
	//__shared__ unsigned sh_ranges[BLKSIZE];
	__shared__ bool sh_end;

	gather_offsets[threadIdx.x] = 0;
	//if (id <= countOfNodes)
		//sh_ranges[threadIdx.x] = ranges[id];
	//__syncthreads();
	if (threadIdx.x == 0)
		sh_end = true;
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

//#ifdef EXPAND_BY_CTA
#if 0
	__shared__ int owner;
	__shared__ unsigned nn;
	
	if (threadIdx.x == 1)
		owner = -1;
	__syncthreads();

	while (true) {
		if (neighborsize >= BLKSIZE) 
			owner = threadIdx.x;
		__syncthreads;

		if (owner == -1) 
			break;

		if (owner == threadIdx.x) {
			nn = id;
			neighborsize = 0;
			owner = -1;
		}
		__syncthreads();

		unsigned offset = nodes[nn];
		unsigned size = nodes[nn + 1] - offset;
		unsigned range = sh_ranges[nn - blockDim.x * blockIdx.x - 1];

		unsigned num = ((size + blockDim.x - 1) / blockDim.x) * blockDim.x;
		unsigned dst;

		for (unsigned i = threadIdx.x; i < num; i += blockDim.x) {
			if (i < size) {
				dst = edges[offset + i];
				if ((states[dst] & 6) == 0 && ranges[dst] == range) {
				#ifdef VERIFY
					scc_root[dst] = scc_root[nn];
				#endif
					setFVis(&states[dst]);
					sh_end = false;
				}
			}
		}
	}
	__syncthreads();
#endif

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
			#ifdef VERIFY
			srcsrc[index] = id;
			#endif
		}
		neighborsdone += i;
		scratch_offset += i;
		__syncthreads();

		#ifdef VERIFY
		unsigned src;
		#endif
		unsigned dst;

		if (threadIdx.x < total_edges) {
			#ifdef VERIFY
			src = srcsrc[threadIdx.x];
			#endif
			dst = edges[gather_offsets[threadIdx.x]];

			//if (src != dst) {
				//assert(src >= blockDim.x * blockIdx.x + 1 && src <= blockDim.x * blockIdx.x + blockDim.x);
				//if ((states[dst] & 6) == 0 && sh_ranges[src - blockDim.x * blockIdx.x - 1] == ranges[dst]) {
				if ((states[dst] & 6) == 0) {
				#ifdef VERIFY
					scc_root[dst] = scc_root[src];
				#endif
					setFVis(&states[dst]);
					sh_end = false;
				}
			//}
		}
		done += BLKSIZE;
		total_edges -= BLKSIZE;
	}
	//__syncthreads();
	if (threadIdx.x == 0 && sh_end == false)
		*end = false;
	__syncthreads();
}
					
//unsigned bwd_reach_t(unsigned *d_nodes, unsigned countOfNodes, unsigned *d_edges, unsigned countOfEdges, unsigned *d_ranges, unsigned char *d_states, alg_opt opt) {
unsigned bwd_reach1_t(unsigned *d_nodes, unsigned countOfNodes, unsigned *d_edges, unsigned countOfEdges, unsigned char *d_states, alg_opt opt) {
				
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
		//bwd_step<<<dimGrid, opt.block_size>>>(d_nodes, d_edges, d_ranges, d_states, countOfNodes, d_end);
		bwd_step1<<<dimGrid, opt.block_size>>>(d_nodes, d_edges, d_states, countOfNodes, d_end);
		gpuErrchk( cudaMemcpy(&end, (void *)d_end, sizeof(*d_end), cudaMemcpyDeviceToHost) );
	#ifdef DEBUG
		printf("bwd iteration %d time:%f ms\n", depth + 1, stopTimer(&timer) / 1000.0);
	#endif
		depth++;
	}

	gpuErrchk( cudaFree((void *) d_end) );
	
	return depth;
}

#ifdef VERIFY
//unsigned fwd_reach_t(unsigned *d_nodes, unsigned countOfNodes, unsigned *d_edges, unsigned countOfEdges, unsigned *d_ranges, unsigned char *d_states, alg_opt opt, unsigned *scc_root) {
unsigned fwd_reach1_t(unsigned *d_nodes, unsigned countOfNodes, unsigned *d_edges, unsigned countOfEdges, unsigned char *d_states, alg_opt opt, unsigned *scc_root) {
#else
//unsigned fwd_reach_t(unsigned *d_nodes, unsigned countOfNodes, unsigned *d_edges, unsigned countOfEdges, unsigned *d_ranges, unsigned char *d_states, alg_opt opt) {
unsigned fwd_reach1_t(unsigned *d_nodes, unsigned countOfNodes, unsigned *d_edges, unsigned countOfEdges, unsigned char *d_states, alg_opt opt) {
#endif
				
	bool volatile *d_end;
	bool end = false;
	
	unsigned depth = 0;
	
	dim3 dimGrid;

#ifdef DEBUG
	timeval timer;
#endif
	
	setGridDimension(&dimGrid, countOfNodes, opt.block_size);
	
	gpuErrchk( cudaMalloc((void**)&d_end, sizeof(*d_end)) );
	
	while(!end) {
		gpuErrchk( cudaMemset((void *)d_end, true, sizeof(*d_end)) ); 
	#ifdef DEBUG
		startTimer(&timer);
	#endif
	#ifdef VERIFY
		//fwd_step<<<dimGrid, opt.block_size>>>(d_nodes, d_edges, d_ranges, d_states, countOfNodes, d_end, scc_root);
		fwd_step1<<<dimGrid, opt.block_size>>>(d_nodes, d_edges, d_states, countOfNodes, d_end, scc_root);
	#else
		//fwd_step<<<dimGrid, opt.block_size>>>(d_nodes, d_edges, d_ranges, d_states, countOfNodes, d_end);
		fwd_step1<<<dimGrid, opt.block_size>>>(d_nodes, d_edges, d_states, countOfNodes, d_end);
	#endif
		gpuErrchk( cudaMemcpy(&end, (void *)d_end, sizeof(*d_end), cudaMemcpyDeviceToHost) );		
	#ifdef DEBUG
		printf("fwd iteration %d time:%f ms\n", depth + 1, stopTimer(&timer) / 1000.0);
	#endif
		depth++;
	}
	
	gpuErrchk( cudaFree((void *) d_end) );
	
	return depth;
}
