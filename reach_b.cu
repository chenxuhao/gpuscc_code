#include <stdint.h>

#include "scc.h"
#include "reach_b.h"
#include "GPUerrchk.h"
#include "timing.h"
#include "cub/cub.cuh"
#include "worklistc.h"
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

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

///*
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
//*/
/*
static __global__ void fwd_step1(unsigned *nodes, unsigned *edges, unsigned char *states, unsigned countOfNodes, volatile bool *end) {
	unsigned id = blockDim.x * blockIdx.x + threadIdx.x + 1;
	typedef cub::BlockScan<unsigned, BLKSIZE> BlockScan;
	__shared__ BlockScan::TempStorage temp_storage;
	__shared__ unsigned gather_offsets[BLKSIZE];
	__shared__ bool sh_end;

	gather_offsets[threadIdx.x] = 0;
	if (threadIdx.x == 0) sh_end = true;
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
				//atomicAdd(frontier_size,1);
				sh_end = false;
			}
		}
		done += BLKSIZE;
		total_edges -= BLKSIZE;
	}
	//__syncthreads();
	if (threadIdx.x == 0 && sh_end == false)
		*end = false;
	//__syncthreads();
}
//*/

//top-down kernel
static __global__ void fwd_step1_td(unsigned *nodes, unsigned *edges, unsigned char *states, unsigned countOfNodes, int *scout_count, int *degree, Worklist2 inwl, Worklist2 outwl) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned nn; 
	if (!inwl.pop_id(id, nn))
		return;
	setFExt(&states[nn]);
	unsigned my_start = nodes[nn];
	unsigned my_end = nodes[nn + 1]; 
	unsigned dst;
	for (unsigned ii = my_start; ii < my_end; ++ii) {
		dst = edges[ii];
		if ((states[dst] & 6) == 0) {
			setFVis(&states[dst]);
			outwl.push(dst);
			atomicAdd(scout_count, degree[dst]);
		}
	}
}

static __global__ void bwd_step1_td(unsigned *nodes, unsigned *edges, unsigned char *states, unsigned countOfNodes, int *scout_count, int *degree, Worklist2 inwl, Worklist2 outwl) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned nn; 
	if (!inwl.pop_id(id, nn))
		return;
	setFExt(&states[nn]);
	unsigned my_start = nodes[nn];
	unsigned my_end = nodes[nn + 1]; 
	unsigned dst;
	for (unsigned ii = my_start; ii < my_end; ++ii) {
		dst = edges[ii];
		if ((states[dst] & 5) == 0) {
			setBVis(&states[dst]);
			outwl.push(dst);
			atomicAdd(scout_count, degree[dst]);
		}
	}
}

static __global__ void fwd_step1_topo(unsigned *nodes, unsigned *edges, unsigned char *states, unsigned countOfNodes, int *scout_count, int *degree, volatile bool *end) {
	unsigned id = blockDim.x * blockIdx.x + threadIdx.x + 1;
	if (id > countOfNodes) return;
	if((states[id] & 70) == 2) { // if it is visited but not expanded
		setFExt(&states[id]);
		unsigned my_start = nodes[id];
		unsigned my_end = nodes[id + 1];
		for(unsigned i = my_start; i < my_end; i++) {
			unsigned dst = edges[i];
			if((states[dst] & 6) == 0) { // if it is not visited
				setFVis(&states[dst]);
				scout_count[dst] = degree[dst];
				//*end = false;
			}
		}
	}
}

static __global__ void bwd_step1_topo(unsigned *nodes, unsigned *edges, unsigned char *states, unsigned countOfNodes, int *scout_count, int *degree, volatile bool *end) {
	unsigned id = blockDim.x * blockIdx.x + threadIdx.x + 1;
	if (id > countOfNodes) return;
	if((states[id] & 37) == 1) { // if it is visited but not expanded
		setBExt(&states[id]);
		unsigned my_start = nodes[id];
		unsigned my_end = nodes[id + 1];
		for(unsigned i = my_start; i < my_end; i++) {
			unsigned dst = edges[i];
			if((states[dst] & 5) == 0) { // if it is not visited
				setBVis(&states[dst]);
				scout_count[dst] = degree[dst];
				//*end = false;
			}
		}
	}
}

//bottom-up kernel
static __global__ void fwd_step1_bu(unsigned *nodes, unsigned *edges, unsigned char *states, unsigned countOfNodes, int *front, int *next, int *frontier_size) {
	unsigned id = blockDim.x * blockIdx.x + threadIdx.x + 1;
	if(id > countOfNodes) return;
	if((states[id] & 6) == 0) { // if it is not visited
		unsigned my_start = nodes[id];
		unsigned my_end = nodes[id + 1];
		for(unsigned i = my_start; i < my_end; i++) {
			unsigned dst = edges[i];
			if(front[dst] == 1) { // if dst is in the frontier (visited but not extended)
				//printf("src=%d, dst=%d\n", id, dst);
				setFVis(&states[id]);
				next[id] = 1;
				//atomicAdd(frontier_size,1);
			}
		}
	}
}

static __global__ void bwd_step1_bu(unsigned *nodes, unsigned *edges, unsigned char *states, unsigned countOfNodes, int *front, int *next, int *frontier_size) {
	unsigned id = blockDim.x * blockIdx.x + threadIdx.x + 1;
	if(id > countOfNodes) return;
	if((states[id] & 5) == 0) { // if it is not visited
		unsigned my_start = nodes[id];
		unsigned my_end = nodes[id + 1];
		for(unsigned i = my_start; i < my_end; i++) {
			unsigned dst = edges[i];
			if(front[dst] == 1) { // if dst is in the frontier (visited but not extended)
				//printf("src=%d, dst=%d\n", id, dst);
				setBVis(&states[id]);
				next[id] = 1;
				//atomicAdd(frontier_size,1);
			}
		}
	}
}

__global__ void QueueToBitmap(int countOfNodes, Worklist2 queue, int *bm) {
	unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < countOfNodes) {
		unsigned src;
		if(queue.pop_id(tid, src)) {
			bm[src] = 1;
		} else {
			bm[src] = 0;
		}
	}
}

__global__ void BitmapToQueue(int countOfNodes, int *bm, Worklist2 queue) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x + 1;
	if(id > countOfNodes) return;
	if(bm[id]) queue.push(id);
}

__global__ void FindFwdQueue(int countOfNodes, unsigned char *states, int *bm) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x + 1;
	if(id > countOfNodes) return;
	if((states[id] & 70) == 2) bm[id] = 1;
	else bm[id] = 0;
}

__global__ void FindBwdQueue(int countOfNodes, unsigned char *states, int *bm) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x + 1;
	if(id > countOfNodes) return;
	if((states[id] & 37) == 1) bm[id] = 1;
	else bm[id] = 0;
}

__global__ void insert(int source, Worklist2 queue) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id == 0) queue.push(source);
	return;
}
///*
#define TD
//hybrid BFS only for phase-1 (to identify the single giant SCC)
unsigned fwd_reach1_b(unsigned *d_nodes, unsigned *d_nodesT, unsigned countOfNodes, unsigned *d_edges, unsigned *d_edgesT, unsigned countOfEdges, unsigned char *d_states, alg_opt opt, int *h_degree, int source) {
	dim3 dimGrid;
	unsigned depth = 0;
#ifdef TD
	Worklist2 queue1(countOfEdges), queue2(countOfEdges);
	Worklist2 *in_frontier = &queue1, *out_frontier = &queue2;
	insert<<<1, 32>>>(source, *in_frontier);
#else
	//bool volatile *d_end;
	//bool end = false;
	//gpuErrchk( cudaMalloc((void**)&d_end, sizeof(*d_end)) );
#endif
	int *front, *next;
	int zero = 0;
	int alpha = 15, beta = 18;
	int nitems = 1;
	int edges_to_check = countOfEdges;
	int scout_count = h_degree[source];
	int *d_scout_count, *d_frontier_size;
	int *d_degree;
#ifdef TD
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_scout_count, sizeof(int)));
#else
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_scout_count, (countOfNodes+1) * sizeof(int)));
#endif
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_frontier_size, sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_degree, (countOfNodes+1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_degree, h_degree, (countOfNodes+1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMalloc((void **)&front, (countOfNodes+1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&next, (countOfNodes+1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemset(front, 0, (countOfNodes+1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemset(next, 0, (countOfNodes+1) * sizeof(int)));

	do {
		if(scout_count > edges_to_check / alpha) {
		//if(0) {
			int awake_count, old_awake_count;
#ifdef TD
			QueueToBitmap<<<((countOfNodes-1)/512+1), 512>>>(countOfNodes, *in_frontier, front);
#else
			FindFwdQueue<<<((countOfNodes-1)/512+1), 512>>>(countOfNodes, d_states, front);
#endif
			awake_count = nitems;
			do {
				depth++;
				old_awake_count = awake_count;
				setGridDimension(&dimGrid, countOfNodes, opt.block_size);
				//gpuErrchk(cudaMemcpy(d_frontier_size, &zero, sizeof(int), cudaMemcpyHostToDevice));
				fwd_step1_bu<<<dimGrid, opt.block_size>>>(d_nodesT, d_edgesT, d_states, countOfNodes, front, next, d_frontier_size);
				//gpuErrchk( cudaMemcpy(&awake_count, (void *)d_frontier_size, sizeof(int), cudaMemcpyDeviceToHost) );
				awake_count = thrust::reduce(thrust::device, next + 1, next + countOfNodes + 1, 0, thrust::plus<int>());
				//printf("BU: iteration=%d, num_frontier=%d\n", depth, awake_count);
				int *temp = front;
				front = next;
				next = temp;
				thrust::fill(thrust::device, next, next + countOfNodes + 1, 0);
			} while((awake_count >= old_awake_count) || (awake_count > countOfNodes / beta));
			//} while(awake_count>0);
			nitems = awake_count;
			scout_count = 1;
#ifdef TD
			in_frontier->clearHost();
			BitmapToQueue<<<((countOfNodes-1)/512+1), 512>>>(countOfNodes, front, *in_frontier);
#endif
		} else{ 
			depth++;
			edges_to_check -= scout_count;
#ifdef TD
			nitems = in_frontier->getSize();
			setGridDimension(&dimGrid, nitems, opt.block_size);
			CUDA_SAFE_CALL(cudaMemcpy(d_scout_count, &zero, sizeof(int), cudaMemcpyHostToDevice));
			fwd_step1_td<<<dimGrid, BLKSIZE>>>(d_nodes, d_edges, d_states, countOfNodes, d_scout_count, d_degree, *in_frontier, *out_frontier);
			CUDA_SAFE_CALL(cudaMemcpy(&scout_count, d_scout_count, sizeof(int), cudaMemcpyDeviceToHost));
			//nitems = out_frontier->nitems();
			nitems = out_frontier->getSize();
			Worklist2 *tmp = in_frontier;
			in_frontier = out_frontier;
			out_frontier = tmp;
			//out_frontier->reset();
			out_frontier->clearHost();
			//printf("TD: iteration=%d, num_frontier=%d\n", depth, nitems);
#else
			setGridDimension(&dimGrid, countOfNodes, opt.block_size);
			//gpuErrchk( cudaMemset((void *)d_end, true, sizeof(*d_end)) );
			//CUDA_SAFE_CALL(cudaMemcpy(d_scout_count, &zero, sizeof(int), cudaMemcpyHostToDevice));
			thrust::fill(thrust::device, d_scout_count, d_scout_count + countOfNodes + 1, 0);
			fwd_step1_topo<<<dimGrid, BLKSIZE>>>(d_nodes, d_edges, d_states, countOfNodes, d_scout_count, d_degree, NULL);
			//CUDA_SAFE_CALL(cudaMemcpy(&scout_count, d_scout_count, sizeof(int), cudaMemcpyDeviceToHost));
			scout_count = thrust::reduce(thrust::device, d_scout_count, d_scout_count + countOfNodes + 1, 0, thrust::plus<int>());
			//gpuErrchk( cudaMemcpy(&end, (void *)d_end, sizeof(*d_end), cudaMemcpyDeviceToHost) );
			//if(end) nitems = 0;
			if(scout_count == 0) nitems = 0;
			//printf("TOPO: iteration=%d, scout_count=%d\n", depth, scout_count);
#endif
		}
	} while (nitems > 0);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaFree(d_scout_count));
	gpuErrchk(cudaFree(d_degree));
	gpuErrchk(cudaFree(front));
	gpuErrchk(cudaFree(next));
	gpuErrchk(cudaFree(d_frontier_size));
	//gpuErrchk( cudaFree((void *) d_end) );
	return depth;
}
//*/
///*
#define BW_TD
unsigned bwd_reach1_b(unsigned *d_nodes, unsigned *d_nodesT, unsigned countOfNodes, unsigned *d_edges, unsigned *d_edgesT, unsigned countOfEdges, unsigned char *d_states, alg_opt opt, int *h_degree, int source) {
	unsigned depth = 0;
	dim3 dimGrid;
#ifdef BW_TD
	Worklist2 queue1(countOfEdges), queue2(countOfEdges);
	Worklist2 *in_frontier = &queue1, *out_frontier = &queue2;
	insert<<<1, 32>>>(source, *in_frontier);
#endif
	
	int *front, *next;
	int zero = 0;
	int alpha = 15, beta = 18;
	int nitems = 1;
	int edges_to_check = countOfEdges;
	int scout_count = h_degree[source];
	int *d_scout_count, *d_frontier_size;
	int *d_degree;
#ifdef BW_TD
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_scout_count, sizeof(int)));
#else
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_scout_count, (countOfNodes+1) * sizeof(int)));
#endif
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_frontier_size, sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_degree, (countOfNodes+1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_degree, h_degree, (countOfNodes+1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMalloc((void **)&front, (countOfNodes+1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&next, (countOfNodes+1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemset(front, 0, (countOfNodes+1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemset(next, 0, (countOfNodes+1) * sizeof(int)));

	do {
		if(scout_count > edges_to_check / alpha) {
		//if(1) {
			int awake_count, old_awake_count;
#ifdef BW_TD
			QueueToBitmap<<<((countOfNodes-1)/512+1), 512>>>(countOfNodes, *in_frontier, front);
#else
			FindBwdQueue<<<((countOfNodes-1)/512+1), 512>>>(countOfNodes, d_states, front);
#endif
			awake_count = nitems;
			do {
				depth++;
				old_awake_count = awake_count;
				setGridDimension(&dimGrid, countOfNodes, opt.block_size);
				//gpuErrchk(cudaMemcpy(d_frontier_size, &zero, sizeof(int), cudaMemcpyHostToDevice));
				bwd_step1_bu<<<dimGrid, opt.block_size>>>(d_nodes, d_edges, d_states, countOfNodes, front, next, d_frontier_size);
				//gpuErrchk( cudaMemcpy(&awake_count, (void *)d_frontier_size, sizeof(int), cudaMemcpyDeviceToHost) );
				awake_count = thrust::reduce(thrust::device, next + 1, next + countOfNodes + 1, 0, thrust::plus<int>());
				//printf("BU: iteration=%d, num_frontier=%d\n", depth, awake_count);
				int *temp = front;
				front = next;
				next = temp;
				thrust::fill(thrust::device, next, next + countOfNodes + 1, 0);
			} while((awake_count >= old_awake_count) || (awake_count > countOfNodes / beta));
			nitems = awake_count;
			scout_count = 1;
#ifdef BW_TD
			in_frontier->clearHost();
			BitmapToQueue<<<((countOfNodes-1)/512+1), 512>>>(countOfNodes, front, *in_frontier);
#endif
		} else{ 
			depth++;
			edges_to_check -= scout_count;
#ifdef BW_TD
			nitems = in_frontier->getSize();
			setGridDimension(&dimGrid, nitems, opt.block_size);
			CUDA_SAFE_CALL(cudaMemcpy(d_scout_count, &zero, sizeof(int), cudaMemcpyHostToDevice));
			bwd_step1_td<<<dimGrid, BLKSIZE>>>(d_nodesT, d_edgesT, d_states, countOfNodes, d_scout_count, d_degree, *in_frontier, *out_frontier);
			CUDA_SAFE_CALL(cudaMemcpy(&scout_count, d_scout_count, sizeof(int), cudaMemcpyDeviceToHost));
			nitems = out_frontier->getSize();
			Worklist2 *tmp = in_frontier;
			in_frontier = out_frontier;
			out_frontier = tmp;
			out_frontier->clearHost();
			//printf("TD: iteration=%d, num_frontier=%d\n", depth, nitems);
#else
			setGridDimension(&dimGrid, countOfNodes, opt.block_size);
			thrust::fill(thrust::device, d_scout_count, d_scout_count + countOfNodes + 1, 0);
			bwd_step1_topo<<<dimGrid, BLKSIZE>>>(d_nodesT, d_edgesT, d_states, countOfNodes, d_scout_count, d_degree, NULL);
			scout_count = thrust::reduce(thrust::device, d_scout_count, d_scout_count + countOfNodes + 1, 0, thrust::plus<int>());
			if(scout_count == 0) nitems = 0;
			//printf("TOPO: iteration=%d, scout_count=%d\n", depth, scout_count);
#endif
		}
	} while (nitems > 0);
	gpuErrchk(cudaFree(d_scout_count));
	gpuErrchk(cudaFree(d_degree));
	gpuErrchk(cudaFree(front));
	gpuErrchk(cudaFree(next));
	gpuErrchk(cudaFree(d_frontier_size));
	return depth;
}
//*/

/*
unsigned bwd_reach1_b(unsigned *d_nodes, unsigned *d_nodesT, unsigned countOfNodes, unsigned *d_edges, unsigned *d_edgesT, unsigned countOfEdges, unsigned char *d_states, alg_opt opt, int *h_degree, int source) {
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
//*/
/*
unsigned fwd_reach1_b(unsigned *d_nodes, unsigned *d_nodesT, unsigned countOfNodes, unsigned *d_edges, unsigned *d_edgesT, unsigned countOfEdges, unsigned char *d_states, alg_opt opt, int *h_degree, int source) {
	bool volatile *d_end;
	bool end = false;
	unsigned depth = 0;
	dim3 dimGrid;
	setGridDimension(&dimGrid, countOfNodes, opt.block_size);
	gpuErrchk( cudaMalloc((void**)&d_end, sizeof(*d_end)) );
	while(!end) {
		gpuErrchk( cudaMemset((void *)d_end, true, sizeof(*d_end)) ); 
		fwd_step1_topo<<<dimGrid, opt.block_size>>>(d_nodes, d_edges, d_states, countOfNodes, d_end);
		gpuErrchk( cudaMemcpy(&end, (void *)d_end, sizeof(*d_end), cudaMemcpyDeviceToHost) );		
		depth++;
	}
	gpuErrchk( cudaFree((void *) d_end) );
	return depth;
}
//*/
