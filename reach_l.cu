#include <stdint.h>

#include "scc.h"
#include "reach_l.h"
#include "GPUerrchk.h"
#include "timing.h"

#include "cub/cub.cuh"

#define	BLKSIZE	256

#define WARPSIZE 32 
#define BLOCK_HIST_SIZE 512

//#define EXPAND_BY_CTA

//#define HISTCULL

//#define PUSH_RANGE
//#define DEBUG


static __device__ bool WarpCull(unsigned nb, unsigned w_id) {

	volatile __shared__ unsigned scratch[BLKSIZE/WARPSIZE][128];
	unsigned hash = nb & 127;
	
	scratch [w_id][hash] = nb;

	if (scratch[w_id][hash] == nb) {
		scratch[w_id][hash] == threadIdx.x;
		if (scratch[w_id][hash] != threadIdx.x)
			return true;
	}
	return false;
}

static __device__ bool HistCull(unsigned nb, volatile unsigned *hist) {

	unsigned hash = nb & (BLOCK_HIST_SIZE - 1);
	if (hist[hash] == 0) {
		hist[hash] = nb;
		return false;
	}

	if (hist[hash] == nb) {
		return true;
	}

	return false;
}

static __global__ void bwd_step(unsigned *nodes, unsigned *edges, unsigned *ranges, unsigned char *states, unsigned countOfNodes, Worklist2 inwl, Worklist2 outwl) {

	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned nn;

	typedef cub::BlockScan<unsigned, BLKSIZE> BlockScan;

	__shared__ BlockScan::TempStorage temp_storage;
	__shared__ unsigned gather_offsets[BLKSIZE];
	__shared__ unsigned srcsrc[BLKSIZE];

	__shared__ unsigned sh_ranges[BLKSIZE];
	__shared__ unsigned srcIndex[BLKSIZE];

	gather_offsets[threadIdx.x] = 0;

	unsigned neighborsize = 0;
	unsigned neighboroffset = 0;
	unsigned scratch_offset = 0;
	unsigned total_edges = 0;

#ifdef HISTCULL
	volatile __shared__ unsigned hist[BLOCK_HIST_SIZE];
	for (unsigned i = threadIdx.x; i < BLOCK_HIST_SIZE; i += BLKSIZE) {
		hist[i] = 0;
	}
	__syncthreads();
#endif

	if (inwl.pop_id(id, nn)) {
	#ifdef HISTCULL
		#ifdef STATE_CULL
		if (!HistCull(nn, hist) && !isBExt(&states[nn])) {
			setBExt(&states[nn]);
		#else
		if (!HistCull(nn, hist)) {
		#endif
			neighboroffset = nodes[nn];
			neighborsize = nodes[nn + 1] - neighboroffset;
		}
	#else
		neighboroffset = nodes[nn];
		neighborsize = nodes[nn + 1] - neighboroffset;

		sh_ranges[threadIdx.x] = ranges[nn];
	#endif
	}

	BlockScan(temp_storage).ExclusiveSum(neighborsize, scratch_offset, total_edges);
	unsigned done = 0;
	unsigned neighborsdone = 0;

	while ((int)total_edges > 0) {
		__syncthreads();
		unsigned i;
		unsigned index;
		for (i = 0; neighborsdone + i < neighborsize && (index = scratch_offset + i - done) < BLKSIZE; i++) {
			gather_offsets[index] = neighboroffset + neighborsdone + i;
			srcsrc[index] = nn;

			srcIndex[index] = threadIdx.x;
		}
		neighborsdone += i;
		scratch_offset += i;
		__syncthreads();

	#ifdef PUSH_RANGE
		unsigned ncnt = 0;
	#endif
		unsigned src;
		unsigned dst;

		if (threadIdx.x < total_edges) {
			src = srcsrc[threadIdx.x];
			dst = edges[gather_offsets[threadIdx.x]];
			
			//if (src != dst) {
				//if ((states[dst] & 5) == 0 && ranges[dst] == ranges[src]) {
				if ((states[dst] & 5) == 0 && ranges[dst] == sh_ranges[srcIndex[threadIdx.x]]) {
					setBVis(&states[dst]);
				#ifdef PUSH_RANGE
					ncnt = 1;
				#else
					outwl.push(dst);
				#endif
				}
			//}
		}
		__syncthreads();
	#ifdef PUSH_RANGE
		outwl.push_1item<BlockScan>(ncnt, dst, BLKSIZE);
	#endif
		done += BLKSIZE;
		total_edges -= BLKSIZE;
	}
}

#ifdef VERIFY
static __global__ void fwd_step(unsigned *nodes, unsigned *edges, unsigned *ranges, unsigned char *states, unsigned countOfNodes, Worklist2 inwl, Worklist2 outwl, unsigned *scc_root) {
#else
static __global__ void fwd_step(unsigned *nodes, unsigned *edges, unsigned *ranges, unsigned char *states, unsigned countOfNodes, Worklist2 inwl, Worklist2 outwl) {
#endif

	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned nn;

	typedef cub::BlockScan<unsigned, BLKSIZE> BlockScan;

	__shared__ BlockScan::TempStorage temp_storage;
	__shared__ unsigned gather_offsets[BLKSIZE];
	__shared__ unsigned srcsrc[BLKSIZE];

	__shared__ unsigned sh_ranges[BLKSIZE];
	__shared__ unsigned srcIndex[BLKSIZE];

	gather_offsets[threadIdx.x] = 0;

	unsigned neighborsize = 0;
	unsigned neighboroffset = 0;
	unsigned scratch_offset = 0;
	unsigned total_edges = 0;

#ifdef HISTCULL
	volatile __shared__ unsigned hist[BLOCK_HIST_SIZE];
	for (unsigned i = threadIdx.x; i < BLOCK_HIST_SIZE; i += BLKSIZE) {
		hist[i] = 0;
	}
	__syncthreads();
#endif

	if (inwl.pop_id(id, nn)) {
	#ifdef HISTCULL
		#ifdef STATE_CULL
		if (!HistCull(nn, hist) && !isFExt(&states[nn])) {
			setFExt(&states[nn]);
		#else
		if (!HistCull(nn, hist)) {
		#endif
			neighboroffset = nodes[nn];
			neighborsize = nodes[nn + 1] - neighboroffset;
		}
	#else
		neighboroffset = nodes[nn];
		neighborsize = nodes[nn + 1] - neighboroffset;

		sh_ranges[threadIdx.x] = ranges[nn];
	#endif
	}

	BlockScan(temp_storage).ExclusiveSum(neighborsize, scratch_offset, total_edges);
	unsigned done = 0;
	unsigned neighborsdone = 0;

	while ((int)total_edges > 0) {
		__syncthreads();
		unsigned i;
		unsigned index;
		for (i = 0; neighborsdone + i < neighborsize && (index = scratch_offset + i - done) < BLKSIZE; i++) {
			gather_offsets[index] = neighboroffset + neighborsdone + i;
			srcsrc[index] = nn;

			srcIndex[index] = threadIdx.x;
		}
		neighborsdone += i;
		scratch_offset += i;
		__syncthreads();

	#ifdef PUSH_RANGE
		unsigned ncnt = 0;
	#endif
		unsigned src;
		unsigned dst;

		if (threadIdx.x < total_edges) {
			src = srcsrc[threadIdx.x];
			dst = edges[gather_offsets[threadIdx.x]];
			
			//if (src != dst) {
				//if ((states[dst] & 6) == 0 && ranges[dst] == ranges[src]) {
				if ((states[dst] & 6) == 0 && ranges[dst] == sh_ranges[srcIndex[threadIdx.x]]) {
					setFVis(&states[dst]);
				#ifdef PUSH_RANGE
					ncnt = 1;
				#else
					outwl.push(dst);
				#endif

				#ifdef VERIFY
					scc_root[dst] = scc_root[src];
				#endif
				}
			//}
		}
		__syncthreads();
	#ifdef PUSH_RANGE
		outwl.push_1item<BlockScan>(ncnt, dst, BLKSIZE);
	#endif
		done += BLKSIZE;
		total_edges -= BLKSIZE;
	}
}

unsigned bwd_reach_l(unsigned *nodes, unsigned *edges, unsigned *ranges, unsigned char *states, unsigned countOfNodes, Worklist2 &wl) {

	unsigned depth = 0;
	Worklist2 inwl(countOfNodes), outwl(countOfNodes);
	Worklist2 *inwlptr, *outwlptr, *tmp;

	inwlptr = &inwl;
	outwlptr = &outwl;

	unsigned wlsz = wl.getSize();
	gpuErrchk( cudaMemcpy(inwl.dwl, wl.dwl, wlsz * sizeof(unsigned), cudaMemcpyDeviceToDevice) );
	gpuErrchk( cudaMemcpy(inwl.dindex, wl.dindex, sizeof(*(wl.dindex)), cudaMemcpyDeviceToDevice) );

	dim3 dimGrid;

#ifdef DEBUG
	timeval timer;
#endif

	do {
		setGridDimension(&dimGrid, wlsz, BLKSIZE);
	#ifdef DEBUG
		startTimer(&timer);
	#endif
		bwd_step<<<dimGrid, BLKSIZE>>>(nodes, edges, ranges, states, countOfNodes, *inwlptr, *outwlptr);
		//gpuErrchk( cudaDeviceSynchronize() );
	
		wlsz = outwlptr->getSize();
	#ifdef DEBUG
		printf("bwd iteration %d time: %f ms wlsz: %d\n", depth + 1, stopTimer(&timer) / 1000.0, wlsz);
	#endif

		tmp = inwlptr;
		inwlptr = outwlptr;
		outwlptr = tmp;

		outwlptr->clearHost();
		depth++;
	} while (wlsz);

	inwlptr->dealloc();
	outwlptr->dealloc();

	return depth;
}

#ifdef VERIFY
unsigned fwd_reach_l(unsigned *nodes, unsigned *edges, unsigned *ranges, unsigned char *states, unsigned countOfNodes, Worklist2 &wl, unsigned *scc_root) {
#else
unsigned fwd_reach_l(unsigned *nodes, unsigned *edges, unsigned *ranges, unsigned char *states, unsigned countOfNodes, Worklist2 &wl) {
#endif

	unsigned depth = 0;
	Worklist2 inwl(countOfNodes), outwl(countOfNodes);
	Worklist2 *inwlptr, *outwlptr, *tmp;

	inwlptr = &inwl;
	outwlptr = &outwl;

	unsigned wlsz = wl.getSize();
	gpuErrchk( cudaMemcpy(inwl.dwl, wl.dwl, wlsz * sizeof(unsigned), cudaMemcpyDeviceToDevice) );
	gpuErrchk( cudaMemcpy(inwl.dindex, wl.dindex, sizeof(*(wl.dindex)), cudaMemcpyDeviceToDevice) );

	dim3 dimGrid;

#ifdef DEBUG	
	timeval timer;
#endif

	do {
		setGridDimension(&dimGrid, wlsz, BLKSIZE);
	#ifdef DEBUG
		startTimer(&timer);
	#endif
	#ifdef VERIFY
		fwd_step<<<dimGrid, BLKSIZE>>>(nodes, edges, ranges, states, countOfNodes, *inwlptr, *outwlptr, scc_root);
	#else
		fwd_step<<<dimGrid, BLKSIZE>>>(nodes, edges, ranges, states, countOfNodes, *inwlptr, *outwlptr);
	#endif
		//gpuErrchk( cudaDeviceSynchronize() );
	
		wlsz = outwlptr->getSize();
	#ifdef DEBUG
		printf("fwd iteration %d time: %f ms wlsz: %d\n", depth + 1, stopTimer(&timer) / 1000.0, wlsz);
	#endif

		tmp = inwlptr;
		inwlptr = outwlptr;
		outwlptr = tmp;

		outwlptr->clearHost();
		depth++;
	} while (wlsz);

	inwlptr->dealloc();
	outwlptr->dealloc();

	return depth;
}

//static __global__ void bwd_step(unsigned *nodes, unsigned *edges, unsigned *ranges, unsigned char *states, unsigned countOfNodes, Worklist2 inwl, Worklist2 outwl) {
static __global__ void bwd_step1(unsigned *nodes, unsigned *edges, unsigned char *states, unsigned countOfNodes, Worklist2 inwl, Worklist2 outwl) {

	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned nn;

	typedef cub::BlockScan<unsigned, BLKSIZE> BlockScan;

	__shared__ BlockScan::TempStorage temp_storage;
	__shared__ unsigned gather_offsets[BLKSIZE];
	//__shared__ unsigned srcsrc[BLKSIZE];

	//__shared__ unsigned sh_ranges[BLKSIZE];
	//__shared__ unsigned srcIndex[BLKSIZE];

	gather_offsets[threadIdx.x] = 0;

	unsigned neighborsize = 0;
	unsigned neighboroffset = 0;
	unsigned scratch_offset = 0;
	unsigned total_edges = 0;

#ifdef HISTCULL
	volatile __shared__ unsigned hist[BLOCK_HIST_SIZE];
	for (unsigned i = threadIdx.x; i < BLOCK_HIST_SIZE; i += BLKSIZE) {
		hist[i] = 0;
	}
	__syncthreads();
#endif

	if (inwl.pop_id(id, nn)) {
	#ifdef HISTCULL
		#ifdef STATE_CULL
		if (!HistCull(nn, hist) && !isBExt(&states[nn])) {
			setBExt(&states[nn]);
		#else
		if (!HistCull(nn, hist)) {
		#endif
			neighboroffset = nodes[nn];
			neighborsize = nodes[nn + 1] - neighboroffset;
		}
	#else
		neighboroffset = nodes[nn];
		neighborsize = nodes[nn + 1] - neighboroffset;

		//sh_ranges[threadIdx.x] = ranges[nn];
	#endif
	}

	BlockScan(temp_storage).ExclusiveSum(neighborsize, scratch_offset, total_edges);
	unsigned done = 0;
	unsigned neighborsdone = 0;

	while ((int)total_edges > 0) {
		__syncthreads();
		unsigned i;
		unsigned index;
		for (i = 0; neighborsdone + i < neighborsize && (index = scratch_offset + i - done) < BLKSIZE; i++) {
			gather_offsets[index] = neighboroffset + neighborsdone + i;
			//srcsrc[index] = nn;

			//srcIndex[index] = threadIdx.x;
		}
		neighborsdone += i;
		scratch_offset += i;
		__syncthreads();

	#ifdef PUSH_RANGE
		unsigned ncnt = 0;
	#endif
		//unsigned src;
		unsigned dst;

		if (threadIdx.x < total_edges) {
			//src = srcsrc[threadIdx.x];
			dst = edges[gather_offsets[threadIdx.x]];
			
			//if (src != dst) {
				//if ((states[dst] & 5) == 0 && ranges[dst] == ranges[src]) {
				//if ((states[dst] & 5) == 0 && ranges[dst] == sh_ranges[srcIndex[threadIdx.x]]) {
				if ((states[dst] & 5) == 0) {
					setBVis(&states[dst]);
				#ifdef PUSH_RANGE
					ncnt = 1;
				#else
					outwl.push(dst);
				#endif
				}
			//}
		}
		__syncthreads();
	#ifdef PUSH_RANGE
		outwl.push_1item<BlockScan>(ncnt, dst, BLKSIZE);
	#endif
		done += BLKSIZE;
		total_edges -= BLKSIZE;
	}
}

#ifdef VERIFY
//static __global__ void fwd_step(unsigned *nodes, unsigned *edges, unsigned *ranges, unsigned char *states, unsigned countOfNodes, Worklist2 inwl, Worklist2 outwl, unsigned *scc_root) {
static __global__ void fwd_step1(unsigned *nodes, unsigned *edges, unsigned char *states, unsigned countOfNodes, Worklist2 inwl, Worklist2 outwl, unsigned *scc_root) {
#else
//static __global__ void fwd_step(unsigned *nodes, unsigned *edges, unsigned *ranges, unsigned char *states, unsigned countOfNodes, Worklist2 inwl, Worklist2 outwl) {
static __global__ void fwd_step1(unsigned *nodes, unsigned *edges, unsigned char *states, unsigned countOfNodes, Worklist2 inwl, Worklist2 outwl) {
#endif

	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned nn;

	typedef cub::BlockScan<unsigned, BLKSIZE> BlockScan;

	__shared__ BlockScan::TempStorage temp_storage;
	__shared__ unsigned gather_offsets[BLKSIZE];
	#ifdef VERIFY
	__shared__ unsigned srcsrc[BLKSIZE];
	#endif

	//__shared__ unsigned sh_ranges[BLKSIZE];
	//__shared__ unsigned srcIndex[BLKSIZE];

	gather_offsets[threadIdx.x] = 0;

	unsigned neighborsize = 0;
	unsigned neighboroffset = 0;
	unsigned scratch_offset = 0;
	unsigned total_edges = 0;

#ifdef HISTCULL
	volatile __shared__ unsigned hist[BLOCK_HIST_SIZE];
	for (unsigned i = threadIdx.x; i < BLOCK_HIST_SIZE; i += BLKSIZE) {
		hist[i] = 0;
	}
	__syncthreads();
#endif

	if (inwl.pop_id(id, nn)) {
	#ifdef HISTCULL
		#ifdef STATE_CULL
		if (!HistCull(nn, hist) && !isFExt(&states[nn])) {
			setFExt(&states[nn]);
		#else
		if (!HistCull(nn, hist)) {
		#endif
			neighboroffset = nodes[nn];
			neighborsize = nodes[nn + 1] - neighboroffset;
		}
	#else
		neighboroffset = nodes[nn];
		neighborsize = nodes[nn + 1] - neighboroffset;

		//sh_ranges[threadIdx.x] = ranges[nn];
	#endif
	}

	BlockScan(temp_storage).ExclusiveSum(neighborsize, scratch_offset, total_edges);
	unsigned done = 0;
	unsigned neighborsdone = 0;

	while ((int)total_edges > 0) {
		__syncthreads();
		unsigned i;
		unsigned index;
		for (i = 0; neighborsdone + i < neighborsize && (index = scratch_offset + i - done) < BLKSIZE; i++) {
			gather_offsets[index] = neighboroffset + neighborsdone + i;
			#ifdef VERIFY
			srcsrc[index] = nn;
			#endif

			//srcIndex[index] = threadIdx.x;
		}
		neighborsdone += i;
		scratch_offset += i;
		__syncthreads();

	#ifdef PUSH_RANGE
		unsigned ncnt = 0;
	#endif
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
				//if ((states[dst] & 6) == 0 && ranges[dst] == ranges[src]) {
				//if ((states[dst] & 6) == 0 && ranges[dst] == sh_ranges[srcIndex[threadIdx.x]]) {
				if ((states[dst] & 6) == 0) {
					setFVis(&states[dst]);
				#ifdef PUSH_RANGE
					ncnt = 1;
				#else
					outwl.push(dst);
				#endif

				#ifdef VERIFY
					scc_root[dst] = scc_root[src];
				#endif
				}
			//}
		}
		__syncthreads();
	#ifdef PUSH_RANGE
		outwl.push_1item<BlockScan>(ncnt, dst, BLKSIZE);
	#endif
		done += BLKSIZE;
		total_edges -= BLKSIZE;
	}
}

//unsigned bwd_reach_l(unsigned *nodes, unsigned *edges, unsigned *ranges, unsigned char *states, unsigned countOfNodes, Worklist2 &wl) {
unsigned bwd_reach1_l(unsigned *nodes, unsigned *edges, unsigned char *states, unsigned countOfNodes, Worklist2 &wl) {

	unsigned depth = 0;
	Worklist2 inwl(countOfNodes), outwl(countOfNodes);
	Worklist2 *inwlptr, *outwlptr, *tmp;

	inwlptr = &inwl;
	outwlptr = &outwl;

	unsigned wlsz = wl.getSize();
	gpuErrchk( cudaMemcpy(inwl.dwl, wl.dwl, wlsz * sizeof(unsigned), cudaMemcpyDeviceToDevice) );
	gpuErrchk( cudaMemcpy(inwl.dindex, wl.dindex, sizeof(*(wl.dindex)), cudaMemcpyDeviceToDevice) );

	dim3 dimGrid;

#ifdef DEBUG
	timeval timer;
#endif

	do {
		setGridDimension(&dimGrid, wlsz, BLKSIZE);
	#ifdef DEBUG
		startTimer(&timer);
	#endif
		//bwd_step<<<dimGrid, BLKSIZE>>>(nodes, edges, ranges, states, countOfNodes, *inwlptr, *outwlptr);
		bwd_step1<<<dimGrid, BLKSIZE>>>(nodes, edges, states, countOfNodes, *inwlptr, *outwlptr);
		//gpuErrchk( cudaDeviceSynchronize() );
	
		wlsz = outwlptr->getSize();
	#ifdef DEBUG
		printf("bwd iteration %d time: %f ms wlsz: %d\n", depth + 1, stopTimer(&timer) / 1000.0, wlsz);
	#endif

		tmp = inwlptr;
		inwlptr = outwlptr;
		outwlptr = tmp;

		outwlptr->clearHost();
		depth++;
	} while (wlsz);

	inwlptr->dealloc();
	outwlptr->dealloc();

	return depth;
}

#ifdef VERIFY
//unsigned fwd_reach_l(unsigned *nodes, unsigned *edges, unsigned *ranges, unsigned char *states, unsigned countOfNodes, Worklist2 &wl, unsigned *scc_root) {
unsigned fwd_reach1_l(unsigned *nodes, unsigned *edges, unsigned char *states, unsigned countOfNodes, Worklist2 &wl, unsigned *scc_root) {
#else
//unsigned fwd_reach_l(unsigned *nodes, unsigned *edges, unsigned *ranges, unsigned char *states, unsigned countOfNodes, Worklist2 &wl) {
unsigned fwd_reach1_l(unsigned *nodes, unsigned *edges, unsigned char *states, unsigned countOfNodes, Worklist2 &wl) {
#endif

	unsigned depth = 0;
	Worklist2 inwl(countOfNodes), outwl(countOfNodes);
	Worklist2 *inwlptr, *outwlptr, *tmp;

	inwlptr = &inwl;
	outwlptr = &outwl;

	unsigned wlsz = wl.getSize();
	gpuErrchk( cudaMemcpy(inwl.dwl, wl.dwl, wlsz * sizeof(unsigned), cudaMemcpyDeviceToDevice) );
	gpuErrchk( cudaMemcpy(inwl.dindex, wl.dindex, sizeof(*(wl.dindex)), cudaMemcpyDeviceToDevice) );

	dim3 dimGrid;

#ifdef DEBUG	
	timeval timer;
#endif

	do {
		setGridDimension(&dimGrid, wlsz, BLKSIZE);
	#ifdef DEBUG
		startTimer(&timer);
	#endif
	#ifdef VERIFY
		//fwd_step<<<dimGrid, BLKSIZE>>>(nodes, edges, ranges, states, countOfNodes, *inwlptr, *outwlptr, scc_root);
		fwd_step1<<<dimGrid, BLKSIZE>>>(nodes, edges, states, countOfNodes, *inwlptr, *outwlptr, scc_root);
	#else
		//fwd_step<<<dimGrid, BLKSIZE>>>(nodes, edges, ranges, states, countOfNodes, *inwlptr, *outwlptr);
		fwd_step1<<<dimGrid, BLKSIZE>>>(nodes, edges, states, countOfNodes, *inwlptr, *outwlptr);
	#endif
		//gpuErrchk( cudaDeviceSynchronize() );
	
		wlsz = outwlptr->getSize();
	#ifdef DEBUG
		printf("fwd iteration %d time: %f ms wlsz: %d\n", depth + 1, stopTimer(&timer) / 1000.0, wlsz);
	#endif

		tmp = inwlptr;
		inwlptr = outwlptr;
		outwlptr = tmp;

		outwlptr->clearHost();
		depth++;
	} while (wlsz);

	inwlptr->dealloc();
	outwlptr->dealloc();

	return depth;
}
