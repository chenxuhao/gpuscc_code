#include <stdint.h>

#include "scc.h"
#include "reach_q.h"
#include "GPUerrchk.h"
#include "timing.h"

#define BLKSIZE 256
//#define DEBUG


static __global__ void bwd_step(unsigned *nodes, unsigned *edges, unsigned *ranges, unsigned char *states, unsigned countOfNodes, bool volatile *end) {

	unsigned id = blockDim.x * blockIdx.x + threadIdx.x + 1;
	if (id > countOfNodes) return;

	unsigned my_start;
	unsigned my_end;
	unsigned my_range;
	unsigned char my_state = states[id];

	//if(!isElim(&my_state) && !isBExt(&my_state) && isBVis(&my_state)) {
	if((my_state & 37) == 1) {
		setBExt(&states[id]);
		my_start = nodes[id];
		my_end = nodes[id + 1];
		my_range = ranges[id];

		unsigned nbi;
		unsigned char state_i;
		for(unsigned i = my_start; i < my_end; i++) {
			nbi = edges[i];
			state_i = states[nbi];
			//if(!isElim(&state_i) && !isBVis(&state_i) && my_range == ranges[nbi]) {
			if((state_i & 5) == 0 && my_range == ranges[nbi]) {
				setBVis(&states[nbi]);
				*end = false;	 
			}
		}
	}
}

#ifdef VERIFY					
static __global__ void fwd_step(unsigned *nodes, unsigned *edges, unsigned *ranges, unsigned char *states, unsigned countOfNodes, bool volatile *end, unsigned *scc_root) {
#else
static __global__ void fwd_step(unsigned *nodes, unsigned *edges, unsigned *ranges, unsigned char *states, unsigned countOfNodes, bool volatile *end) {
#endif

	unsigned id = blockDim.x * blockIdx.x + threadIdx.x + 1;
	if (id > countOfNodes) return;

	unsigned my_start;
	unsigned my_end;
	unsigned my_range;
	unsigned char my_state = states[id];
	
	//if(!isElim(&my_state) && !isFExt(&my_state) && isFVis(&my_state)) {
	if((my_state & 70) == 2) {
		setFExt(&states[id]);
		my_start = nodes[id];
		my_end = nodes[id + 1];
		my_range = ranges[id];

		unsigned nbi;
		unsigned char state_i;
		for(unsigned i = my_start; i < my_end; i++) {
			nbi = edges[i];
			state_i = states[nbi];
			//if(!isElim(&state_i) && !isFVis(&state_i) && my_range == ranges[nbi]) {
			if((state_i & 6) == 0 && my_range == ranges[nbi]) {
				#ifdef VERIFY
				scc_root[nbi] = scc_root[id];
				#endif
				setFVis(&states[nbi]);
				*end = false;	 
			}
		}
	}		
}
					
unsigned bwd_reach_q(unsigned *d_nodes, unsigned countOfNodes, unsigned *d_edges, unsigned countOfEdges, unsigned *d_ranges, unsigned char *d_states, alg_opt opt) {
				
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
unsigned fwd_reach_q(unsigned *d_nodes, unsigned countOfNodes, unsigned *d_edges, unsigned countOfEdges, unsigned *d_ranges, unsigned char *d_states, alg_opt opt, unsigned *scc_root) {
#else
unsigned fwd_reach_q(unsigned *d_nodes, unsigned countOfNodes, unsigned *d_edges, unsigned countOfEdges, unsigned *d_ranges, unsigned char *d_states, alg_opt opt) {
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
	if (id > countOfNodes) return;

	unsigned my_start;
	unsigned my_end;
	//unsigned my_range;
	unsigned char my_state = states[id];

	//if(!isElim(&my_state) && !isBExt(&my_state) && isBVis(&my_state)) {
	if((my_state & 37) == 1) {
		setBExt(&states[id]);
		my_start = nodes[id];
		my_end = nodes[id + 1];
		//my_range = ranges[id];

		unsigned nbi;
		unsigned char state_i;
		for(unsigned i = my_start; i < my_end; i++) {
			nbi = edges[i];
			state_i = states[nbi];
			//if(!isElim(&state_i) && !isBVis(&state_i) && my_range == ranges[nbi]) {
			//if((state_i & 5) == 0 && my_range == ranges[nbi]) {
			if((state_i & 5) == 0) {
				setBVis(&states[nbi]);
				*end = false;	 
			}
		}
	}
}

#ifdef VERIFY					
//static __global__ void fwd_step(unsigned *nodes, unsigned *edges, unsigned *ranges, unsigned char *states, unsigned countOfNodes, bool volatile *end, unsigned *scc_root) {
static __global__ void fwd_step1(unsigned *nodes, unsigned *edges, unsigned char *states, unsigned countOfNodes, bool volatile *end, unsigned *scc_root) {
#else
//static __global__ void fwd_step(unsigned *nodes, unsigned *edges, unsigned *ranges, unsigned char *states, unsigned countOfNodes, bool volatile *end) {
static __global__ void fwd_step1(unsigned *nodes, unsigned *edges, unsigned char *states, unsigned countOfNodes, bool volatile *end) {
#endif

	unsigned id = blockDim.x * blockIdx.x + threadIdx.x + 1;
	if (id > countOfNodes) return;

	unsigned my_start;
	unsigned my_end;
	//unsigned my_range;
	unsigned char my_state = states[id];
	
	//if(!isElim(&my_state) && !isFExt(&my_state) && isFVis(&my_state)) {
	if((my_state & 70) == 2) {
		setFExt(&states[id]);
		my_start = nodes[id];
		my_end = nodes[id + 1];
		//my_range = ranges[id];

		unsigned nbi;
		unsigned char state_i;
		for(unsigned i = my_start; i < my_end; i++) {
			nbi = edges[i];
			state_i = states[nbi];
			//if(!isElim(&state_i) && !isFVis(&state_i) && my_range == ranges[nbi]) {
			//if((state_i & 6) == 0 && my_range == ranges[nbi]) {
			if((state_i & 6) == 0) {
				#ifdef VERIFY
				scc_root[nbi] = scc_root[id];
				#endif
				setFVis(&states[nbi]);
				*end = false;	 
			}
		}
	}		
}
					
//unsigned bwd_reach_q(unsigned *d_nodes, unsigned countOfNodes, unsigned *d_edges, unsigned countOfEdges, unsigned *d_ranges, unsigned char *d_states, alg_opt opt) {
unsigned bwd_reach1_q(unsigned *d_nodes, unsigned countOfNodes, unsigned *d_edges, unsigned countOfEdges, unsigned char *d_states, alg_opt opt) {
				
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
//unsigned fwd_reach_q(unsigned *d_nodes, unsigned countOfNodes, unsigned *d_edges, unsigned countOfEdges, unsigned *d_ranges, unsigned char *d_states, alg_opt opt, unsigned *scc_root) {
unsigned fwd_reach1_q(unsigned *d_nodes, unsigned countOfNodes, unsigned *d_edges, unsigned countOfEdges, unsigned char *d_states, alg_opt opt, unsigned *scc_root) {
#else
//unsigned fwd_reach_q(unsigned *d_nodes, unsigned countOfNodes, unsigned *d_edges, unsigned countOfEdges, unsigned *d_ranges, unsigned char *d_states, alg_opt opt) {
unsigned fwd_reach1_q(unsigned *d_nodes, unsigned countOfNodes, unsigned *d_edges, unsigned countOfEdges, unsigned char *d_states, alg_opt opt) {
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
