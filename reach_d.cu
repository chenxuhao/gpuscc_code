#include <stdint.h>

#include "scc.h"
#include "reach_d.h"
#include "GPUerrchk.h"

#define	BLOCK	256
//#define TRACE


static __global__ void bwd_step(unsigned *nodes, unsigned *edges, unsigned *ranges, unsigned char *states, unsigned countOfNodes, Worklist2 inwl, Worklist2 outwl) {

	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned nn;

	if (!inwl.pop_id(id, nn))
		return;

	unsigned my_start = nodes[nn];
	unsigned my_end = nodes[nn + 1];	
	unsigned my_range = ranges[nn];
	unsigned dst;
	
	for (unsigned ii = my_start; ii < my_end; ++ii) {
		dst = edges[ii];
		//if (dst != nn) {
			if ((states[dst] & 5) == 0 && my_range == ranges[dst]) {
				setBVis(&states[dst]);
				outwl.push(dst);
			}
		//}
	}
}

#ifdef VERIFY
static __global__ void fwd_step(unsigned *nodes, unsigned *edges, unsigned *ranges, unsigned char *states, unsigned countOfNodes, Worklist2 inwl, Worklist2 outwl, unsigned *scc_root) {
#else
static __global__ void fwd_step(unsigned *nodes, unsigned *edges, unsigned *ranges, unsigned char *states, unsigned countOfNodes, Worklist2 inwl, Worklist2 outwl) {
#endif

	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned nn;

	if (!inwl.pop_id(id, nn))
		return;

	unsigned my_start = nodes[nn];
	unsigned my_end = nodes[nn + 1];
	unsigned my_range = ranges[nn];
	unsigned dst;
	
	for (unsigned ii = my_start; ii < my_end; ++ii) {
		dst = edges[ii];
		//if (dst != nn) {
			if ((states[dst] & 6) == 0 && my_range == ranges[dst]) {
				setFVis(&states[dst]);
			#ifdef VERIFY
				scc_root[dst] = scc_root[nn];
			#endif
				outwl.push(dst);
			}
		//}
	}
}

unsigned bwd_reach_d(unsigned *nodes, unsigned *edges, unsigned *ranges, unsigned char *states, unsigned countOfNodes, Worklist2 &wl) {

	unsigned depth = 0;
	Worklist2 inwl(countOfNodes), outwl(countOfNodes);
	Worklist2 *inwlptr, *outwlptr, *tmp;

	inwlptr = &inwl;
	outwlptr = &outwl;

	unsigned wlsz = wl.getSize();
	gpuErrchk( cudaMemcpy(inwl.dwl, wl.dwl, wlsz * sizeof(unsigned), cudaMemcpyDeviceToDevice));
	gpuErrchk( cudaMemcpy(inwl.dindex, wl.dindex, sizeof(*(wl.dindex)), cudaMemcpyDeviceToDevice) );
	

	dim3 dimGrid;
	cudaError_t error;

	do {
		setGridDimension(&dimGrid, wlsz, BLOCK);
		bwd_step<<<dimGrid, BLOCK>>>(nodes, edges, ranges, states, countOfNodes, *inwlptr, *outwlptr);
		//cudaDeviceSynchronize();
	
		wlsz = outwlptr->getSize();

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
unsigned fwd_reach_d(unsigned *nodes, unsigned *edges, unsigned *ranges, unsigned char *states, unsigned countOfNodes, Worklist2 &wl, unsigned *scc_root) {
#else
unsigned fwd_reach_d(unsigned *nodes, unsigned *edges, unsigned *ranges, unsigned char *states, unsigned countOfNodes, Worklist2 &wl) {
#endif

	unsigned depth = 0;
	Worklist2 inwl(countOfNodes), outwl(countOfNodes);
	Worklist2 *inwlptr, *outwlptr, *tmp;

	inwlptr = &inwl;
	outwlptr = &outwl;

	unsigned wlsz = wl.getSize();
	gpuErrchk( cudaMemcpy(inwl.dwl, wl.dwl, wlsz * sizeof(unsigned), cudaMemcpyDeviceToDevice));
	gpuErrchk( cudaMemcpy(inwl.dindex, wl.dindex, sizeof(*(wl.dindex)), cudaMemcpyDeviceToDevice) );
	

	dim3 dimGrid;
	cudaError_t error;

	do {
		setGridDimension(&dimGrid, wlsz, BLOCK);
	#ifdef VERIFY
		fwd_step<<<dimGrid, BLOCK>>>(nodes, edges, ranges, states, countOfNodes, *inwlptr, *outwlptr, scc_root);
	#else
		fwd_step<<<dimGrid, BLOCK>>>(nodes, edges, ranges, states, countOfNodes, *inwlptr, *outwlptr);
	#endif
		//cudaDeviceSynchronize();
	
		wlsz = outwlptr->getSize();

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

	if (!inwl.pop_id(id, nn))
		return;

	unsigned my_start = nodes[nn];
	unsigned my_end = nodes[nn + 1];	
	//unsigned my_range = ranges[nn];
	unsigned dst;
	
	for (unsigned ii = my_start; ii < my_end; ++ii) {
		dst = edges[ii];
		//if (dst != nn) {
			//if ((states[dst] & 5) == 0 && my_range == ranges[dst]) {
			if ((states[dst] & 5) == 0) {
				setBVis(&states[dst]);
				outwl.push(dst);
			}
		//}
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

	if (!inwl.pop_id(id, nn))
		return;

	unsigned my_start = nodes[nn];
	unsigned my_end = nodes[nn + 1];
	//unsigned my_range = ranges[nn];
	unsigned dst;
	
	for (unsigned ii = my_start; ii < my_end; ++ii) {
		dst = edges[ii];
		//if (dst != nn) {
			//if ((states[dst] & 6) == 0 && my_range == ranges[dst]) {
			if ((states[dst] & 6) == 0) {
				setFVis(&states[dst]);
			#ifdef VERIFY
				scc_root[dst] = scc_root[nn];
			#endif
				outwl.push(dst);
			}
		//}
	}
}

//unsigned bwd_reach_d(unsigned *nodes, unsigned *edges, unsigned *ranges, unsigned char *states, unsigned countOfNodes, Worklist2 &wl) {
unsigned bwd_reach1_d(unsigned *nodes, unsigned *edges, unsigned char *states, unsigned countOfNodes, Worklist2 &wl) {

	unsigned depth = 0;
	Worklist2 inwl(countOfNodes), outwl(countOfNodes);
	Worklist2 *inwlptr, *outwlptr, *tmp;

	inwlptr = &inwl;
	outwlptr = &outwl;

	unsigned wlsz = wl.getSize();
	gpuErrchk( cudaMemcpy(inwl.dwl, wl.dwl, wlsz * sizeof(unsigned), cudaMemcpyDeviceToDevice));
	gpuErrchk( cudaMemcpy(inwl.dindex, wl.dindex, sizeof(*(wl.dindex)), cudaMemcpyDeviceToDevice) );
	

	dim3 dimGrid;
	cudaError_t error;

	do {
		setGridDimension(&dimGrid, wlsz, BLOCK);
		//bwd_step<<<dimGrid, BLOCK>>>(nodes, edges, ranges, states, countOfNodes, *inwlptr, *outwlptr);
		bwd_step1<<<dimGrid, BLOCK>>>(nodes, edges, states, countOfNodes, *inwlptr, *outwlptr);
		//cudaDeviceSynchronize();
	
		wlsz = outwlptr->getSize();

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
//unsigned fwd_reach_d(unsigned *nodes, unsigned *edges, unsigned *ranges, unsigned char *states, unsigned countOfNodes, Worklist2 &wl, unsigned *scc_root) {
unsigned fwd_reach1_d(unsigned *nodes, unsigned *edges, unsigned char *states, unsigned countOfNodes, Worklist2 &wl, unsigned *scc_root) {
#else
//unsigned fwd_reach_d(unsigned *nodes, unsigned *edges, unsigned *ranges, unsigned char *states, unsigned countOfNodes, Worklist2 &wl) {
unsigned fwd_reach1_d(unsigned *nodes, unsigned *edges, unsigned char *states, unsigned countOfNodes, Worklist2 &wl) {
#endif

	unsigned depth = 0;
	Worklist2 inwl(countOfNodes), outwl(countOfNodes);
	Worklist2 *inwlptr, *outwlptr, *tmp;

	inwlptr = &inwl;
	outwlptr = &outwl;

	unsigned wlsz = wl.getSize();
	gpuErrchk( cudaMemcpy(inwl.dwl, wl.dwl, wlsz * sizeof(unsigned), cudaMemcpyDeviceToDevice));
	gpuErrchk( cudaMemcpy(inwl.dindex, wl.dindex, sizeof(*(wl.dindex)), cudaMemcpyDeviceToDevice) );
	

	dim3 dimGrid;
	cudaError_t error;

	do {
		setGridDimension(&dimGrid, wlsz, BLOCK);
	#ifdef VERIFY
		//fwd_step<<<dimGrid, BLOCK>>>(nodes, edges, ranges, states, countOfNodes, *inwlptr, *outwlptr, scc_root);
		fwd_step1<<<dimGrid, BLOCK>>>(nodes, edges, states, countOfNodes, *inwlptr, *outwlptr, scc_root);
	#else
		//fwd_step<<<dimGrid, BLOCK>>>(nodes, edges, ranges, states, countOfNodes, *inwlptr, *outwlptr);
		fwd_step1<<<dimGrid, BLOCK>>>(nodes, edges, states, countOfNodes, *inwlptr, *outwlptr);
	#endif
		//cudaDeviceSynchronize();
	
		wlsz = outwlptr->getSize();

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
