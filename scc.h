#ifndef SCC_H
#define SCC_H
// 2^20 = 1048575 
#define PIVOT_HASH_CONST 1048575 


enum reach_strategy {
	QUADRATIC,
	TOPO,
	DATA,
	LINEAR,
	HYBRID,
	COMBINE,
	BEST,
	ALL
};


struct alg_opt { 
	reach_strategy alg;
	int trim_strat;
	int trim_switch_limit;
	bool trim_both;
	int lin_cull_strat;
	int block_size;
	int warp_size;
};

inline void setGridDimension(dim3 *dimGrid, unsigned countOfThreads, unsigned block) {
/*	
	if((countOfThreads + block - 1)/block > 65535) {
		int dim = ceill(sqrt(countOfThreads / block));
		dimGrid->x = dim;
		dimGrid->y = dim;
		dimGrid->z = 1;
	}
	else {
		dimGrid->x = (countOfThreads + block - 1)/block;
		dimGrid->y = 1;
		dimGrid->z = 1;
	}
*/
	dimGrid->x = (countOfThreads + block - 1) / block;
	dimGrid->y = 1;
	dimGrid->z = 1;
}

__host__ __device__ inline bool isElim(unsigned char *states) {
	return (*states & 4);
}

__host__ __device__ inline bool isFVis(unsigned char *states) {
	return (*states & 2);
}

__host__ __device__ inline bool isBVis(unsigned char *states) {
	return (*states & 1);
}

__host__ __device__ inline void setElim(unsigned char *states) {
	*states |= 4;
}

__host__ __device__ inline void setFVis(unsigned char *states) {
	*states |= 2;
}

__host__ __device__ inline void setBVis(unsigned char *states) {
	*states |= 1;
}
__host__ __device__ inline void setTrim(unsigned char *states) {
	*states |= 8;
}

__host__ __device__ inline void setPivot(unsigned char *states) {
	*states |= 16;
}

__host__ __device__ inline void setFExt(unsigned char *states) {
	*states |= 64;
}

__host__ __device__ inline void setBExt(unsigned char *states) {
	*states |= 32;
}

__host__ __device__ inline bool isFExt(unsigned char *states) {
	return (*states & 64);
}

__host__ __device__ inline bool isBExt(unsigned char *states) {
	return (*states & 32);
}


#endif
