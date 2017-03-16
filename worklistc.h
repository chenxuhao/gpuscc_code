#pragma once

#include "cutil_subset.h"
#include "cub/cub.cuh"

static unsigned zero = 0;

struct Worklist1 {
	unsigned *dwl, *wl;
	unsigned length, *dnsize;
	unsigned *dindex;

	Worklist1(size_t nsize)
	{
		//printf("nsize=%d ", nsize);
		//wl = (unsigned *) calloc(nsize, sizeof(unsigned));
		CUDA_SAFE_CALL(cudaMalloc(&dwl, nsize * sizeof(unsigned)));
		CUDA_SAFE_CALL(cudaMalloc(&dnsize, 1 * sizeof(unsigned)));
		CUDA_SAFE_CALL(cudaMalloc(&dindex, 1 * sizeof(unsigned)));

		CUDA_SAFE_CALL(cudaMemcpy(dnsize, &nsize, 1 * sizeof(unsigned), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy((void *) dindex, &zero, 1 * sizeof(zero), cudaMemcpyHostToDevice));
		//CUDA_SAFE_CALL(cudaMemcpy(&length, dnsize, 1 * sizeof(unsigned), cudaMemcpyDeviceToHost));
		//printf("length=%d\tdindex=%p\n", length, dindex);
	}

	~Worklist1()
	{
		//CUDA_SAFE_CALL(cudaFree(dwl));
	}

	void dealloc()
	{
		CUDA_SAFE_CALL(cudaFree(dwl));
		CUDA_SAFE_CALL(cudaFree(dnsize));
		CUDA_SAFE_CALL(cudaFree(dindex));
	}

	void update_cpu()
	{
		//unsigned nsize = nitems();
		unsigned nsize = getSize();
		CUDA_SAFE_CALL(cudaMemcpy(wl, dwl, nsize  * sizeof(unsigned), cudaMemcpyDeviceToHost));
	}

	void display_items()
	{
		//unsigned nsize = nitems();
		unsigned nsize = getSize();
		CUDA_SAFE_CALL(cudaMemcpy(wl, dwl, nsize  * sizeof(unsigned), cudaMemcpyDeviceToHost));

		printf("WL: ");
		for(unsigned i = 0; i < nsize; i++)
			printf("%d %d, ", i, wl[i]);

		printf("\n");
		return;
	}

	//void reset()
	void clearHost()
	{
		CUDA_SAFE_CALL(cudaMemcpy((void *) dindex, &zero, 1 * sizeof(zero), cudaMemcpyHostToDevice));
	}

	//unsigned nitems()
	unsigned getSize()
	{
		unsigned index;
		//printf("&dindex=%p &index=%p\n", dindex, &index);
		CUDA_SAFE_CALL(cudaMemcpy(&index, /*(void *)*/ dindex, 1 * sizeof(index), cudaMemcpyDeviceToHost));

		return index;
	}

	__device__ 
	unsigned push(unsigned item)
	{
		unsigned lindex = atomicAdd((unsigned *) dindex, 1);
		if(lindex >= *dnsize)
			return 0;

		dwl[lindex] = item;
		return 1;
	}

	unsigned pushHost(unsigned item)
	{
		unsigned index;
		CUDA_SAFE_CALL(cudaMemcpy(&index, dindex, sizeof(*dindex), cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(&dwl[index], &item, sizeof(item), cudaMemcpyHostToDevice));
		index++;
		CUDA_SAFE_CALL(cudaMemcpy(dindex, &index, sizeof(index), cudaMemcpyHostToDevice));
		return 1;
	}


	//__device__
	#if 0
	unsigned pushnitems(unsigned n_items, unsigned *items)
	{
		unsigned index = nitems();
		for (unsigned i = 0; i < n_items; i++)
			wl[index + i] = items[i];

		//unsigned length = index + n_items;
		index += n_items;
		cudaMemcpy(dindex, &index, sizeof(unsigned), cudaMemcpyHostToDevice);
		cudaMemcpy(dwl, wl, index * sizeof(unsigned), cudaMemcpyHostToDevice);	
		return n_items;
	}
	#endif

	__device__
	unsigned pop(unsigned &item)
	{
		unsigned lindex = atomicSub((unsigned *) dindex, 1);
		if(lindex <= 0)
		{
			*dindex = 0;
			return 0;
		}

		item = dwl[lindex - 1];
		return 1;
	}
};

struct Worklist2: public Worklist1 {
Worklist2(unsigned nsize) : Worklist1(nsize) {}

	template <typename T>
	__device__ __forceinline__
	unsigned push_1item(unsigned nitem, unsigned item, unsigned threads_per_block)
	{
		__shared__ typename T::TempStorage temp_storage;
		__shared__ unsigned queue_index;
		unsigned total_items = 0;
		unsigned thread_data = nitem;


		T(temp_storage).ExclusiveSum(thread_data, thread_data, total_items);
		__syncthreads();
		//printf("threadIdx.x=%d\ttotal_items=%d\tnitem=%d\tthread_data=%d\n", threadIdx.x, total_items, nitem, thread_data);

		if(threadIdx.x == 0)
		{	
			//if(debug) printf("t: %d\n", total_items);
			//printf("blockIdx.x=%d\tdindex=%d\ttotal_items=%d\n", blockIdx.x, *dindex, total_items);
			queue_index = atomicAdd((unsigned *) dindex, total_items);
			//printf("blockIdx.x=%d\tdindex=%d\ttotal_items=%d\n", blockIdx.x, *dindex, total_items);
			//printf("queueindex: %d %d %d %d %d\n", blockIdx.x, threadIdx.x, queue_index, thread_data + n_items, total_items);
		}

		__syncthreads();

		if(nitem == 1)
		{
			if(queue_index + thread_data >= *dnsize)
			{
				printf("GPU: exceeded length: %u %u %u %u\n", queue_index, thread_data, total_items, *dnsize);
				return 0;
			}

			//dwl[queue_index + thread_data] = item;
			cub::ThreadStore<cub::STORE_CG>(dwl + queue_index + thread_data, item);
		}

		//printf("threadIdx.x=%d\tnitem=%d\tthread_data=%d\n", threadIdx.x, nitem, thread_data);
		__syncthreads();
		return total_items;
	}

	template <typename T>
	__device__ __forceinline__
	unsigned push_nitems(unsigned n_items, unsigned *items, unsigned threads_per_block)
	{
		__shared__ typename T::TempStorage temp_storage;
		__shared__ unsigned queue_index;
		unsigned total_items;

		unsigned thread_data = n_items;

		T(temp_storage).ExclusiveSum(thread_data, thread_data, total_items);

		if(threadIdx.x == 0)
		{	
			queue_index = atomicAdd((unsigned *) dindex, total_items);
			//printf("queueindex: %d %d %d %d %d\n", blockIdx.x, threadIdx.x, queue_index, thread_data + n_items, total_items);
		}

		__syncthreads();


		for(unsigned i = 0; i < n_items; i++)
		{
			//printf("pushing %d to %d\n", items[i], queue_index + thread_data + i);
			if(queue_index + thread_data + i >= *dnsize)
			{
				printf("GPU: exceeded length: %u %u %u %u\n", queue_index, thread_data, i, *dnsize);
				return 0;
			}

			dwl[queue_index + thread_data + i] = items[i];
		}

		return total_items;
	}

	__device__ 
	unsigned pop_id(unsigned id, unsigned &item)
	{
		if(id < *dindex)
		{
			item = cub::ThreadLoad<cub::LOAD_CG>(dwl + id);
			//item = dwl[id];
			return 1;
		}

		return 0;
	}
};

