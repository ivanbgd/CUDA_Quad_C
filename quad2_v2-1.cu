/* Calculates a definite integral by using three different rules.
   Compares sequential to parallel implementations. */

/* My code is written in such a way that the block dimension has to be 1024, because of total loop unrolling. */

/* Version 2 calculates the array elements and performs first level of sum reduction at once.
   This way it makes only 10^9 / 1024, which is around million, transfers from SM cores to global memory, when n = 10^9.
   It also allocates only maximum 512 kB array. It has around million blocks of threads, ie. grid size is around million.
   "Dash 1" uses 1D grid and 1D blocks.
   For some reason, it doesn't want to work with grid dimension larger than 65535, even though it should.
   It says: "Cuda error: kernel invocation: invalid argument.".
   That means we can't deploy 10^9 threads in that configuration, and we have to repeat the kernels 15 times,
   and do the final reduction on the host.
   We will still have around million transfers from registers to global memory, because that is the total number of
   blocks, and we perform reduction at the block level. Only, the array will be at most 512 kB.
   We will split the region that needs to be calculated in 15 subregions, or whatever is needed. We will work on
   the subregions as if they are a complete region, but we will reduce their sums on the host. This way we'll get
   the full area under the curve.
   Faster than v1, plus works with large arrays. Slightly less accurate, but not much. */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_OF_GPU_THREADS 1024		// Size of a block of threads
#define BLOCK_DIM NUM_OF_GPU_THREADS
#define GRID_DIM_MAX_X 65535
#define ACCURACY 0.01

typedef struct Results Results;

struct Results {
	double valQuad;
	double valTrap;
	double valSimp;
	double timeQuad;
	double timeTrap;
	double timeSimp;
};


// The function whose integral we calculate
inline __host__ __device__ double f(const double x) {
	register const double pi = 3.141592653589793;
	double value;
	value = 50.0 / ( pi * ( 2500.0 * x * x + 1.0 ) );
	return value;
}

void checkCUDAError(const char *msg) {
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

/*************************/
/* SEQUENTIAL ALGORITHMS */
/*************************/

// Quadratic rule  
void seqQuad(const unsigned n, const double a, const double b, double *total, double *ExecTime) {
	unsigned i;
	double x;	
	double total_q = 0.0;
	
	// Create events for timing execution
	cudaEvent_t start = cudaEvent_t();
	cudaEvent_t stop = cudaEvent_t();
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Record time into start event
	cudaEventRecord(start, 0);  // 0 is the default stream id	
	
	for (i = 0; i < n; i++) {
		x = ((double)(n - 1 - i)*a + (double)(i)*b) / (double)(n - 1);
		total_q = total_q + f(x);
	}
	total_q = (b - a) * total_q / (double)n;
	
	// Record time into stop event
	cudaEventRecord(stop, 0);

	// Synchronize stop event to wait for end of kernel execution on stream 0
	cudaEventSynchronize(stop);

	// Compute elapsed time (done by CUDA run-time)
	float execTime = 0.f;
	cudaEventElapsedTime(&execTime, start, stop);

	// Release events
	cudaEventDestroy(start);
	cudaEventDestroy(stop);	
	
	*total = total_q;
	*ExecTime = (double)execTime;
}

// Trapezoidal rule
void seqTrap(const unsigned n, const double a, const double b, double *total, double *ExecTime) {
	unsigned i;
	double x;
	const double width = (b - a)/n;
	double total_t = 0.0;
	
	// Create events for timing execution
	cudaEvent_t start = cudaEvent_t();
	cudaEvent_t stop = cudaEvent_t();
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Record time into start event
	cudaEventRecord(start, 0);  // 0 is the default stream id	
	
	{ i = 0; x = a + i*width; total_t += 0.5*f(x); }		// loop peeling
	for (i = 1; i < n - 1; ++i) {
		x = a + i*width;
		total_t = total_t + f(x);
	}
	{ i = n - 1; x = a + i*width; total_t += 0.5*f(x); }	// loop peeling
	total_t = width * total_t;
	
	// Record time into stop event
	cudaEventRecord(stop, 0);

	// Synchronize stop event to wait for end of kernel execution on stream 0
	cudaEventSynchronize(stop);

	// Compute elapsed time (done by CUDA run-time)
	float execTime = 0.f;
	cudaEventElapsedTime(&execTime, start, stop);

	// Release events
	cudaEventDestroy(start);
	cudaEventDestroy(stop);	
	
	*total = total_t;
	*ExecTime = (double)execTime;
}

// Simpson 1/3 rule  
void seqSimp(const unsigned n, const double a, const double b, double *total, double *ExecTime) {
	unsigned i;
	double x;
	const double width = (b - a)/n;
	double total_s = 0.0;
	
	// Create events for timing execution
	cudaEvent_t start = cudaEvent_t();
	cudaEvent_t stop = cudaEvent_t();
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Record time into start event
	cudaEventRecord(start, 0);  // 0 is the default stream id	
	
	{ i = 0; x = a + i*width; total_s = total_s + f(x); }			// loop peeling
	for (i = 1; i < n - 2; ++i) {
		x = a + i*width;
		total_s = total_s + 4*f(x);
		++i;
		x = a + i*width;
		total_s = total_s + 2*f(x);
	}	
	{ i = n - 1; x = a + i*width; total_s = total_s + f(x); }	// loop peeling
	total_s = width / 3 * total_s;
	
	// Record time into stop event
	cudaEventRecord(stop, 0);

	// Synchronize stop event to wait for end of kernel execution on stream 0
	cudaEventSynchronize(stop);

	// Compute elapsed time (done by CUDA run-time)
	float execTime = 0.f;
	cudaEventElapsedTime(&execTime, start, stop);

	// Release events
	cudaEventDestroy(start);
	cudaEventDestroy(stop);	
	
	*total = total_s;
	*ExecTime = (double)execTime;
}

Results sequential(const unsigned n, const double a, const double b) {
	Results results;
	double total_q, total_t, total_s;	// the results
	double wtime_q, wtime_t, wtime_s;	// execution times
	
	seqQuad(n, a, b, &total_q, &wtime_q);
	seqTrap(n, a, b, &total_t, &wtime_t);
	seqSimp(n, a, b, &total_s, &wtime_s);

	results.valQuad = total_q;
	results.valTrap = total_t;
	results.valSimp = total_s;
	results.timeQuad = wtime_q;
	results.timeTrap = wtime_t;
	results.timeSimp = wtime_s;
	
	return results;
}

/***********************/
/* PARALLEL ALGORITHMS */
/***********************/

// Kernel that performs sum reduction.
__global__ void sumReductionKernel(double *arrayDevice, double *sumDevice, const unsigned dim) {
	__shared__ double sdata[BLOCK_DIM];
	unsigned tid = threadIdx.x;
	unsigned i = blockIdx.x*blockDim.x + tid;
	
	// Load block in the shared memory
	if (i < dim) {
		sdata[tid] = arrayDevice[i];
	}
	else {
		sdata[tid] = 0.0;
	}
	
	// Synchronization is necessary after loading of sdata, to make sure that all threads of the block have loaded their element into sdata.
	__syncthreads();

	if (blockDim.x >= 1024) {
		if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads(); }	
	if (blockDim.x >= 512) {
		if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
	if (blockDim.x >= 256) {
		if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
	if (blockDim.x >= 128) {
		if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
	if (tid < 32) {
		// Warp size is 32 threads, so the next instructions don't need synchronization. It's implicitly performed on the warp level. That saves a lot of time.
		if (blockDim.x >= 64) sdata[tid] += sdata[tid + 32];
		if (blockDim.x >= 32) sdata[tid] += sdata[tid + 16];
		if (blockDim.x >= 16) sdata[tid] += sdata[tid + 8];
		if (blockDim.x >= 8) sdata[tid] += sdata[tid + 4];
		if (blockDim.x >= 4) sdata[tid] += sdata[tid + 2];
		if (blockDim.x >= 2) sdata[tid] += sdata[tid + 1];
	}	
	
	// Thread 0 writes result for this block to global mem
	if (tid == 0) {
		sumDevice[blockIdx.x] = sdata[0];
	}
}

/* Quadratic rule */

// This kernel calculates values of f(x), performs one level of reduction, and puts the partial sums into global memory.
__global__ void parQuadKernel(double *arrayDevice, unsigned n, double a, double b) {
	__shared__ double sdata[BLOCK_DIM];
	unsigned tid = threadIdx.x;
	unsigned i = blockIdx.x*blockDim.x + tid;
	
	// Load block in the shared memory
	if (i < n) {
		double x = ((double)(n - 1 - i)*a + (double)(i)*b) / (double)(n - 1);
		sdata[tid] = f(x);
	}
	else {
		sdata[tid] = 0.0;
	}	

	// Synchronization is necessary after loading of sdata, to make sure that all threads of the block have loaded their element into sdata.
	__syncthreads();
	
	if (blockDim.x >= 1024) {
		if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads(); }	
	if (blockDim.x >= 512) {
		if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
	if (blockDim.x >= 256) {
		if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
	if (blockDim.x >= 128) {
		if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
	if (tid < 32) {
		// Warp size is 32 threads, so the next instructions don't need synchronization. It's implicitly performed on the warp level. That saves a lot of time.
		if (blockDim.x >= 64) sdata[tid] += sdata[tid + 32];
		if (blockDim.x >= 32) sdata[tid] += sdata[tid + 16];
		if (blockDim.x >= 16) sdata[tid] += sdata[tid + 8];
		if (blockDim.x >= 8) sdata[tid] += sdata[tid + 4];
		if (blockDim.x >= 4) sdata[tid] += sdata[tid + 2];
		if (blockDim.x >= 2) sdata[tid] += sdata[tid + 1];
	}	
	
	// Thread 0 writes result for this block to global mem
	if (tid == 0) {
		arrayDevice[blockIdx.x] = sdata[0];
	}
}

void parQuad(const unsigned n, const double a, const double b, double *total, double *ExecTime) {
	const double width = (b - a)/n;
	unsigned numBlocksTotal = ceil((double)n / BLOCK_DIM);
	const unsigned numGrids = ceil((double)numBlocksTotal / GRID_DIM_MAX_X);	
	double total_q = 0.0;

	// New values for n, a and b - for the subregions.
	const unsigned npart = ceil((double)n / numGrids);
	const double widthRegion = (b - a)/numGrids;
	double apart = a;
	double bpart = apart + widthRegion;
	
	// Create events for timing execution
	cudaEvent_t start = cudaEvent_t();
	cudaEvent_t stop = cudaEvent_t();
	cudaEventCreate(&start);
	cudaEventCreate(&stop);	

	// Record time into start event
	cudaEventRecord(start, 0);  // 0 is the default stream id
	
	while (numBlocksTotal) {
		unsigned numBlocks = numBlocksTotal < GRID_DIM_MAX_X ? numBlocksTotal : GRID_DIM_MAX_X;
		unsigned newDim = numBlocks;
		double *arrayDevice = NULL;
		const size_t size = numBlocks * sizeof(double);
		double *sumDevice = NULL;
		size_t sumSize;
		
		numBlocksTotal -= numBlocks;
		
		// Allocate memory on the GPU for the array that holds values of partial sums, after first level of reduction
		cudaMalloc((void **) &arrayDevice, size);		
		
		// Launch kernel that calculates values of f(x) and performs one level of reduction
		parQuadKernel<<< numBlocks, BLOCK_DIM >>>(arrayDevice, npart, apart, bpart);
		
		apart += (widthRegion + width);
		bpart += (widthRegion + width);
		
		numBlocks = ceil((double)newDim / BLOCK_DIM);
		sumSize = numBlocks * sizeof(double);
		cudaMalloc((void **) &sumDevice, sumSize);		
		
		// Launch kernel that performs sum reduction
		sumReductionKernel<<< numBlocks, BLOCK_DIM >>>(arrayDevice, sumDevice, newDim);

		while (numBlocks > 1) {
			newDim = numBlocks;
			numBlocks = ceil((double)newDim / BLOCK_DIM);
			sumReductionKernel<<< numBlocks, BLOCK_DIM >>>(sumDevice, sumDevice, newDim);
		}
		
		// Copy results back to the host
		double result;
		cudaMemcpy(&result, sumDevice, sizeof(double), cudaMemcpyDeviceToHost);

		total_q += result;	

		// Free CUDA memory
		cudaFree(sumDevice);
		cudaFree(arrayDevice);		
	}
	
	total_q = (b - a) * total_q / (double)n;

	// Record time into stop event
	cudaEventRecord(stop, 0);

	// Synchronize stop event to wait for end of kernel execution on stream 0
	cudaEventSynchronize(stop);

	// Compute elapsed time (done by CUDA run-time)
	float elapsed = 0.f;
	cudaEventElapsedTime(&elapsed, start, stop);

	// Release events
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	*total = total_q;
	*ExecTime = (double)elapsed;
}

/* Trapezoidal rule */

// This kernel calculates values of f(x), performs one level of reduction, and puts the partial sums into global memory.
__global__ void parTrapKernel(double *arrayDevice, unsigned n, double a, double width) {
	__shared__ double sdata[BLOCK_DIM];
	unsigned tid = threadIdx.x;
	unsigned i = blockIdx.x*blockDim.x + tid;
	
	// Load block in the shared memory
	if (i < n) {
		double x = a + i*width;
		if (i == 0 || i == n - 1)		
			sdata[tid] = 0.5*f(x);
		else
			sdata[tid] = f(x);		
	}
	else {
		sdata[tid] = 0.0;
	}	

	// Synchronization is necessary after loading of sdata, to make sure that all threads of the block have loaded their element into sdata.
	__syncthreads();
	
	if (blockDim.x >= 1024) {
		if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads(); }	
	if (blockDim.x >= 512) {
		if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
	if (blockDim.x >= 256) {
		if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
	if (blockDim.x >= 128) {
		if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
	if (tid < 32) {
		// Warp size is 32 threads, so the next instructions don't need synchronization. It's implicitly performed on the warp level. That saves a lot of time.
		if (blockDim.x >= 64) sdata[tid] += sdata[tid + 32];
		if (blockDim.x >= 32) sdata[tid] += sdata[tid + 16];
		if (blockDim.x >= 16) sdata[tid] += sdata[tid + 8];
		if (blockDim.x >= 8) sdata[tid] += sdata[tid + 4];
		if (blockDim.x >= 4) sdata[tid] += sdata[tid + 2];
		if (blockDim.x >= 2) sdata[tid] += sdata[tid + 1];
	}	
	
	// Thread 0 writes result for this block to global mem
	if (tid == 0) {
		arrayDevice[blockIdx.x] = sdata[0];
	}	
}

void parTrap(const unsigned n, const double a, const double b, double *total, double *ExecTime) {
	const double width = (b - a)/n;
	unsigned numBlocksTotal = ceil((double)n / BLOCK_DIM);
	const unsigned numGrids = ceil((double)numBlocksTotal / GRID_DIM_MAX_X);	
	double total_t = 0.0;

	// New values for n, a and b - for the subregions.
	const unsigned npart = ceil((double)n / numGrids);
	const double widthRegion = (b - a)/numGrids;
	double apart = a;
	
	// Create events for timing execution
	cudaEvent_t start = cudaEvent_t();
	cudaEvent_t stop = cudaEvent_t();
	cudaEventCreate(&start);
	cudaEventCreate(&stop);	
	
	// Record time into start event
	cudaEventRecord(start, 0);  // 0 is the default stream id

	while (numBlocksTotal) {
		unsigned numBlocks = numBlocksTotal < GRID_DIM_MAX_X ? numBlocksTotal : GRID_DIM_MAX_X;
		unsigned newDim = numBlocks;
		double *arrayDevice = NULL;
		const size_t size = numBlocks * sizeof(double);
		double *sumDevice = NULL;
		size_t sumSize;
		
		numBlocksTotal -= numBlocks;
		
		// Allocate memory on the GPU for the array that holds values of partial sums, after first level of reduction
		cudaMalloc((void **) &arrayDevice, size);		
		
		// Launch kernel that calculates values of f(x) and performs one level of reduction
		parTrapKernel<<< numBlocks, BLOCK_DIM >>>(arrayDevice, npart, apart, width);
		
		apart += (widthRegion + width);
		
		numBlocks = ceil((double)newDim / BLOCK_DIM);
		sumSize = numBlocks * sizeof(double);
		cudaMalloc((void **) &sumDevice, sumSize);		
		
		// Launch kernel that performs sum reduction
		sumReductionKernel<<< numBlocks, BLOCK_DIM >>>(arrayDevice, sumDevice, newDim);

		while (numBlocks > 1) {
			newDim = numBlocks;
			numBlocks = ceil((double)newDim / BLOCK_DIM);
			sumReductionKernel<<< numBlocks, BLOCK_DIM >>>(sumDevice, sumDevice, newDim);
		}
		
		// Copy results back to the host
		double result;
		cudaMemcpy(&result, sumDevice, sizeof(double), cudaMemcpyDeviceToHost);

		total_t += result;	

		// Free CUDA memory
		cudaFree(sumDevice);
		cudaFree(arrayDevice);		
	}
	
	total_t = width * total_t;

	// Record time into stop event
	cudaEventRecord(stop, 0);

	// Synchronize stop event to wait for end of kernel execution on stream 0
	cudaEventSynchronize(stop);

	// Compute elapsed time (done by CUDA run-time)
	float elapsed = 0.f;
	cudaEventElapsedTime(&elapsed, start, stop);

	// Release events
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	*total = total_t;
	*ExecTime = (double)elapsed;
}

/* Simpson 1/3 rule   */

// This kernel calculates values of f(x), and puts them into global memory.
__global__ void parSimpKernel(double *arrayDevice, unsigned n, double a, double width) {
	__shared__ double sdata[BLOCK_DIM];
	unsigned tid = threadIdx.x;
	unsigned i = blockIdx.x*blockDim.x + tid;
	
	// Load block in the shared memory
	if (i < n) {
		double x = a + i*width;
		if (i == 0 || i == n - 1) {
			sdata[tid] = f(x);
		}
		else {
			if (i % 2 == 1)
				sdata[tid] = 4*f(x);
			else
				sdata[tid] = 2*f(x);			
		}
	}
	else {
		sdata[tid] = 0.0;
	}	

	// Synchronization is necessary after loading of sdata, to make sure that all threads of the block have loaded their element into sdata.
	__syncthreads();
	
	if (blockDim.x >= 1024) {
		if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads(); }	
	if (blockDim.x >= 512) {
		if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
	if (blockDim.x >= 256) {
		if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
	if (blockDim.x >= 128) {
		if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
	if (tid < 32) {
		// Warp size is 32 threads, so the next instructions don't need synchronization. It's implicitly performed on the warp level. That saves a lot of time.
		if (blockDim.x >= 64) sdata[tid] += sdata[tid + 32];
		if (blockDim.x >= 32) sdata[tid] += sdata[tid + 16];
		if (blockDim.x >= 16) sdata[tid] += sdata[tid + 8];
		if (blockDim.x >= 8) sdata[tid] += sdata[tid + 4];
		if (blockDim.x >= 4) sdata[tid] += sdata[tid + 2];
		if (blockDim.x >= 2) sdata[tid] += sdata[tid + 1];
	}	
	
	// Thread 0 writes result for this block to global mem
	if (tid == 0) {
		arrayDevice[blockIdx.x] = sdata[0];
	}
}

void parSimp(const unsigned n, const double a, const double b, double *total, double *ExecTime) {
	const double width = (b - a)/n;
	unsigned numBlocksTotal = ceil((double)n / BLOCK_DIM);
	const unsigned numGrids = ceil((double)numBlocksTotal / GRID_DIM_MAX_X);		
	double total_s = 0.0;

	// New values for n, a and b - for the subregions.
	const unsigned npart = ceil((double)n / numGrids);
	const double widthRegion = (b - a)/numGrids;
	double apart = a;	

	// Create events for timing execution
	cudaEvent_t start = cudaEvent_t();
	cudaEvent_t stop = cudaEvent_t();
	cudaEventCreate(&start);
	cudaEventCreate(&stop);	
	
	// Record time into start event
	cudaEventRecord(start, 0);  // 0 is the default stream id

	while (numBlocksTotal) {
		unsigned numBlocks = numBlocksTotal < GRID_DIM_MAX_X ? numBlocksTotal : GRID_DIM_MAX_X;
		unsigned newDim = numBlocks;
		double *arrayDevice = NULL;
		const size_t size = numBlocks * sizeof(double);
		double *sumDevice = NULL;
		size_t sumSize;
		
		numBlocksTotal -= numBlocks;
		
		// Allocate memory on the GPU for the array that holds values of partial sums, after first level of reduction
		cudaMalloc((void **) &arrayDevice, size);		
		
		// Launch kernel that calculates values of f(x) and performs one level of reduction
		parSimpKernel<<< numBlocks, BLOCK_DIM >>>(arrayDevice, npart, apart, width);
		
		apart += (widthRegion + width);
		
		numBlocks = ceil((double)newDim / BLOCK_DIM);
		sumSize = numBlocks * sizeof(double);
		cudaMalloc((void **) &sumDevice, sumSize);		
		
		// Launch kernel that performs sum reduction
		sumReductionKernel<<< numBlocks, BLOCK_DIM >>>(arrayDevice, sumDevice, newDim);

		while (numBlocks > 1) {
			newDim = numBlocks;
			numBlocks = ceil((double)newDim / BLOCK_DIM);
			sumReductionKernel<<< numBlocks, BLOCK_DIM >>>(sumDevice, sumDevice, newDim);
		}
		
		// Copy results back to the host
		double result;
		cudaMemcpy(&result, sumDevice, sizeof(double), cudaMemcpyDeviceToHost);

		total_s += result;	

		// Free CUDA memory
		cudaFree(sumDevice);
		cudaFree(arrayDevice);		
	}
	
	total_s = width / 3 * total_s;

	// Record time into stop event
	cudaEventRecord(stop, 0);

	// Synchronize stop event to wait for end of kernel execution on stream 0
	cudaEventSynchronize(stop);

	// Compute elapsed time (done by CUDA run-time)
	float elapsed = 0.f;
	cudaEventElapsedTime(&elapsed, start, stop);

	// Release events
	cudaEventDestroy(start);
	cudaEventDestroy(stop);	

	*total = total_s;
	*ExecTime = (double)elapsed;
}

Results parallel(const unsigned n, const double a, const double b) {
	Results results;
	
	parQuad(n, a, b, &results.valQuad, &results.timeQuad);
	parTrap(n, a, b, &results.valTrap, &results.timeTrap);
	parSimp(n, a, b, &results.valSimp, &results.timeSimp);	
	
	return results;
}

void compareAndPrint(const unsigned n, const double a, const double b) {
	Results seq, par;
	
	seq = sequential(n, a, b);
	par = parallel(n, a, b);	

	printf("  Sequential estimate quadratic rule   = %24.16f\n", seq.valQuad);
	printf("  Parallel estimate quadratic rule     = %24.16f\n", par.valQuad);
	printf("Sequential time quadratic rule   = %f ms\n", seq.timeQuad);
	printf("Parallel time quadratic rule     = %f ms\n", par.timeQuad);	
	if (fabs(seq.valQuad - par.valQuad) < ACCURACY)
		printf("\tTest PASSED!\n");
	else
		printf("\a\tTest FAILED!!!\n");
	printf ("\n");
	
	printf("  Sequential estimate trapezoidal rule = %24.16f\n", seq.valTrap);
	printf("  Parallel estimate trapezoidal rule   = %24.16f\n", par.valTrap);
	printf("Sequential time trapezoidal rule = %f ms\n", seq.timeTrap);
	printf("Parallel time trapezoidal rule   = %f ms\n", par.timeTrap);	
	if (fabs(seq.valTrap - par.valTrap) < ACCURACY)
		printf("\tTest PASSED!\n");
	else
		printf("\a\tTest FAILED!!!\n");	
	printf ("\n");
	
	printf("  Sequential estimate Simpson 1/3 rule = %24.16f\n", seq.valSimp);
	printf("  Parallel estimate Simpson 1/3 rule   = %24.16f\n", par.valSimp);
	printf("Sequential time Simpson 1/3 rule = %f ms\n", seq.timeSimp);
	printf("Parallel time Simpson 1/3 rule   = %f ms\n", par.timeSimp);	
	if (fabs(seq.valSimp - par.valSimp) < ACCURACY)
		printf("\tTest PASSED!\n");
	else
		printf("\a\tTest FAILED!!!\n");	
	printf ("\n");
}

int main(int argc, char *argv[]) {
	unsigned n;
	double a;
	double b; 

	if (argc != 4) {
		n = 10000000;
		a = 0.0;
		b = 10.0;
	}
	else {
		n = (unsigned)atoi(argv[1]);
		a = atof(argv[2]);
		b = atof(argv[3]);
	}

	printf("\n");
	printf("QUAD:\n");
	printf("  Estimate the integral of f(x) from A to B.\n");
	printf("  f(x) = 50 / ( pi * ( 2500 * x * x + 1 ) ).\n");
	printf("\n");
	printf("  A        = %f\n", a);
	printf("  B        = %f\n", b);
	printf("  N        = %u\n", n);
	printf("\n");
	
	// We can add this for correct time measurement in the nvprof profiler.
	cudaDeviceSynchronize();

	compareAndPrint(n, a, b);

	printf("\n  Normal end of execution.\n");
	printf("\n");

	return 0;	
}