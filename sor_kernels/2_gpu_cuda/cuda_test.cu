#include <cstdio>
#include <cstdlib>
#include <math.h>

// Assertion to check for errors
//Wraps around CUDA function and terminates when an error is detected
#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"CUDA_SAFE_CALL: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

//Used in launching kernals
#define NUM_THREADS_PER_BLOCK 	256
#define NUM_BLOCKS 				16
#define PRINT_TIME 				1 		//Boolean for measuring/printing time
#define SM_ARR_LEN				50000	//Default array size can be specified on command line
#define TOL						1e-6	//Wiggle room for epsilon on GPU

#define IMUL(a, b) __mul24(a, b)	//CUDA intrinsic multiply of least significant 24-bits

void initializeArray1D(float *arr, int len, int seed);	//Initializes an array with random floats

//CUDA kernel!
__global__ void kernel_add (int arrLen, float* x, float* y, float* result) {
	//blockDim gives the length of the dimension, Idx gives position in the grid
	//ThreadIdx gives position within block
	//gridDim gives the number of blocks in the grid
	const int tid = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;		//Unique threadID
	const int threadN = IMUL(blockDim.x, gridDim.x);	//Total number of threads
	
	int i;
	
	//Loop for every element modulo thread ID
	for(i = tid; i < arrLen; i += threadN) {
		result[i] = (1e-6 * x[i] ) + (1e-7 * y[i]) + 0.25;
	}
}

int main(int argc, char **argv){
	int arrLen = 0;
		
	// GPU Timing variables
	cudaEvent_t start, stop;
	float elapsed_gpu;
	
	// Arrays on GPU global memory
	float *d_x;
	float *d_y;
	float *d_result;

	// Arrays on the host memory
	float *h_x;
	float *h_y;
	float *h_result;
	float *h_result_gold;
	
	int i, errCount = 0, zeroCount = 0;
	
	if (argc > 1) {
		arrLen  = atoi(argv[1]);
	}
	else {
		arrLen = SM_ARR_LEN;
	}

	printf("Length of the array = %d\n", arrLen);

	// Allocate GPU memory
	size_t allocSize = arrLen * sizeof(float);
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_x, allocSize));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_y, allocSize));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_result, allocSize));
		
	// Allocate arrays on host memory
	h_x                        = (float *) malloc(allocSize);
	h_y                        = (float *) malloc(allocSize);
	h_result                   = (float *) malloc(allocSize);
	h_result_gold              = (float *) malloc(allocSize);
	
	// Initialize the host arrays
	printf("\nInitializing the arrays ...");
	// Arrays are initialized with a known seed for reproducability
	initializeArray1D(h_x, arrLen, 2453);
	initializeArray1D(h_y, arrLen, 1467);
	printf("\t... done\n\n");
	
	
#if PRINT_TIME
	// Create the cuda events
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// Record event on the default stream
	cudaEventRecord(start, 0);
#endif
	
	// Transfer the arrays to the GPU memory
	CUDA_SAFE_CALL(cudaMemcpy(d_x, h_x, allocSize, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_y, h_y, allocSize, cudaMemcpyHostToDevice));
	  
	// Launch the kernel
	//Declare a 1D grid and 1D blocks
	kernel_add<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(arrLen, d_x, d_y, d_result);

	// Check for errors during launch
	//cudaPeek explicitly returns the error
	CUDA_SAFE_CALL(cudaPeekAtLastError());
	
	// Transfer the results back to the host
	CUDA_SAFE_CALL(cudaMemcpy(h_result, d_result, allocSize, cudaMemcpyDeviceToHost));
	
#if PRINT_TIME
	// Stop and destroy the timer
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_gpu, start, stop);
	printf("\nGPU time: %f (msec)\n", elapsed_gpu);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
#endif
	
	// Compute the results on the host
	for(i = 0; i < arrLen; i++) {
		h_result_gold[i] = (1e-6 * h_x[i]) + (1e-7 * h_y[i]) + 0.25;
	}
	
	// Compare the results to see if the GPU messed up
	for(i = 0; i < arrLen; i++) {
		//TOL required because GPU's are not precise
		if (abs(h_result_gold[i] - h_result[i]) > TOL) {
			errCount++;
		}
		if (h_result[i] == 0) {
			zeroCount++;
		}
	}
	
	//Print the results for debugging
	/*
	for(i = 0; i < 50; i++) {
		printf("%d:\t%.8f\t%.8f\n", i, h_result_gold[i], h_result[i]);
	}
	*/
	
	if (errCount > 0) {
		printf("\n@ERROR: TEST FAILED: %d results did not matched\n", errCount);
	}
	else if (zeroCount > 0){
		printf("\n@ERROR: TEST FAILED: %d results (from GPU) are zero\n", zeroCount);
	}
	else {
		printf("\nTEST PASSED: All results matched\n");
	}
	
	// Free-up device and host memory
	CUDA_SAFE_CALL(cudaFree(d_x));
	CUDA_SAFE_CALL(cudaFree(d_y));
	CUDA_SAFE_CALL(cudaFree(d_result));
		   
	free(h_x);
	free(h_y);
	free(h_result);
		
	return 0;
}

void initializeArray1D(float *arr, int len, int seed) {
	int i;
	float randNum;
	srand(seed);

	for (i = 0; i < len; i++) {
		randNum = (float) rand();
		arr[i] = randNum;
	}
}
