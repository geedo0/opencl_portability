#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <time.h>
#include "cuPrintf.cu"
#include "cuPrintf.cuh"

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
#define PRINT_TIME 				1 		//Boolean for measuring/printing time
#define SM_ARR_LEN				2000	//Default array size can be specified on command line
#define TOL						.00001
#define EPSILON					.05
#define GIG 					1000000000
#define IMUL(a, b) __mul24(a, b)	//CUDA intrinsic multiply of least significant 24-bits
#define MAT_INDEX(r,c,l)		(r*l+c)


void initializeArray2D(float *arr, int len, int seed);	//Initializes an array with random floats
void SOR_blocked(float* data, int length);
struct timespec diff(struct timespec start, struct timespec end);

#define THREAD_DIM	125
__global__ void kernel_sor__single(int N, float* A, float* B) {
	//Each thread is responsible for a 125x125 element square
	//These constants indicate the origin of that square
	const int baseRow = IMUL(threadIdx.y, THREAD_DIM);
	const int baseCol = IMUL(threadIdx.x, THREAD_DIM);
	const int rowN = IMUL(blockDim.x, THREAD_DIM);

	//Each loop iteration is two SOR computations
	for(int i = 0; i < 1000; i++) {
		//Each thread operates on a 125*125 square
		//Transfers go A to B then B to A, the final result is stored in A
		//j iterates over columns, k iterates over rows
		for(int j = baseRow; j < baseRow + THREAD_DIM; j++) {
			for(int k = baseCol; k < baseCol + THREAD_DIM; k++) {
				//Check for a boundary element
				if(j>0 && j<(N-1) && k>0 && k<(N-1)) {
					B[j*rowN+k] = (
						A[j*rowN + k-1] +
						A[j*rowN + k] +
						A[j*rowN + k+1] +
						A[(j-1)*rowN + k] +
						A[(j+1)*rowN + k]) * .2;
				}
				else {	//Boundary, element doesn't change
					B[j*rowN+k] = A[j*rowN+k];
				}
			}
		}
		//Sync threads after each SOR to ensure clean data for next phase
		__syncthreads();
		for(int j = baseRow; j < baseRow + THREAD_DIM; j++) {
			for(int k = baseCol; k < baseCol + THREAD_DIM; k++) {
				//Check for a boundary element
				if(j>0 && j<(N-1) && k>0 && k<(N-1)) {
					A[j*rowN+k] = (
						B[j*rowN + k-1] +
						B[j*rowN + k] +
						B[j*rowN + k+1] +
						B[(j-1)*rowN + k] +
						B[(j+1)*rowN + k]) * .2;
				}
				else {	//Boundary, element doesn't change
					A[j*rowN+k] = B[j*rowN+k];
				}
			}
		}
		__syncthreads();
	}
}

//Compute a single iteration of SOR with separate input/output arrays
//A to B kernel
__global__ void kernel_sor_1(int N, float* A, float* B) {
	//BlockDim = dimensions of block in threads
	//GridDim = dimensions of grid in blocks
	//BlockIdx = position w/n grid
	//ThreadIdx = position w/n block
	const int row = IMUL(blockDim.y, blockIdx.y) + threadIdx.y;
	const int col = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
	const int rowN = IMUL(blockDim.x, gridDim.x);
	const int tid = IMUL(rowN, row) + col;

	#ifdef DEBUG
	cuPrintf("bx: %d by: %d tx: %d ty: %d, row: %d, col: %d tid %d \n"
		, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, row, col, tid);
	#endif

	//If the row and col values are not boundary elements, compute the average
	if(row > 0 && row < (N-1) && col > 0 && col < (N-1)) {
		B[row*rowN+col] = (
			A[row*rowN + col-1] +
			A[row*rowN + col] +
			A[row*rowN + col+1] +
			A[(row-1)*rowN + col] +
			A[(row+1)*rowN + col]) * .2;
	}
	//else the element does not change
	else {
		B[row*rowN + col] = A[row*rowN + col];
	}
}

//B to A kernel
//Dual to allow alternating input/output arrays
__global__ void kernel_sor_2(int N, float* A, float* B) {
	//BlockDim = dimensions of block in threads
	//GridDim = dimensions of grid in blocks
	//BlockIdx = position w/n grid
	//ThreadIdx = position w/n block
	const int row = IMUL(blockDim.y, blockIdx.y) + threadIdx.y;
	const int col = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
	const int rowN = IMUL(blockDim.x, gridDim.x);
	const int tid = IMUL(rowN, row) + col;

	#ifdef DEBUG
	cuPrintf("bx: %d by: %d tx: %d ty: %d, row: %d, col: %d tid %d \n"
		, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, row, col, tid);
	#endif

	//If the row and col values are not boundary elements, compute the average
	if(row > 0 && row < (N-1) && col > 0 && col < (N-1)) {
		A[row*rowN+col] = (
			B[row*rowN + col-1] +
			B[row*rowN + col] +
			B[row*rowN + col+1] +
			B[(row-1)*rowN + col] +
			B[(row+1)*rowN + col]) * .2;
	}
	//else the element does not change
	else {
		A[row*rowN + col] = B[row*rowN + col];
	}
}

int main(int argc, char **argv){
	struct timespec time1, time2, delta;

	int arrLen = 0;
		
	// GPU Timing variables
	cudaEvent_t start, stop;
	float elapsed_gpu;
	
	// Arrays on GPU global memory
	float *d_A;
	float *d_B;

	// Arrays on the host memory
	float *h_input;
	float *h_result;
	
	int i, j, errCount = 0, zeroCount = 0;
	
	if (argc > 1) {
		arrLen  = atoi(argv[1]);
	}
	else {
		arrLen = SM_ARR_LEN;
	}

	#ifdef DEBUG
	printf("Length of the array = %d\n", arrLen);
	#endif

	// Allocate GPU memory
	size_t allocSize = arrLen * arrLen * sizeof(float);		//Create a square matrix
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_A, allocSize));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_B, allocSize));
		
	// Allocate arrays on host memory
	h_input                        = (float *) malloc(allocSize);
	h_result                   = (float *) malloc(allocSize);
	
	#ifdef DEBUG// Initialize the host arrays
	printf("\nInitializing the arrays ...");
	#endif
	// Arrays are initialized with a known seed for reproducability
	initializeArray2D(h_input, arrLen, 2453);
	#ifdef DEBUG
	printf("\t... done\n\n");
	#endif
	
	#ifdef DEBUG
	cudaPrintfInit();
	#endif
#if PRINT_TIME
	// Create the cuda events
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// Record event on the default stream
	cudaEventRecord(start, 0);
#endif
	
	// Transfer the arrays to the GPU memory
	CUDA_SAFE_CALL(cudaMemcpy(d_A, h_input, allocSize, cudaMemcpyHostToDevice));
	  
	 /*
	//Single block implementation
	dim3 DimBlock(16,16,1);
	kernel_sor__single<<<1, DimBlock>>>(arrLen, d_A, d_B);
	*/

	//Multiple block implementation
	dim3 DimBlock(16, 16, 1);
	dim3 DimGrid(125,125,1);
	//2000 SOR iterations, results of final iteration are stored on d_A
	for(i = 0; i < 1000; i++) {
		kernel_sor_1<<<DimGrid, DimBlock>>>(arrLen, d_A, d_B);
		cudaDeviceSynchronize();
		kernel_sor_2<<<DimGrid, DimBlock>>>(arrLen, d_A, d_B);
		cudaDeviceSynchronize();
	}


	// Check for errors during launch
	//cudaPeek explicitly returns the error
	CUDA_SAFE_CALL(cudaPeekAtLastError());
	
	// Transfer the results back to the host
	CUDA_SAFE_CALL(cudaMemcpy(h_result, d_A, allocSize, cudaMemcpyDeviceToHost));
	
#if PRINT_TIME
	// Stop and destroy the timer
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_gpu, start, stop);
	printf("\nGPU time: %f (msec)\n", elapsed_gpu);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
#endif
	#ifdef DEBUG
	cudaDeviceSynchronize();
	cudaPrintfDisplay(stdout, true);
	cudaPrintfEnd();
	#endif
	
	// Compute the results on the host, this changes stores the results in h_input
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
	SOR_blocked(h_input, arrLen);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
    delta = diff(time1, time2);
    printf("CPU time: %ld (nsec)", (long int) (GIG * delta.tv_sec + delta.tv_nsec));
	
	// Compare the results to see if the GPU messed up
	float error, percentError;
	//Good error is when the result falls within epsilon, bad error is when it doesn't.
	double totalError=0, goodError=0, badError=0;
	for(i = 0; i < arrLen; i++) {
		for(j = 0; j < arrLen; j++) {
			//TOL required because GPU's are not precise
			error = abs(h_input[MAT_INDEX(i,j,arrLen)] - h_result[MAT_INDEX(i,j,arrLen)]);
			percentError = error/(h_input[MAT_INDEX(i,j,arrLen)]);
			if (percentError > EPSILON) {
				errCount++;
				fprintf(stderr, "row: %d col: %d gpu: %f cpu: %f eps: %f\n",
					i, j, h_result[MAT_INDEX(i,j,arrLen)], h_input[MAT_INDEX(i,j,arrLen)], error);
				badError += (double) error;
			}
			else {
				goodError += (double) error;
			}
			if (h_result[MAT_INDEX(i,j,arrLen)] == 0) {
				zeroCount++;
			}
		}
	}
	totalError = goodError + badError;
	
	int passed = 1;
	int elements = arrLen*arrLen;
	if (errCount > 0) {
		passed = 0;
		printf("\n@ERROR: TEST FAILED: %d results did not matched.", errCount);
		printf("\nAverage error: %f, Good error: %f Bad error: %f",
			totalError/(elements),
			goodError/(elements-errCount),
			badError/errCount);
	}
	if(zeroCount > 0) {
		passed = 0;
		printf("\n@ERROR: TEST FAILED: %d results (from GPU) are zero\n", zeroCount);
	}
	if(passed)
		printf("\nTEST PASSED: All results matched\n");
	
	// Free-up device and host memory
	CUDA_SAFE_CALL(cudaFree(d_A));
	CUDA_SAFE_CALL(cudaFree(d_B));
		   
	free(h_input);
	free(h_result);
		
	return 0;
}


#define MINVAL   0.0
#define MAXVAL  10.0

double fRand(double fMin, double fMax)
{
    double f = ((double) rand()) / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

void initializeArray2D(float *arr, int len, int seed) {
	int i;
	float randNum;
	srand(seed);
	for (i = 0; i < len * len; i++) {
		randNum = (float) fRand((double)(MINVAL),(double)(MAXVAL));
		arr[i] = randNum;
	}
}

//Optimized sequential algorithm
#define BLOCK_SIZE	2
#define OMEGA 		1.9759		//Tuned for length 2000
void SOR_blocked(float* data, int length)
{
  long int i, j, ii, jj;
  double change, mean_change = 100;
  int iters = 0;

  //while ((mean_change/(double)(length*length)) > (double)TOL) {
  while (iters<2000) {
    iters++;
    mean_change = 0;
    for (ii = 1; ii < length-1; ii+=BLOCK_SIZE) 
      for (jj = 1; jj < length-1; jj+=BLOCK_SIZE)
	for (i = ii; i < ii+BLOCK_SIZE; i++)
	  for (j = jj; j < jj+BLOCK_SIZE; j++) {
	    change = data[i*length+j] - .25 * (data[(i-1)*length+j] +
					      data[(i+1)*length+j] +
					      data[i*length+j+1] +
					      data[i*length+j-1]);
	    data[i*length+j] -= change * OMEGA;
	    if (change < 0){
	      change = -change;
	    }
	    mean_change += change;
	  }
	
    if (abs(data[(length-2)*(length-2)]) > 10.0*(MAXVAL - MINVAL)) {
      printf("\n PROBABLY DIVERGENCE iter = %d", iters);
      break;
    }

    //#ifdef DEBUG
    //Make sure we're doing work
    fprintf(stderr, ".");
    if(!(iters%50)) fprintf(stderr, "\n");
    //#endif
  }
	
	printf("\n iters = %d", iters);
}

struct timespec diff(struct timespec start, struct timespec end)
{
  struct timespec temp;
  if ((end.tv_nsec-start.tv_nsec)<0) {
    temp.tv_sec = end.tv_sec-start.tv_sec-1;
    temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
  } else {
    temp.tv_sec = end.tv_sec-start.tv_sec;
    temp.tv_nsec = end.tv_nsec-start.tv_nsec;
  }
  return temp;
}