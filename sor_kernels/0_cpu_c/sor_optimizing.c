#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "utils.h"

#define GIG 1000000000

#define N       10000
#define ITERS   10

#define IDEAL_BLOCK 4

#define TOL 0.00001
#define OMEGA 1.94

//Limit the number of iterations for faster execution
#define MAX_ITERS	100

/*****************************************************************************/
main(int argc, char *argv[])
{
  int *iterations;
  void SOR(vec_ptr v, int *iterations);
  void SOR_blocked(vec_ptr v, int *iterations, int b);

  long int i, j;
  long int block_size;
  uint64_t acc;
  long int MAXSIZE = N;

  // declare and initialize the vector structure
  vec_ptr v0 = new_vec(MAXSIZE);
  iterations = (int *) malloc(sizeof(int));

  //Get blocked SOR data
  for(i=1000; i<=N; i+=1000) {
    //long int this_size = i - ((i-2)%IDEAL_BLOCK);
    long int this_size = i;
    acc=0;
    for(j=0; j<ITERS; j++) {
      fprintf(stderr, "\n(%d %d)", this_size, j);
      init_vector_rand(v0, this_size);
      set_vec_length(v0, this_size);
      tick();
      SOR(v0, iterations);
      //SOR_blocked(v0, iterations, IDEAL_BLOCK);
      tock();
      acc += get_execution_time();
    }
    //length, time(ns)
    printf("%d, %lld\n", this_size, acc/ ITERS + 0.5);
  }

  printf("\n");
  
}

/* SOR */
void SOR(vec_ptr v, int *iterations)
{
  long int i, j;
  long int length = get_vec_length(v);
  data_t *data = get_vec_start(v);
  double change, mean_change = 100;   // start w/ something big
  int iters = 0;

  while ((mean_change/(double)(length*length)) > (double)TOL) {
    iters++;
    mean_change = 0;
    for (i = 1; i < length-1; i++) 
      for (j = 1; j < length-1; j++) {
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
      printf("\n PROBABLY DIVERGENCE iter = %ld", iters);
      break;
    }
	//Limit the number of iterations, this adds a constant factor to the kernel
	if(iters == MAX_ITERS) break;
  }
   *iterations = iters;
}

//Requires that (length-2) is divisible by the block size
void SOR_blocked(vec_ptr v, int *iterations, int b)
{
  long int i, j, ii, jj;
  long int length = get_vec_length(v);
  data_t *data = get_vec_start(v);
  double change, mean_change = 100;
  int iters = 0;
  while (((mean_change/(double)(length*length)) > (double)TOL) || 1) {
    iters++;
    mean_change = 0;
    for (ii = 1; ii < length-1; ii+=b) 
      for (jj = 1; jj < length-1; jj+=b)
	for (i = ii; i < ii+b; i++)
	  for (j = jj; j < jj+b; j++) {
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
	if(iters == MAX_ITERS) break;
  }
  *iterations = iters;
}
 
