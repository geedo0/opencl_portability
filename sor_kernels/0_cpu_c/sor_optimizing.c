#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "utils.h"

#define GIG 1000000000

#define N       1682
#define ITERS   10

#define TOL 0.00001
#define OMEGA 1.94

//Limit the number of iterations for faster execution
#define MAX_ITERS	100

/*****************************************************************************/
main(int argc, char *argv[])
{
  struct timespec diff(struct timespec start, struct timespec end);
  struct timespec time1, time2;
  struct timespec time_delta;
  long int execution_times[9];
  int *iterations;
  void SOR(vec_ptr v, int *iterations);
  void SOR_blocked(vec_ptr v, int *iterations, int b);

  long int i, j;
  long int block_size;
  long int acc;
  long int MAXSIZE = N;

  // declare and initialize the vector structure
  vec_ptr v0 = new_vec(MAXSIZE);
  iterations = (int *) malloc(sizeof(int));

  //Get un-blocked SOR data
  acc=0;
  for(j=0; j<ITERS; j++) {
    fprintf(stderr, "\n(%d)",j);
    init_vector_rand(v0, MAXSIZE);
    set_vec_length(v0, MAXSIZE);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
    SOR(v0, iterations);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
    time_delta = diff(time1,time2);
    acc += (long int) (GIG * time_delta.tv_sec + time_delta.tv_nsec);
  }
  execution_times[0] = acc / ITERS + 0.5;

  //Get blocked data
  for(i=1; i<=8; i++) {
    acc=0;
    for(j=0; j<ITERS; j++) {
      fprintf(stderr, "\n(%d,%d)",i,j);
      init_vector_rand(v0, MAXSIZE);
      set_vec_length(v0, MAXSIZE);
      block_size = i*2;
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
      SOR_blocked(v0, iterations, block_size);
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
      time_delta = diff(time1,time2);
      acc += (long int) (GIG * time_delta.tv_sec + time_delta.tv_nsec);
    }
    execution_times[i] = acc / ITERS + 0.5;
  }

  for(i=0; i<=8; i++) {
    block_size = i==0 ? 1 : i*2;
    printf("%d, %ld\n", block_size, execution_times[i]);
  }

  printf("\n");
  
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

/* SOR w/ blocking */
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
 
