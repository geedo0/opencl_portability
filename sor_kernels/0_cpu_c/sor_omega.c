#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "utils.h"

#define N  1010

#define TOL 0.001

#define O_ITERS 50        // # of OMEGA values to be tested
#define PER_O_ITERS 10    // trials per OMEGA value
double OMEGA = 1.67;     // OMEGA base - first OMEGA tested
#define OMEGA_INC 0.01   // OMEGA increment for each O_ITERS

main(int argc, char *argv[])
{
  double convergence[O_ITERS][2];
  int *iterations;
  void SOR(vec_ptr v, int *iterations);

  long int i, j;
  long int MAXSIZE = N;

  // declare and initialize the vector structure
  vec_ptr v0 = new_vec(MAXSIZE);
  iterations = (int *) malloc(sizeof(int));

  for (i = 0; (i < O_ITERS) && (OMEGA < 1.99); i++) {
    fprintf(stderr,"\n%0.2f", OMEGA);
    double acc = 0.0;
    for (j = 0; j < PER_O_ITERS; j++) {
      set_vec_length(v0, MAXSIZE);
      init_vector_rand(v0, MAXSIZE);
      SOR(v0,iterations);
      acc += (double)(*iterations);
      //printf(", %d", *iterations);
    }
    convergence[i][0] = OMEGA;
    convergence[i][1] = acc/(double)(PER_O_ITERS);
    OMEGA += OMEGA_INC;
  }

  for (i = 0; i < O_ITERS; i++)
    printf("\n%0.2f %0.1f", convergence[i][0], convergence[i][1]);

  printf("\n");
  
}

void SOR(vec_ptr v, int *iterations)
{
  long int i, j;
  long int length = get_vec_length(v);
  data_t *data = get_vec_start(v);
  double change, mean_change = 100000000;   // start w/ something big
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
  }
   *iterations = iters;
}
