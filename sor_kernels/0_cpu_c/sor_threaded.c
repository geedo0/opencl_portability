#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <pthread.h>

#include "utils.h"

#define GIG 1000000000

#define N       1680
#define ITERS   10

#define TOL 0.00001
#define OMEGA 1.96

//Limit the number of iterations for faster execution
#define MAX_ITERS 100

#define BLOCK_SIZE 4

typedef struct {
  vec_ptr v;    //Shared data
  long int tid; //Unique thread ID
  double *diff; //Maintain a pointer to the shared diff
  pthread_mutex_t *lock;
  pthread_barrier_t *barr;
} worker_data;

void SOR_multithreaded(vec_ptr v, int *iterations);
void *SOR_worker(void *threadarg);
void *SOR_worker_blocked(void *threadarg);

int NUM_THREADS;

main(int argc, char *argv[])
{
  struct timespec diff(struct timespec start, struct timespec end);
  struct timespec time1, time2;
  struct timespec time_delta;
  long int execution_times[9];
  int *iterations;
  void SOR(vec_ptr v, int *iterations);
  void SOR_blocked(vec_ptr v, int *iterations, int b);

  long int i, j, k;
  long int acc;
  long int MAXSIZE = N;

  // declare and initialize the vector structure
  vec_ptr v0 = new_vec(MAXSIZE);
  iterations = (int *) malloc(sizeof(int));

  //Get single-threaded data
  acc=0;
  for(j=0; j<ITERS; j++) {
    fprintf(stderr, "\n(%d)",j);
    init_vector_rand(v0, MAXSIZE);
    set_vec_length(v0, MAXSIZE);
    clock_gettime(CLOCK_MONOTONIC, &time1);
    SOR(v0, iterations);
    clock_gettime(CLOCK_MONOTONIC, &time2);
    time_delta = diff(time1,time2);
    acc += (long int) (GIG * time_delta.tv_sec + time_delta.tv_nsec);
  }
  execution_times[0] = acc / ITERS + 0.5;

  //Get multi-threaded data
  for(i=1; i<=8; i++) {
    acc=0;
    NUM_THREADS = i*2;
    for(j=0; j<ITERS; j++) {
      fprintf(stderr, "\n(%d,%d)",i,j);
      init_vector_rand(v0, MAXSIZE);
      set_vec_length(v0, MAXSIZE);
      clock_gettime(CLOCK_MONOTONIC, &time1);
      SOR_multithreaded(v0,iterations);
      clock_gettime(CLOCK_MONOTONIC, &time2);
      time_delta = diff(time1,time2);
      acc += (long int) (GIG * time_delta.tv_sec + time_delta.tv_nsec);
    }
    execution_times[i] = acc / ITERS + 0.5;
  }

  for(i=0; i<=8; i++) {
    NUM_THREADS = i==0 ? 1 : i*2;
    printf("%d, %ld\n", NUM_THREADS, execution_times[i]);
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

void SOR(vec_ptr v, int *iterations)
{
  long int i, j;
  long int length = get_vec_length(v);
  data_t *data = get_vec_start(v);
  double change, mean_change = 100;   // start w/ something big
  int iters = 0;

  while (((mean_change/(double)(length*length)) > (double)TOL) || 1) {
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
 
//Creates threads running the SOR solver using stripwise decomposition
//based on the pseudocode from Culler Parallel Computer Architecture Ch. 2
//NUM_THREADS is a global variable set prior to invocation
void SOR_multithreaded(vec_ptr v, int *iterations)
{
  long int i;
  pthread_t threads[NUM_THREADS];
  pthread_mutex_t diff_mutex;
  pthread_barrier_t worker_barrier;
  int rc;
  double global_diff = 0;
  worker_data thread_data[NUM_THREADS];

  if(rc = pthread_mutex_init(&diff_mutex, NULL)) {
    fprintf(stderr, "Error %d creating mutex\n", rc);
    exit(-1);
  }

  if(rc = pthread_barrier_init(&worker_barrier, NULL, NUM_THREADS)) {
    fprintf(stderr, "Error %d creating barrier\n", rc);
    exit(-1);
  }

  //Need to turn main thread into a worker?
  for(i = 0; i < NUM_THREADS; i++) {
    thread_data[i].v = v;
    thread_data[i].tid = i;
    thread_data[i].diff = &global_diff;
    thread_data[i].lock = &diff_mutex;
    thread_data[i].barr = &worker_barrier;
    rc = pthread_create(&threads[i], NULL, SOR_worker, (void*) &thread_data[i]);
    if(rc) {
      fprintf(stderr, "Error %d from creating thread\n", rc);
      exit(-1);
    }
  }
  //Join on all threads to make sure we are done
  for(i = 0; i < NUM_THREADS; i++) {
    if(rc = pthread_join(threads[i], NULL)) {
      fprintf(stderr, "Error %d from joining thread %d\n", rc, i);
    }
  }
}

void *SOR_worker(void *threadarg) {
  long int i, j;
  int rc;
  int done = 0;
  int iters = 0;
  worker_data *myData = (worker_data*) threadarg;
  long int length = get_vec_length(myData->v);
  data_t *data = get_vec_start(myData->v);
  data_t temp;
  double myDiff = 0;
  double change;
  //Assume that the length is a multiple of the threads
  long int myMin = 1 + (myData->tid) * length/NUM_THREADS;
  long int myMax = myMin + length/NUM_THREADS - 1;

  while(!done) {
    iters++;
    myDiff = 0;
    *(myData->diff) = 0; 
    for(i = myMin; i < myMax; i++) {
      for(j = 1; j < length - 1; j++) {
        change = data[i*length+j] - .25*( data[(i-1)*length+j] +
                                          data[(i+1)*length+j] +
                                          data[i*length+j+1] +
                                          data[i*length+j-1]);
        data[i*length+j] -= change * OMEGA;
        myDiff += abs(change);
      }
    }
    pthread_mutex_lock(myData->lock);
    *(myData->diff) += myDiff;
    pthread_mutex_unlock(myData->lock);

    pthread_barrier_wait(myData->barr);
    if((*(myData->diff)/(double)(length*length) < (double)TOL) || (iters > MAX_ITERS))
      done = 1;
    pthread_barrier_wait(myData->barr);
  }
}

void *SOR_worker_blocked(void *threadarg) {
  long int i, j, ii, jj;
  int rc;
  int done = 0;
  int iters = 0;
  worker_data *myData = (worker_data*) threadarg;
  long int length = get_vec_length(myData->v);
  data_t *data = get_vec_start(myData->v);
  data_t temp;
  double myDiff = 0;
  double change;
  //Assume that the length is a multiple of the threads
  long int myMin = 1 + (myData->tid) * length/NUM_THREADS;
  long int myMax = myMin + length/NUM_THREADS - 1;

  while(!done) {
    iters++;
    myDiff = 0;
    *(myData->diff) = 0;
    for (ii = myMin; ii < myMax; ii+=BLOCK_SIZE) 
      for (jj = 1; jj < length-1; jj+=BLOCK_SIZE)
        for (i = ii; i < ii+BLOCK_SIZE; i++)
          for (j = jj; j < jj+BLOCK_SIZE; j++) {
            change = data[i*length+j] - .25*( data[(i-1)*length+j] +
                                              data[(i+1)*length+j] +
                                              data[i*length+j+1] +
                                              data[i*length+j-1]);
            data[i*length+j] -= change * OMEGA;
            myDiff += abs(change);
    }
    pthread_mutex_lock(myData->lock);
    *(myData->diff) += myDiff;
    pthread_mutex_unlock(myData->lock);

    pthread_barrier_wait(myData->barr);
    if((*(myData->diff)/(double)(length*length) < (double)TOL) || (iters > MAX_ITERS))
      done = 1;
    pthread_barrier_wait(myData->barr);
  }
}