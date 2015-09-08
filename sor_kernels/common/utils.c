#include <stdlib.h>
#include <stdio.h>
#include "utils.h"

struct timespec time1, time2;

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

uint64_t get_execution_time() {
  struct timespec delta = diff(time1,time2);
  return (uint64_t) (GIG * (uint64_t) delta.tv_sec + (uint64_t) delta.tv_nsec);
}

void print_metrics(int N, uint64_t run_time) {
  double ns_per_element = (double) run_time / (double) (N*N);
  printf("\nTotal time: %lld ns", run_time);
  printf("\n%f ns per element. (N=%d)\n", ns_per_element, N);
}

char * getKernelSource(char *filename)
{
    FILE *file = fopen(filename, "r");
    if (!file)
    {
        fprintf(stderr, "Error: Could not open kernel source file\n");
        exit(EXIT_FAILURE);
    }
    fseek(file, 0, SEEK_END);
    int len = ftell(file) + 1;
    rewind(file);

    char *source = (char *)calloc(sizeof(char), len);
    if (!source)
    {
        fprintf(stderr, "Error: Could not allocate memory for source string\n");
        exit(EXIT_FAILURE);
    }
    fread(source, sizeof(char), len, file);
    fclose(file);
    return source;
}

vec_ptr new_vec(long int len) {
  /* Allocate and declare header structure */
  vec_ptr result = (vec_ptr) malloc(sizeof(vec_rec));
  if (!result)
  	return NULL;  /* Couldn't allocate storage */
  result->len = len;

  /* Allocate and declare array */
  if (len > 0) {
    data_t *data = (data_t *) calloc(len*len, sizeof(data_t));
    if (!data) {
      free((void *) result);
      printf("\n COULDN'T ALLOCATE STORAGE \n");
      return NULL;  /* Couldn't allocate storage */
    }
    result->data = data;
  }
  else
  	result->data = NULL;
  
  return result;
}

data_t *get_vec_start(vec_ptr v) {
  return v->data;
}

long int get_vec_length(vec_ptr v) {
  return v->len;
}

int set_vec_length(vec_ptr v, long int index) {
  v->len = index;
  return 1;
}

int init_vector(vec_ptr v, long int len) {
  long int i;

  if (len > 0) {
    v->len = len;
    for (i = 0; i < len*len; i++)
    	v->data[i] = (data_t)(i);
    return 1;
  }
  else
  	return 0;
}

int init_vector_rand(vec_ptr v, long int len) {
  long int i;

  if (len > 0) {
    v->len = len;
    for (i = 0; i < len*len; i++)
      v->data[i] = (data_t)(fRand((double)(MINVAL),(double)(MAXVAL)));
    return 1;
  }
  else
  	return 0;
}

int print_vector(vec_ptr v) {
  long int i, j, len;

  len = v->len;
  printf("\n length = %ld", len);
  for (i = 0; i < len; i++) {
    printf("\n");
    for (j = 0; j < len; j++)
      printf("%.4f ", (data_t)(v->data[i*len+j]));
  }
}

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