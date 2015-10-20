#ifndef UTILS_H
#define UTILS_H

#include <stdint.h>
#include <time.h>

#define GIG 1000000000
#define MINVAL   0.0
#define MAXVAL  100.0

extern struct timespec time1, time2;

#define tick()	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1)
#define tock()	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2)

typedef float data_t;

typedef struct {
  long int len;
  data_t *data;
} vec_rec, *vec_ptr;

/* Number of bytes in a vector (SSE sense) */
#define VBYTES 16

/* Number of elements in a vector (SSE sense) */
#define VSIZE VBYTES/sizeof(data_t)

typedef data_t vec_t __attribute__ ((vector_size(VBYTES)));
typedef union {
  vec_t v;
  data_t d[VSIZE];
} pack_t;

struct timespec diff(struct timespec start, struct timespec end);
uint64_t get_execution_time();
void print_metrics(int N, uint64_t run_time);
char* getKernelSource(char *filename);
vec_ptr new_vec(long int len);
data_t *get_vec_start(vec_ptr v);
long int get_vec_length(vec_ptr v);
int set_vec_length(vec_ptr v, long int index);
int init_vector(vec_ptr v, long int len);
int init_vector_rand(vec_ptr v, long int len);
int print_vector(vec_ptr v);
double fRand(double fMin, double fMax);
void initializeArray2D(float *arr, int len, int seed);

#endif
