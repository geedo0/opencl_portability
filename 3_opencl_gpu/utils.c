#include "utils.h"

struct timespec diff(struct timespec start, struct timespec end);

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

void print_matrix(int N, float *a) {
  int ii, jj;
  for(ii=0; ii<N; ii++) {
    for(jj=0; jj<N; jj++) {
      printf("%.1f, ", a[ii*N+jj]);
    }
    printf("\n");
  }
}

void print_image(PixMatrix image) {
  int ii, jj;
  for(ii=0; ii<image.dim.y; ii++) {
    for(jj=0; jj<image.dim.x; jj++) {
      printf("%.2hhx, ", image.px[ii*image.dim.x + jj]);
    }
    printf("\n");
  }
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

void init_point(Point *p, uint32_t x, uint32_t y) {
  p->x = x;
  p->y = y;
}

void init_image(PixMatrix *img, int32_t ox, int32_t oy, int32_t dx, int32_t dy, uint16_t *px) {
  img->px = px;
  img->orig.x = ox;
  img->orig.y = oy;
  img->dim.x = dx;
  img->dim.y = dy;
}

void init_matrix(int N, float *matrix) {
    int ii, jj;
    //Initialize with random float values
    srand(0xdeadbeef);
    for(ii==0; ii<N; ii++)
        for(jj=0; jj<N; jj++) {
            matrix[ii*N + jj] = (float) rand() / (float) RAND_MAX * MAX_VALUE;
        }
}

//Optimized sequential algorithm
#define BLOCK_SIZE  2
#define OMEGA       1.95
void SOR_blocked(float* data, int length)
{
  long int i, j, ii, jj;
  double change, mean_change = 100;
  int iters = 0;

  //while ((mean_change/(double)(length*length)) > (double)TOL) {
  while (iters<ITERS) {
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
    
    if (abs(data[(length-2)*(length-2)]) > 10.0*(MAX_VALUE)) {
      printf("\n PROBABLY DIVERGENCE iter = %d", iters);
      break;
    }
  }
}

void print_metrics(int N, uint64_t run_time) {
  double ns_per_element = (double) run_time / (double) (N*N);
  printf("\nTotal time: %lld ns", run_time);
  printf("\n%f ns per element. (N=%d)\n", ns_per_element, N);
}

void check_results(uint16_t *host, uint16_t *device, int N) {
  int ii, jj;
  int error_count = 0;
  uint32_t error, error_normalized;
  printf("\nChecking results using tolerance %f\n", TOL);
  for(ii=0; ii<N; ii++) {
    for(jj=0; jj<N; jj++) {
      error = abs(host[ii*N + jj] - device[ii*N + jj]);
      if(error > TOL) {
        printf("Mismatch at %d,%d: host %d device %d\n",
          ii, jj, host[ii*N + jj], device[ii*N + jj]);
        error_count++;
      }
    }
  }
  printf("%d errors detected\n", error_count);
}