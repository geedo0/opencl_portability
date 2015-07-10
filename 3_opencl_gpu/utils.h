#ifndef MY_UTILS_H
#define MY_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <time.h>

#define GIG 1000000000
#define MAX_VALUE   1009
#define ITERS   500
#define TOL 5

extern struct timespec time1, time2;

struct PixMatrix;
struct Point;

typedef struct Point_struct {
	int32_t x;
	int32_t y;
} Point;

typedef struct Image_struct {
	uint16_t *px;
	Point dim;
	Point orig;
} PixMatrix;

#define tick()	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1)
#define tock()	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2)

uint64_t get_execution_time();
void print_metrics(int N, uint64_t run_time);

char* getKernelSource(char *filename);

void init_matrix(int N, float *matrix);
void init_point(Point *p, uint32_t x, uint32_t y);
void init_image(PixMatrix *img, int32_t ox, int32_t oy, int32_t dx, int32_t dy,
	uint16_t *px);

void print_matrix(int N, float *a);
void print_image(PixMatrix image);
void check_results(uint16_t *host, uint16_t *device, int N);
void SOR_blocked(float *data, int length);

#endif
