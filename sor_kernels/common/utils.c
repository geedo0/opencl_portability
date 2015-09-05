#include <stdlib.h>
#include <stdio.h>
#include "utils.h"

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

double fRand(double fMin, double fMax) {
    double f = (double)random() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}
