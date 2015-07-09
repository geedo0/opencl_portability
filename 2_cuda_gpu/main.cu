#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#ifdef USE_MAGICK
extern "C" {
  #include <wand/MagickWand.h>
}
#endif

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

#define GIG 1000000000

long int get_execution_time() {
  struct timespec delta = diff(time1,time2);
  return (long int) (GIG * delta.tv_sec + delta.tv_nsec);
}

#define tick()  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1)
#define tock()  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2)

typedef struct Point_struct {
  int32_t x;
  int32_t y;
} Point;

typedef struct Image_struct {
  uint16_t *px;
  Point dim;
  Point orig;
} PixMatrix;

void init_matrix(int N, uint16_t *matrix) {
    int ii, jj;
    //Initialize with random integers
    srand(0xdeadbeef);
    for(ii==0; ii<N; ii++)
        for(jj=0; jj<N; jj++) {
            matrix[ii*N + jj] =  rand() % 0x0000ffff;
        }
}

void print_metrics(int N, long int run_time) {
  double ns_per_element = (double) run_time / (double) (N*N);
  printf("\nTotal time: %ld ns", run_time);
  printf("\n%f ns per element. (N=%d)\n", ns_per_element, N);
}
void convolve_image(PixMatrix x, PixMatrix h, PixMatrix y);
void parseArguments(int argc, char *argv[], char **input_file, char **output_file);

#ifdef USE_MAGICK
#define ThrowWandException(wand) \
{ \
  char \
    *description; \
 \
  ExceptionType \
    severity; \
 \
  description=MagickGetException(wand,&severity); \
  (void) fprintf(stderr,"%s %s %lu %s\n",GetMagickModule(),description); \
  description=(char *) MagickRelinquishMemory(description); \
  exit(-1); \
}
#endif

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

#define N   1024
#define CONV_N  3
__global__ void kernel_conv(uint16_t *x, uint16_t *y, float *h) {
  //BlockDim = dimensions of block in threads
  //GridDim = dimensions of grid in blocks
  //BlockIdx = position w/n grid
  //ThreadIdx = position w/n block
  const int ii = blockDim.y*blockIdx.y + threadIdx.y;
  const int jj = blockDim.x*blockIdx.x + threadIdx.x;

    int iii, jjj;
    float px;
    int o_x, o_y;
    //Set the new origin
    o_x = jj - 1;
    o_y = ii - 1;
    px = 0;
    for(iii=0; iii<CONV_N; iii++) {
    for(jjj=0; jjj<CONV_N; jjj++) {
      if(((o_y+iii) < 0) || ((o_y+iii) >= N) || 
        ((o_x+jjj) < 0) || ((o_x+jjj) >= N)) {
        //For now, boundary elements take on the value of the origin
        px += ((float) x[(o_y+1)*N + (o_x+1)])*h[iii*CONV_N+jjj];
      }
      else {
        px += ((float) x[(o_y+iii)*N + (o_x+jjj)])*h[iii*CONV_N+jjj];
      }
    }
    }
    px = px < 0 ? 0 : px;
    px = px > 65535 ? 65535 : px;
    y[ii*N+jj] = (uint16_t) px;
}


void init_image(PixMatrix *img, int32_t ox, int32_t oy, int32_t dx, int32_t dy, uint16_t *px) {
  img->px = px;
  img->orig.x = ox;
  img->orig.y = oy;
  img->dim.x = dx;
  img->dim.y = dy;
}


//Bottom Sobel
float input_kernel[] = {
  -1,-2,-1,
  0,0,0,
  1,2,1
};

uint16_t input_image[N*N];
uint16_t output_image[N*N];

int main(int argc,char **argv) {
  #ifdef USE_MAGICK
  MagickBooleanType
    status;

  MagickPixelPacket
    pixel;

  MagickWand
    *modified_wand,
    *image_wand;

  PixelIterator
    *modified_iterator,
    *iterator;

  PixelWand
    **modified_pixels,
    **pixels;
  #endif

  register ssize_t
    x;

  size_t
    width;

  ssize_t
    y;

  int ii, jj;
  int numElements = N*N;

  PixMatrix input, image_kernel, output;

  init_image(&input, 1, 1, N, N, input_image);
  init_image(&image_kernel, 1, 1, 3, 3, (uint16_t*) input_kernel);
  init_image(&output, 1, 1, N, N, output_image);

  char *input_file=NULL, *output_file=NULL;
  parseArguments(argc, argv, &input_file, &output_file);
  if((input_file==NULL) || (output_file==NULL)) {
    exit(1);
  }

  #ifdef USE_MAGICK
  /*
    Read an image.
  */
  MagickWandGenesis();
  image_wand = NewMagickWand();
  status = MagickReadImage(image_wand, input_file);
  if (status == MagickFalse)
    ThrowWandException(image_wand);
  modified_wand = CloneMagickWand(image_wand);

  /*
    Read pixels into buffer
  */
  iterator = NewPixelIterator(image_wand);
  if (iterator == (PixelIterator *) NULL)
    ThrowWandException(image_wand);
  for (y=0; y < (ssize_t) MagickGetImageHeight(image_wand); y++) {
    pixels = PixelGetNextIteratorRow(iterator, &width);
    if (pixels == (PixelWand **) NULL)
      break;
    for (x=0; x < (ssize_t) width; x++) {
      PixelGetMagickColor(pixels[x], &pixel);
      //Convert to 8-bit grayscale
      input_image[y*N+x] = pixel.red;
    }
  }
  if (y < (ssize_t) MagickGetImageHeight(image_wand))
    ThrowWandException(image_wand);
  iterator = DestroyPixelIterator(iterator);
  image_wand = DestroyMagickWand(image_wand);
  #else
  init_matrix(N, input_image);
  #endif
  
  //CUDA Boiler Plate
  uint16_t *d_input_image, *d_output_image;
  float *d_kernel;
  size_t allocSize = numElements * sizeof(uint16_t);
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_input_image, allocSize));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_output_image, allocSize));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_kernel, 9*sizeof(float)));

  CUDA_SAFE_CALL(cudaMemcpy(d_input_image, input_image, allocSize, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_kernel, input_kernel, 9*sizeof(float), cudaMemcpyHostToDevice));

  /*
    Modify the image buffer
  */


//--------------------------------------------------------------------------------
// Compute the reference solution on the CPU
//--------------------------------------------------------------------------------
    printf("\n===== CPU Convolution ======\n");

    tick();
    convolve_image(input, image_kernel, output);
    tock();


    for(ii=0; ii<10; ii++)
      printf("%d, ",output_image[ii]);
    printf("\n");
    print_metrics(N, get_execution_time());


//--------------------------------------------------------------------------------
// CUDA Convolution
//--------------------------------------------------------------------------------
    printf("\n===== CUDA Convolution ======\n");

    tick();
    dim3 DimBlock(16, 16, 1);
    dim3 DimGrid(64,64,1);
    kernel_conv<<<DimGrid, DimBlock>>>(d_input_image, d_output_image, d_kernel);
    cudaDeviceSynchronize();
    tock();

    CUDA_SAFE_CALL(cudaPeekAtLastError());
    CUDA_SAFE_CALL(cudaMemcpy(output_image, d_output_image, allocSize, cudaMemcpyDeviceToHost));
    print_metrics(N, get_execution_time());
    //check_results(h_A, h_result, N);

    for(ii=0; ii<10; ii++)
      printf("%d, ",output_image[ii]);
    printf("\n");


//--------------------------------------------------------------------------------
// Clean up!
//--------------------------------------------------------------------------------
  CUDA_SAFE_CALL(cudaFree(d_input_image));
  CUDA_SAFE_CALL(cudaFree(d_output_image));
  CUDA_SAFE_CALL(cudaFree(d_kernel));

  #ifdef USE_MAGICK
  /*
    Write back the modified image buffer
  */

  modified_iterator = NewPixelIterator(modified_wand);
  if (modified_iterator == (PixelIterator *) NULL)
    ThrowWandException(modified_wand);
  for (y=0; y < (ssize_t) MagickGetImageHeight(modified_wand); y++) {
    modified_pixels = PixelGetNextIteratorRow(modified_iterator, &width);
    if(modified_pixels == (PixelWand **) NULL)
      break;
    for (x=0; x < (ssize_t) width; x++) {
      //Create a pixel
      PixelGetMagickColor(modified_pixels[x], &pixel);
      pixel.red = output_image[y*N+x];
      pixel.blue = output_image[y*N+x];
      pixel.green = output_image[y*N+x];
      PixelSetMagickColor(modified_pixels[x], &pixel);
    }
    (void) PixelSyncIterator(modified_iterator);
  }
  if (y < (ssize_t) MagickGetImageHeight(modified_wand))
    ThrowWandException(modified_wand);
  modified_iterator = DestroyPixelIterator(modified_iterator);

  /*
    Write the image then destroy it.
  */
  status = MagickWriteImages(modified_wand, output_file, MagickTrue);
  if (status == MagickFalse)
    ThrowWandException(image_wand);
  modified_wand = DestroyMagickWand(modified_wand);
  MagickWandTerminus();
  #endif

  return EXIT_SUCCESS;
}

void convolve_image(PixMatrix x, PixMatrix h, PixMatrix y) {
  int32_t ii, jj;
  int32_t iii, jjj;
  double px;
  Point ko;
  float *kernel = (float*) h.px;  //bad bad hack

  for(ii=0; ii<x.dim.y; ii++) {
    for(jj=0; jj<x.dim.x; jj++) {
      ko.x = jj - h.orig.x;
      ko.y = ii - h.orig.y;
      px = 0;
      for(iii=0; iii<h.dim.y; iii++) {
        for(jjj=0; jjj<h.dim.x; jjj++) {
          if(((ko.y+iii) < 0) || ((ko.y+iii) >= x.dim.y) || 
            ((ko.x+jjj) < 0) || ((ko.x+jjj) >= x.dim.x)) {
            //For now, boundary elements take on the value of the origin
            px += x.px[(ko.y+h.orig.y)*x.dim.x + (ko.x+h.orig.x)]*kernel[iii*h.dim.x+jjj];
          }
          else {
            px += x.px[(ko.y+iii)*x.dim.x + (ko.x+jjj)]*kernel[iii*h.dim.x+jjj];
          }
        }
      }
      px = px < 0 ? 0 : px;
      px = px > 65535 ? 65535 : px;
      y.px[ii*x.dim.x+jj] = (uint16_t) px;
    }
  }
}

void parseArguments(int argc, char *argv[], char **input_file, char **output_file)
{
  for (int i = 1; i < argc; i++)
  {
    if(!strcmp(argv[i], "-if")) {
      *input_file = argv[++i];
    }
    else if(!strcmp(argv[i], "-of")) {
      *output_file = argv[++i];
    }
  }
}
