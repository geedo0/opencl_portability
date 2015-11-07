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

#include "../utils.h"
#include "../utils.c"

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

#ifdef USE_MAGICK
#define N   1024
#else
#define N 16000
#endif
#define CONV_N  3

__global__ void kernel_conv(uint16_t *x, uint16_t *y, int16_t *h) {
  //BlockDim = dimensions of block in threads
  //GridDim = dimensions of grid in blocks
  //BlockIdx = position w/n grid
  //ThreadIdx = position w/n block
  const int ii = blockDim.y*blockIdx.y + threadIdx.y;
  const int jj = blockDim.x*blockIdx.x + threadIdx.x;

    int iii, jjj;
    int32_t px;
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
        px += ((int32_t) x[(o_y+1)*N + (o_x+1)])*h[iii*CONV_N+jjj];
      }
      else {
        px += ((int32_t) x[(o_y+iii)*N + (o_x+jjj)])*h[iii*CONV_N+jjj];
      }
    }
    }
    px = px < 0 ? 0 : px;
    px = px > 65535 ? 65535 : px;
    y[ii*N+jj] = (uint16_t) px;
}

//Bottom Sobel
int16_t input_kernel[] = {
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

  #ifdef USE_MAGICK
  char *input_file=NULL, *output_file=NULL;
  parseArguments(argc, argv, &input_file, &output_file);
  if((input_file==NULL) || (output_file==NULL)) {
    exit(1);
  }
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
  int16_t *d_kernel;
  size_t allocSize = numElements * sizeof(uint16_t);
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_input_image, allocSize));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_output_image, allocSize));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_kernel, 9*sizeof(int16_t)));

  CUDA_SAFE_CALL(cudaMemcpy(d_input_image, input_image, allocSize, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_kernel, input_kernel, 9*sizeof(int16_t), cudaMemcpyHostToDevice));

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

    print_metrics(N, get_execution_time());


//--------------------------------------------------------------------------------
// CUDA Convolution
//--------------------------------------------------------------------------------
    printf("\n===== CUDA Convolution ======\n");
	for(ii=100; ii <= N; ii+=100) {
		jj = ii - (ii%16);
		int32_t length = jj/16;
    dim3 DimBlock(16, 16, 1);
		dim3 DimGrid(length, length, 1);
		tick();
    kernel_conv<<<DimGrid, DimBlock>>>(d_input_image, d_output_image, d_kernel);
    cudaDeviceSynchronize();
    tock();

    CUDA_SAFE_CALL(cudaPeekAtLastError());
    CUDA_SAFE_CALL(cudaMemcpy(output_image, d_output_image, allocSize, cudaMemcpyDeviceToHost));
		printf("%d, %lld\n", jj, get_execution_time());
	}

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
