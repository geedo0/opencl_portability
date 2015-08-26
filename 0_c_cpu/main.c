#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#ifdef USE_MAGICK
#include <wand/MagickWand.h>
#endif

#include "../utils.h"

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

#ifdef USE_MAGICK
#define N     1024
#else
#define N     16000
#endif

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

  register ssize_t
    x;

  size_t
    width;

  ssize_t
    y;
  #endif

  int ii, jj;

  PixMatrix input, kernel, output;

  init_image(&input, 1, 1, N, N, input_image);
  init_image(&kernel, 1, 1, 3, 3, (uint16_t*) input_kernel);
  init_image(&output, 1, 1, N, N, output_image);

  #ifdef USE_MAGICK
  if (argc != 3)
    {
      (void) fprintf(stdout,"Usage: %s input output\n",argv[0]);
      exit(1);
    }

  /*
    Read an image.
  */
  MagickWandGenesis();
  image_wand = NewMagickWand();
  status = MagickReadImage(image_wand, argv[1]);
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
  //Initialize with random data
  init_matrix(N, input.px);
  #endif

  /*
    Modify the image buffer
  */
  #ifdef USE_MAGICK
  tick();
  convolve_image(input, kernel, output);
  tock();
  print_metrics(N, get_execution_time());
  #else
  printf("length, time (ns)\n");
  for(ii=100; ii<=16000; ii+=100) {
    input.dim.x = ii;
    input.dim.y = ii;
    tick();
    convolve_image(input, kernel, output);
    tock();
    printf("%d, %lld\n", ii, get_execution_time());
  }
  #endif

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
  status = MagickWriteImages(modified_wand, argv[2], MagickTrue);
  if (status == MagickFalse)
    ThrowWandException(image_wand);
  modified_wand = DestroyMagickWand(modified_wand);
  MagickWandTerminus();
  #endif

  return(0);
}
