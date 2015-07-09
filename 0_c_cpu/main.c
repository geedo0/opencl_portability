#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <wand/MagickWand.h>
#include "utils.h"

void convolve_image(PixMatrix x, PixMatrix h, PixMatrix y);

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

#define N     1024

//Bottom Sobel
float input_kernel[] = {
  -1,-2,-1,
  0,0,0,
  1,2,1
};

uint16_t input_image[N*N];
uint16_t output_image[N*N];

int main(int argc,char **argv) {
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

  int ii, jj;

  PixMatrix input, kernel, output;

  init_image(&input, 1, 1, N, N, input_image);
  init_image(&kernel, 1, 1, 3, 3, (uint16_t*) input_kernel);
  init_image(&output, 1, 1, N, N, output_image);

  if (argc != 3)
    {
      (void) fprintf(stdout,"Usage: %s input output\n",argv[0]);
      exit(0);
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
  
  /*
    Modify the image buffer
  */
  convolve_image(input, kernel, output);

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
  return(0);
}

void convolve_image(PixMatrix x, PixMatrix h, PixMatrix y) {
  uint32_t ii, jj;
  uint32_t iii, jjj;
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