#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <unistd.h>
#else
#include <CL/cl.h>
#endif

#include "err_code.h"
#include "device_picker.h"
#include "utils.h"

#ifdef USE_MAGICK
#include <wand/MagickWand.h>
#endif

void convolve_image(PixMatrix x, PixMatrix h, PixMatrix y);

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

  cl_uint deviceIndex = 0;
  char *input_file, *output_file;
  parseArguments(argc, argv, &deviceIndex, &input_file, &output_file);

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

  //OpenCL Boiler Plate
  cl_mem d_input_image, d_kernel, d_output_image;
  char * kernelsource;

  cl_int err;             // error code returned from OpenCL calls
  cl_device_id     device;        // compute device id 
  cl_context       context;       // compute context
  cl_command_queue commands;      // compute command queue
  cl_program       program;       // compute program
  cl_kernel        kernel;        // compute kernel


//--------------------------------------------------------------------------------
// Create a context, queue and device.
//--------------------------------------------------------------------------------
    // Get list of devices
    cl_device_id devices[MAX_DEVICES];
    unsigned numDevices = getDeviceList(devices);

    // Check device index in range
    if (deviceIndex >= numDevices)
    {
      printf("Invalid device index (try '--list')\n");
      return EXIT_FAILURE;
    }

    device = devices[deviceIndex];

    char name[MAX_INFO_STRING];
    getDeviceName(device, name);
    printf("\nUsing OpenCL device: %s\n", name);

    // Create a compute context
    context = clCreateContext(0, 1, &device, NULL, NULL, &err);
    checkError(err, "Creating context");
    // Create a command queue
    commands = clCreateCommandQueue(context, device, 0, &err);
    checkError(err, "Creating command queue");

//--------------------------------------------------------------------------------
// Setup the buffers, initialize matrices, and write them into global memory
//--------------------------------------------------------------------------------
    d_input_image = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            sizeof(uint16_t) * numElements, input_image, &err);
    checkError(err, "Copying input image");
    d_output_image = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                            sizeof(uint16_t) * numElements, NULL, &err);
    checkError(err, "Creating output buffer");
    d_kernel = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            sizeof(uint16_t) * numElements, input_kernel, &err);
    checkError(err, "Copying kernel");

    #ifdef USE_MAGICK
    printf("Convolving input file %s with %d pixels.\n", input_file, numElements);
    #endif

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
// OpenCL Convolution
//--------------------------------------------------------------------------------
    kernelsource = getKernelSource("./conv.cl");
    // Create the comput program from the source buffer
    program = clCreateProgramWithSource(context, 1, (const char **) &kernelsource, NULL, &err);
    checkError(err, "Creating program from sor.cl");
    free(kernelsource);
    // Build the program
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n%s\n", err_code(err));
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        return EXIT_FAILURE;
    }

    // Create the compute kernel from the program
    kernel = clCreateKernel(program, "conv", &err);
    checkError(err, "Creating kernel from conv.cl");

    printf("\n===== OpenCL Convolution ======\n");

    tick();
    const size_t global[2] = {N, N};
    const size_t local[2] = {16,16};

    err =  clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input_image);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_output_image);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_kernel);
    checkError(err, "Setting kernel args");
    err = clEnqueueNDRangeKernel(
        commands,
        kernel,
        2, NULL,
        &global, &local,
        0, NULL, NULL);
    checkError(err, "Enqueueing kernel");
    err = clFinish(commands);
    checkError(err, "Waiting for kernel to finish");
    tock();

    err = clEnqueueReadBuffer(
        commands, d_output_image, CL_TRUE, 0,
        sizeof(uint16_t) * numElements, output_image,
        0, NULL, NULL);
    checkError(err, "Reading back image");
    
    print_metrics(N, get_execution_time());
    //check_results(h_A, h_result, N);

//--------------------------------------------------------------------------------
// Clean up!
//--------------------------------------------------------------------------------
    clReleaseMemObject(d_input_image);
    clReleaseMemObject(d_output_image);
    clReleaseMemObject(d_kernel);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

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