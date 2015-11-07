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

#define N     10000

float array_A[N*N];

int main(int argc,char **argv) {
  int ii, jj, kk;

  int numElements = N*N;

  cl_uint deviceIndex = 0;
  parseArguments(argc, argv, &deviceIndex);

  initializeArray2D(array_A, N, 2453);

  //OpenCL Boiler Plate
  cl_mem d_A, d_B;
  char *kernelsource;

  cl_int err;             // error code returned from OpenCL calls
  cl_device_id     device;        // compute device id 
  cl_context       context;       // compute context
  cl_command_queue commands;      // compute command queue
  cl_program       program;       // compute program
  cl_kernel        kernel_a, kernel_b;        // compute kernel

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
  d_A = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
          sizeof(float) * numElements, array_A, &err);
  checkError(err, "Copying array_A");
  d_B = clCreateBuffer(context, CL_MEM_READ_WRITE,
                    sizeof(float) * numElements, NULL, &err);
  checkError(err, "Creating array_B");

  /*
  //--------------------------------------------------------------------------------
  // Compute the reference solution on the CPU
  //--------------------------------------------------------------------------------
  printf("\n===== CPU Convolution ======\n");

  tick();
  convolve_image(input, image_kernel, output);
  tock();

  print_metrics(N, get_execution_time());
  */


  //--------------------------------------------------------------------------------
  // OpenCL SOR
  //--------------------------------------------------------------------------------
  kernelsource = getKernelSource("./sor.cl");
  // Create the program from the source buffer
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
  kernel_a = clCreateKernel(program, "sor", &err);
  checkError(err, "Creating kernel_a from sor.cl");
  
  kernel_b = clCreateKernel(program, "sor", &err);
  checkError(err, "Creating kernel_b from sor.cl");

  int arrLen = N;

  err =  clSetKernelArg(kernel_a, 0, sizeof(cl_int), &arrLen);
  checkError(err, "Setting kernel_a args 0");
  err = clSetKernelArg(kernel_a, 1, sizeof(cl_mem), &d_A);
  checkError(err, "Setting kernel_a args 1");
  err = clSetKernelArg(kernel_a, 2, sizeof(cl_mem), &d_B);
  checkError(err, "Setting kernel_a args 2");
  
  err =  clSetKernelArg(kernel_b, 0, sizeof(cl_int), &arrLen);
  err |= clSetKernelArg(kernel_b, 1, sizeof(cl_mem), &d_B);
  err |= clSetKernelArg(kernel_b, 2, sizeof(cl_mem), &d_A);
  checkError(err, "Setting kernel_b args");

  printf("\n===== OpenCL Convolution ======\n");
  
/*
  for(ii=0;ii<10;ii++) {
    for(jj=0;jj<10;jj++) {
      printf("%.2f, ",array_A[ii*arrLen+jj]);
    }
    printf("\n");
  }
*/

  size_t global[2];
  size_t local[2];

  for(ii=0; ii<9; ii++) {
    int this_size = 1000 + 1000*ii;
    this_size = this_size - (this_size%32);
    fprintf(stderr, "size %d\n", this_size);
    global[0] = this_size;
    global[1] = this_size;
    local[0] = 32;
    local[1] = 32;
    
    uint64_t acc = 0;
    for(kk=0; kk < 10; kk++) {
      tick();
      for(jj=0; jj < 50; jj++) {
        err = clEnqueueNDRangeKernel(
          commands, kernel_a, 2, NULL,
          (size_t *) &global, (size_t *) &local, 0, NULL, NULL);
        checkError(err, "Enqueueing kernel_a");
        err = clFinish(commands);
        checkError(err, "Waiting for kernel_a to finish");

        err = clEnqueueNDRangeKernel(
          commands, kernel_b, 2, NULL,
          (size_t *) &global, (size_t *) &local, 0, NULL, NULL);
        checkError(err, "Enqueueing kernel_b");
        err = clFinish(commands);
        checkError(err, "Waiting for kernel_b to finish");
      }
      tock();
      acc += get_execution_time();
    }
    //length, time
    printf("%d, %f\n", this_size, ((double) acc)/10 );
  }

  err = clEnqueueReadBuffer(
    commands, d_A, CL_TRUE, 0,
    sizeof(float) * numElements, array_A,
    0, NULL, NULL);
  checkError(err, "Reading back image");


/*
  printf("\nResults\n");
  for(ii=0;ii<10;ii++) {
    for(jj=0;jj<10;jj++) {
      printf("%.2f, ",array_A[ii*arrLen+jj]);
    }
    printf("\n");
  }
*/

  //--------------------------------------------------------------------------------
  // Clean up!
  //--------------------------------------------------------------------------------
  clReleaseMemObject(d_A);
  clReleaseMemObject(d_B);
  clReleaseProgram(program);
  clReleaseKernel(kernel_a);
  clReleaseKernel(kernel_b);
  clReleaseCommandQueue(commands);
  clReleaseContext(context);

  return EXIT_SUCCESS;
}
