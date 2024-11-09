﻿#include <stdio.h>

#ifndef __CUDACC__
#define __CUDACC__
#endif

#ifdef __CUDACC__
// Code that should only be compiled when CUDA is available

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

__host__ __device__ void SumParts(int* c, const int* a, const int* b)
{
   *c = *a + *b;
}

__global__ void addOtherKernel(int *c, const int *a, const int *b)
{
   int i = threadIdx.x;
   SumParts(c+i, a+i, b+i);
   c[i] += a[i] - 1;
}

__global__ void IncKernelFunc(int *c)
{
   int i = threadIdx.x;
   c[i]++;
}

//extern void IncInCPP(int* dest);
//__global__ void IncExtKernel(int *c)
//{
//   int i = threadIdx.x;
//   IncInCPP(c + i); // cannot call __host__
//}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    //addKernel<<<1, size>>>(dev_c, dev_a, dev_b);
    addOtherKernel<<<1, size>>>(dev_c, dev_a, dev_b);
    //IncKernelFunc<<<1, size>>>(dev_c);
    //IncExtKernel<<<1, size>>>(dev_c);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}

bool CudaWork(int* c, const int* a, const int* b, unsigned int size)
{
   size_t freeMem, totalMem;
   {
      cudaError_t status = cudaMemGetInfo(&freeMem, &totalMem);
      if (status != cudaSuccess) {
         printf("Error: %s\n", cudaGetErrorString(status));
         return false;
      }

      printf("Total memory on GPU: %.2f MB\n", totalMem / (1024.0 * 1024.0));
      printf("Free memory available for cudaMalloc: %.2f MB\n", freeMem / (1024.0 * 1024.0));
   }

   // Add vectors in parallel.
   cudaError_t cudaStatus = addWithCuda(c, a, b, size);
   if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "addWithCuda failed!");
      return false;
   }

   return true;
}

bool CudaClose()
{
   // cudaDeviceReset must be called before exiting in order for profiling and
   // tracing tools such as Nsight and Visual Profiler to show complete traces.
   auto cudaStatus = cudaDeviceReset();
   if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "cudaDeviceReset failed!");
      return false;
   }

   return true;
}

#else
// Code stubs, compiled when CUDA is not available
bool CudaWork(int* c, const int* a, const int* b, unsigned int size)
{
   fprintf(stderr, "CUDA is not available on this platform.\n");
   return false;
}

bool CudaClose()
{
   return true;
}
#endif

