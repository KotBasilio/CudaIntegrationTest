#include <stdio.h>

// redundant -- to make intellisense work
#ifndef __CUDACC__
#define __CUDACC__
#endif

#ifdef __CUDACC__
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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

// using CUDA to add vectors in parallel.
int addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
   size_t freeMem, totalMem;
   int *dev_a = 0;
   int *dev_b = 0;
   int *dev_c = 0;
   auto alcSize = size * sizeof(int) * 1024 * 1024;
   int st;//cudaError_t st;

   // Choose which GPU to run on, change this on a multi-GPU system.
   st = cudaSetDevice(0);
   if (st != cudaSuccess) {
      fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
      goto Error;
   }

   // Allocate GPU buffers for three vectors (two input, one output)    .
   st = cudaMalloc((void**)&dev_c, alcSize) 
      + cudaMalloc((void**)&dev_a, alcSize) 
      + cudaMalloc((void**)&dev_b, alcSize);
   if (st != cudaSuccess) {
      fprintf(stderr, "cudaMalloc failed!");
      goto Error;
   }

   st = cudaMemGetInfo(&freeMem, &totalMem);
   if (st == cudaSuccess) {
      printf("    after allocs : %.2f MB\n", freeMem / (1024.0 * 1024.0));
   }

   // Copy input vectors from host memory to GPU buffers.
   st = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
   if (st != cudaSuccess) {
      fprintf(stderr, "cudaMemcpy failed!");
      goto Error;
   }

   st = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
   if (st != cudaSuccess) {
      fprintf(stderr, "cudaMemcpy failed!");
      goto Error;
   }

   // Launch a kernel on the GPU with one thread for each element.
   //addKernel<<<1, size>>>(dev_c, dev_a, dev_b);
   addOtherKernel<<<1, size>>>(dev_c, dev_a, dev_b);
   //IncKernelFunc<<<1, size>>>(dev_c);
   //IncExtKernel<<<1, size>>>(dev_c);

   // Check for any errors launching the kernel
   st = cudaGetLastError();
   if (st != cudaSuccess) {
      fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString((cudaError_t)st));
      goto Error;
   }

   st = cudaMemGetInfo(&freeMem, &totalMem);
   if (st == cudaSuccess) {
      printf("    after kernels: %.2f MB\n", freeMem / (1024.0 * 1024.0));
   }

   // cudaDeviceSynchronize waits for the kernel to finish, and returns
   // any errors encountered during the launch.
   st = cudaDeviceSynchronize();
   if (st != cudaSuccess) {
      fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", st);
      goto Error;
   }

   st = cudaMemGetInfo(&freeMem, &totalMem);
   if (st == cudaSuccess) {
      printf("    after sync   : %.2f MB\n", freeMem / (1024.0 * 1024.0));
   }

   // Copy output vector from GPU buffer to host memory.
   st = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
   if (st != cudaSuccess) {
      fprintf(stderr, "cudaMemcpy failed!");
      goto Error;
   }

Error:
   cudaFree(dev_c);
   cudaFree(dev_a);
   cudaFree(dev_b);
    
   return st;
}

// Helper function for calculating core numbers based on version
int getCudaCoresPerSM(int major, int minor) {
   if (major == 8 && minor == 6) return 128; // Ampere (RTX 30xx)
   if (major == 8 && minor == 0) return 64;  // Ampere (A100)
   if (major == 7 && minor == 5) return 64;  // Turing (T4)
   if (major == 7 && minor == 0) return 64;  // Volta  (V100)
   if (major == 6 && minor == 1) return 128; // Pascal (P100)
   return -1; // unknonw
}

bool DetectCUDA()
{
   int deviceCount;
   cudaGetDeviceCount(&deviceCount);
   printf("Available GPUs: %d\n", deviceCount);
   if (!deviceCount) {
      return false;
   }

   for (int i = 0; i < deviceCount; ++i) {
      cudaDeviceProp deviceProp;
      cudaGetDeviceProperties(&deviceProp, i);
      printf("Device %d: %s\n", i, deviceProp.name);
   }

   {
      int device;
      cudaGetDevice(&device);

      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, device);

      int coresPerSM = getCudaCoresPerSM(prop.major, prop.minor);
      int totalCores = prop.multiProcessorCount * coresPerSM;

      printf("GPU name: %s\n", prop.name);
      printf("Streaming Multiprocessors (SM): %d\n", prop.multiProcessorCount);
      if (totalCores > 0) {
         printf("CUDA Cores per SM: %d\n", coresPerSM);
         printf("Total CUDA Cores : %d\n", totalCores);
      } else {
         printf("CUDA Cores per SM is not recognized: version %d.%d\n", prop.major, prop.minor);
      }
   }

   size_t freeMem, totalMem;
   {
      cudaError_t status = cudaMemGetInfo(&freeMem, &totalMem);
      if (status != cudaSuccess) {
         fprintf(stderr, "Error: %s\n", cudaGetErrorString(status));
         return false;
      }

      printf("GPU memory total : %.2f MB\n", totalMem / (1024.0 * 1024.0));
      printf("       available : %.2f MB\n", freeMem / (1024.0 * 1024.0));
   }

   return true;
}

bool CudaWork(int* c, const int* a, const int* b, unsigned int size)
{
   DetectCUDA();

   // Add vectors in parallel.
   auto st = addWithCuda(c, a, b, size);
   if (st) {
      fprintf(stderr, "addWithCuda failed!");
      return false;
   }

   return true;
}

bool CudaClose()
{
   // cudaDeviceReset must be called before exiting in order for profiling and
   // tracing tools such as Nsight and Visual Profiler to show complete traces.
   auto st = cudaDeviceReset();
   if (st != cudaSuccess) {
      fprintf(stderr, "cudaDeviceReset failed!");
      return false;
   }

   // final stat
   printf("CUDA reset the device; ");
   size_t freeMem, totalMem;
   st = cudaMemGetInfo(&freeMem, &totalMem);
   if (st == cudaSuccess) {
      printf("memory after reset : %.2f MB\n", freeMem / (1024.0 * 1024.0));
   }

   return true;
}

#else // Code stubs, compiled when CUDA is not available

bool DetectCUDA()
{
   fprintf(stderr, "CUDA didn't even compile. Surely it's unavailable on this platform.\nIt's very sad to work without CUDA.\n");
   return false;
}

bool CudaWork(int* c, const int* a, const int* b, unsigned int size)
{
   return false;
}

bool CudaClose()
{
   return true;
}

#endif

