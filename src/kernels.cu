#include <stdio.h>

// redundant -- to make intellisense work
#ifndef __CUDACC__
#define __CUDACC__
#endif

#ifndef __CUDACC__
// Code stubs, compiled when CUDA is not available
bool DetectCUDA()
{
   fprintf(stderr, "CUDA didn't even compile. Surely it's unavailable on this platform.\nReverting to CPU.\n");
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

#else
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__ void IncKernelFunc(int *c)
{
   int i = threadIdx.x;
   c[i]++;
}

__host__ __device__ void SumParts(int* c, const int* a, const int* b)
{
   *c = *a + *b;
}

__global__ void addOtherKernel(int *c, const int *a, const int *b)
{
   int i = threadIdx.x;
   SumParts(c+i, a+i, b+i);
   c[i] += a[i]/2 + 10;
}

#ifdef TEST_CONSTANT_MEMORY
__constant__ int constantArray[20];

__global__ void kerWithConstMem(int *c, const int *a, const int *b)
{
   int i = threadIdx.x;
   SumParts(c+i, a+i, b+i);
   c[i] += a[i]/2 + constantArray[i];
}
#endif

__global__ void kerFillMapped(float *mapped, float *map2)
{
   int thrID = threadIdx.x;
   mapped[thrID] = thrID * 1.f;
   map2[thrID] = 10.f + thrID;
}

// --------------------------------------------------------------
// host part

size_t freeMem, totalMem;

void ShowMemStat(const char *title)
{
   auto st = cudaMemGetInfo(&freeMem, &totalMem);
   if (st == cudaSuccess) {
      printf("    %s: %.2f MB\n", title, freeMem / (1024.0 * 1024.0));
   }
}

// Calc core numbers based on version
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
   // Any GPUs?
   int deviceCount;
   cudaGetDeviceCount(&deviceCount);
   printf("Available GPUs: %d\n", deviceCount);
   if (!deviceCount) {
      return false;
   }

   // Can we choose one? 
   auto st = cudaSetDevice(0);// idx is available later via cudaGetDevice()
   if (st != cudaSuccess) {
      fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
      return false;
   }

   // list some props
   float perMeg  = 1.f / (1024.f * 1024.f);
   float perKilo = 1.f / (1024.f);
   for (int i = 0; i < deviceCount; ++i) {
      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, i);

      int coresPerSM = getCudaCoresPerSM(prop.major, prop.minor);
      int totalCores = prop.multiProcessorCount * coresPerSM;

      printf("Device %d: %s\n", i, prop.name);
      printf("Streaming Multiprocessors (SM): %d\n", prop.multiProcessorCount);
      if (totalCores > 0) {
         printf("CUDA Cores per SM  : %d\n", coresPerSM);
         printf("Total CUDA Cores   : %d\n", totalCores);
         printf("Async Engines      : %d\n", prop.asyncEngineCount);
      } else {
         printf("CUDA Cores per SM is not recognized: version %d.%d\n", prop.major, prop.minor);
      }
      printf("Memory global / shared / totalConstMem: %.2f MB / %.2f KB / %.2f KB\n", 
         prop.totalGlobalMem    * perMeg,     // Global memory available on device in bytes 
         prop.sharedMemPerBlock * perKilo,    // Shared memory available per block in bytes 
         prop.totalConstMem     * perKilo);   // Constant memory available on device in bytes
      printf("Map Host Memory  : %s\n", (prop.canMapHostMemory) ? "supported" : "missing");
   }

   {
      cudaError_t status = cudaMemGetInfo(&freeMem, &totalMem);
      if (status != cudaSuccess) {
         fprintf(stderr, "Error: %s\n", cudaGetErrorString(status));
         return false;
      }

      printf("GPU memory total : %.2f MB\n", totalMem * perMeg);
      printf("       available : %.2f MB\n", freeMem  * perMeg);
   }

   return true;
}

bool CopyBuffersToDev(int* c, const int* a, const int* b, 
   int *&dev_a,   int *&dev_b,   int *&dev_c, unsigned int size)
{
   auto alcSize = size * sizeof(int) * 1024 * 1024;
   int st;

   // Allocate GPU buffers for three vectors (two input, one output)    .
   st = cudaMalloc((void**)&dev_c, alcSize) 
      + cudaMalloc((void**)&dev_a, alcSize) 
      + cudaMalloc((void**)&dev_b, alcSize);
   if (st != cudaSuccess) {
      fprintf(stderr, "cudaMalloc failed!");
      return false;
   }

   ShowMemStat("after allocs ");

   // Copy input vectors from host memory to GPU buffers.
   st = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice)
      + cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
   if (st != cudaSuccess) {
      fprintf(stderr, "cudaMemcpy failed!");
      return false;
   }

   return true;
}

bool FillConstantMem()
{
   #ifdef TEST_CONSTANT_MEMORY
      // fill
      int host_arr[20];
      for (int i = 0; i < 20; i++) {
         host_arr[i] = 10*(i+1);
      }

      // pass to device
      auto st = cudaMemcpyToSymbol(constantArray, host_arr, sizeof(host_arr));
      if (st != cudaSuccess) {
         fprintf(stderr, "FillConstantMem failed!");
         return false;
      }

      ShowMemStat("after const  ");
   #endif

   return true;
}

float* hostArrMapped = nullptr, *host2ndMapped = nullptr;
float* devArrMapped = nullptr, *devSecondMapped = nullptr;

void MapSomeMemory()
{
   size_t size = 6  * 1024 * 1024 * sizeof(float);
   cudaHostAlloc((void**)&hostArrMapped, size, cudaHostAllocMapped);
   cudaHostGetDevicePointer((void**)&devArrMapped, hostArrMapped, 0);

   for (auto i = 0; i < 200; i++) {
      hostArrMapped[i] = 888;
   }

   ShowMemStat("after map-1  ");

   cudaHostAlloc((void**)&host2ndMapped, size, cudaHostAllocMapped);
   cudaHostGetDevicePointer((void**)&devSecondMapped, host2ndMapped, 0);

   ShowMemStat("after map-2  ");

   // Now we have:
   // -- hostArrMapped here and 
   // -- devArrMapped there on the device memory
   // For example, launch kernel using devArrMapped
}

int PokeCudaAPI(int *c, const int *a, const int *b, unsigned int size)
{
   int *dev_a = nullptr;
   int *dev_b = nullptr;
   int *dev_c = nullptr;
   int st = cudaSuccess;// cudaError_t

   if (!FillConstantMem()) {
      goto CleanUp;
   }

   if (!CopyBuffersToDev(c, a, b, dev_a, dev_b, dev_c, size)) {
      goto CleanUp;
   }

#ifdef TEST_CONSTANT_MEMORY
   // Launch a kernel on the GPU with one thread for each element.
   //addKernel<<<1, size>>>(dev_c, dev_a, dev_b);
   //addOtherKernel<<<1, size>>>(dev_c, dev_a, dev_b);
   //IncKernelFunc<<<1, size>>>(dev_c);
   kerWithConstMem<<<1, size>>>(dev_c, dev_a, dev_b);
   if (cudaGetLastError() != cudaSuccess) {
      fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaGetLastError()));
      goto CleanUp;
   }
#endif

   // Fill mapped memory -- one thread for each element.
   kerFillMapped<<<1, size>>>(devArrMapped, devSecondMapped);
   if (cudaGetLastError() != cudaSuccess) {
      fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaGetLastError()));
      goto CleanUp;
   }
                           
   // <<<<<<<<<<<< MAIN SYNC >>>>>>>>>>>>>>>>
   //    cudaDeviceSynchronize waits for the kernel to finish, and returns
   //    any errors encountered during the launch.
   st = cudaDeviceSynchronize();
   if (st != cudaSuccess) {
      fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", st);
      goto CleanUp;
   }

   ShowMemStat("after sync   ");

   // Copy output vector from GPU buffer to host memory.
   st = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
   if (st != cudaSuccess) {
      fprintf(stderr, "cudaMemcpy failed!");
      goto CleanUp;
   }

CleanUp:
   cudaFree(dev_c);
   cudaFree(dev_a);
   cudaFree(dev_b);
   cudaFreeHost(hostArrMapped);
   cudaFreeHost(host2ndMapped);
    
   ShowMemStat("after free   ");

   return st;
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
   ShowMemStat("device reset ");

   return true;
}

bool CudaWork(int* c, const int* a, const int* b, unsigned int size)
{
   DetectCUDA();

   MapSomeMemory();

   // Add vectors in parallel.
   auto st = PokeCudaAPI(c, a, b, size);
   if (st) {
      fprintf(stderr, "PokeCudaAPI failed!");
      return false;
   }

   return true;
}

#endif // cudacc

