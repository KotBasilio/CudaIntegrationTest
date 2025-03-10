/*
   2024 GPU acceleration by Serge Mironov
   requires CUDA technology
   its toolkit is available here:
   https://developer.nvidia.com/cuda-toolkit
*/

#include <iostream>
#include <iomanip>
#include <sstream>

#include "../include/dll.h"
#include "SolverIF.h"
#include "Init.h"
#include "ABsearch.h"
#include "System.h"
#include "Scheduler.h"

#include "LogSubsys.cu"

// device part of Carpenter
class CarpImpl {
   System sysdep;
   //Memory memory;
   //Scheduler scheduler;

public:
   __device__ void SolveBoardOnGPU(int idx);

   __device__ int  SolveBoard(
      const struct deal &dl,
      const int target,
      const int solutions,
      const int mode,
      struct futureTricks * futp,
      struct ThreadData* thrp);

   futureTricks* d_futures = nullptr;  // Persistent GPU buffer for results
   boards* d_chunk = nullptr;          // GPU copy of chunk
   CarpImpl* d_carp = nullptr;         // GPU copy of this class
   int noOfBoards = 0;

   CarpImpl() {
      cudaMalloc(&d_futures, MAXNOOFBOARDS * sizeof(futureTricks));
      cudaMalloc(&d_chunk, sizeof(boards));
      cudaMalloc(&d_carp, sizeof(CarpImpl));
   }

   ~CarpImpl() {
      cudaFree(d_futures);
      cudaFree(d_chunk);
      cudaFree(d_carp);
   }

   void SyncUp(boards& chunk) {
      if (d_carp && d_chunk) {
         // store total
         noOfBoards = chunk.noOfBoards;

         // Copy chunk from host to device before running kernel
         cudaMemcpy(d_chunk, &chunk, sizeof(*d_chunk), cudaMemcpyHostToDevice);

         // Copy latest Carpenter state to GPU
         cudaMemcpy(d_carp, this, sizeof(*d_carp), cudaMemcpyHostToDevice);
      } else {
         noOfBoards = 0;
      }
   }

   void SyncDown(futureTricks* h_Futures, int maxFut) {
      if (d_futures && (noOfBoards == maxFut)) {
         // Copy results back from GPU to CPU
         cudaDeviceSynchronize();
         cudaMemcpy(h_Futures, d_futures, maxFut * sizeof(futureTricks), cudaMemcpyDeviceToHost);
      } else {
         printf("... mismatch boards count: %d ", noOfBoards);
      }
   }
};

Carpenter::Carpenter()
{
   // LogSubsystem is __managed__, so we initialize it explicitly
   myLog.Initialize();

   Himpl = new CarpImpl;
}

Carpenter::~Carpenter()
{
   if (Himpl) {
      // destory
      delete Himpl;
      Himpl = nullptr;

      // Print the log on destruction
      cudaDeviceSynchronize();
      myLog.PrintLog();
   }
}

extern __constant__ int d_highestRank[8192];
extern __constant__ int d_lowestRank[8192];

void Carpenter::Overlook(const futureTricks* h_Futures, int maxFut)
{
   // Copy mFutures[] to device (GPU)
   cudaMemcpy(Himpl->d_futures, h_Futures, maxFut * sizeof(futureTricks), cudaMemcpyHostToDevice);
}

__global__ void CarpFanOut(CarpImpl* d_carp)
{
   int idx = threadIdx.x;
   d_carp->SolveBoardOnGPU(idx);
}

void Carpenter::SolveChunk(boards& chunk)
{
   printf("... %d boards on GPU ... ", chunk.noOfBoards);
   if (Himpl) {
      // Prepare Carpenter on GPU
      Himpl->SyncUp(chunk);

      // Launch kernels using device-side Carpenter
      CarpFanOut <<< 1, chunk.noOfBoards >>> (Himpl->d_carp);
   }
}

void Carpenter::SyncDown(futureTricks* h_Futures, int maxFut)
{
   if (Himpl) {
      printf("\n Sync down %d boards from GPU ... ", maxFut);
      Himpl->SyncDown(h_Futures, maxFut);
   }
}


// ==========
__device__ void CarpImpl::SolveBoardOnGPU(int idx)
{

   if (idx == 16) {
      LOG(INVALID_ARGUMENT);
   }

   // Ensure we don’t go out of bounds
   if (idx < d_chunk->noOfBoards) {
      // Write garbage directly into mFutures[]
      d_futures[idx].cards = 999;   // Fake number of tricks
      d_futures[idx].suit[0] = 3;   // Fake suit
      d_futures[idx].rank[0] = 14;  // Fake rank (Ace)
      d_futures[idx].score[0] = -99;// Fake score
   }

   //deal* myDeal = chunk.deals + idx;
   //d_carp->SolveBoard(
   //   chunk.deals[i],
   //   chunk.target[i],
   //   chunk.solutions[i],
   //   chunk.mode[i],
   //   nullptr,
   //   nullptr
   //);
}

__device__ int CarpImpl::SolveBoard(const deal& dl, const int target, const int solutions, const int mode, futureTricks* futp, ThreadData* thrp)
{
   return 1;
}
