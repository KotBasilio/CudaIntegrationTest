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

class CarpImpl {
   System sysdep;
   //Memory memory;
   //Scheduler scheduler;

public:
   CarpImpl() {
      // LogSubsystem is __managed__, so we initialize it explicitly
      //cudaDeviceSynchronize();
      myLog.Initialize();
   }

   ~CarpImpl() {
      // Print the log on destruction
      cudaDeviceSynchronize();
      myLog.PrintLog();
   }
};

Carpenter::Carpenter()
{
   Instance = new CarpImpl();
}

Carpenter::~Carpenter()
{
   if (Instance) {
      delete Instance;
      Instance = nullptr;
   }
}

__device__ int Carpenter::SolveBoard(const deal& dl, const int target, const int solutions, const int mode, futureTricks* futp, ThreadData* thrp)
{
   return 1;
}

extern __constant__ int d_highestRank[8192];
extern __constant__ int d_lowestRank[8192];

__global__ void CarpFanOut(Carpenter * carp, boards & chunk)
{
   int i = threadIdx.x;
   i += d_lowestRank[i];
   if (i == 164) {
      LOG(SUCCESS);
   }
   //deal* myDeal = chunk.deals + i;
   //carp->SolveBoard(
   //   chunk.deals[i],
   //   chunk.target[i],
   //   chunk.solutions[i],
   //   chunk.mode[i],
   //   nullptr,
   //   nullptr
   //);
}

void Carpenter::SolveChunk(boards& chunk)
{
   printf("...");
   CarpFanOut << <1, chunk.noOfBoards >> > (this, chunk);
}

