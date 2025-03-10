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
   //if (i == 164) {
   //   LOG(SUCCESS);
   //}
   if (i == 16) {
      LOG(INVALID_ARGUMENT);
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

//__global__ void CarpFanGarbage(Carpenter *carp, boards & chunk) {
//   int idx = threadIdx.x;
//
//   // Ensure we don't go out of bounds
//   if (idx < chunk.noOfBoards) {
//      // Write garbage values into mFutures (futureTricks struct)
//      chunk.futures[idx].cards = 999;  // Garbage value for number of tricks
//      chunk.futures[idx].suit[0] = 3;  // Fake suit
//      chunk.futures[idx].rank[0] = 14; // Fake rank (Ace)
//      chunk.futures[idx].score[0] = -99; // Fake score
//   }
//}

void Carpenter::SolveChunk(boards& chunk)
{
   printf("... %d boards on GPU ... ", chunk.noOfBoards);
   CarpFanOut <<< 1, chunk.noOfBoards >>> (this, chunk);
   //CarpFanGarbage <<< 1, chunk.noOfBoards >>> (this, chunk);
}

