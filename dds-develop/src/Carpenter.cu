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

extern System sysdep;
extern Memory memory;
extern Scheduler scheduler;

__global__ void CarpFanOut(Carpenter * carp, boards & chunk)
{
   int i = threadIdx.x;
   deal* myDeal = chunk.deals + i;
   carp->Solve(myDeal);
}

void Carpenter::SolveChunk(boards& chunk)
{
   printf("...");
   CarpFanOut << <1, chunk.noOfBoards >> > (this, chunk);
}

class CarpImpl
{

};

Carpenter::Carpenter()
{
}

Carpenter::~Carpenter()
{
   printf("~");
}

__device__ void Carpenter::Solve(deal* myDeal)
{

}

__device__ int Carpenter::SolveBoard(const deal& dl, const int target, const int solutions, const int mode, futureTricks* futp, ThreadData* thrp)
{
   return 1;
}


