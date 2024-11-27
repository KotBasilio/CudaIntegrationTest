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

__global__ void CarpFanOut(Carpenter * carp, boards & chunk)
{
   int i = threadIdx.x;
   //deal* myDeal = chunk.deals + i;
   carp->SolveBoard(
      chunk.deals[i],
      chunk.target[i],
      chunk.solutions[i],
      chunk.mode[i],
      nullptr,
      nullptr
   );
}

void Carpenter::SolveChunk(boards& chunk)
{
   printf("...");
   CarpFanOut << <1, chunk.noOfBoards >> > (this, chunk);
}

class CarpImpl
{
   System sysdep;
   //Memory memory;
   //Scheduler scheduler;

public:
   CarpImpl() {}
   ~CarpImpl() {}
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
      printf("~");
   }
}

__global__ void kerCarpTest(void)
{
   int i = threadIdx.x;
   i++;
}

void Carpenter::SmallTest()
{
   printf("...");
   unsigned int size = 100;
   kerCarpTest << <1, size >> > ();
}

//__device__ void Carpenter::Dummy(deal* myDeal)
//{
//
//}
//

__device__ int Carpenter::SolveBoard(const deal& dl, const int target, const int solutions, const int mode, futureTricks* futp, ThreadData* thrp)
{
   return 1;
}


