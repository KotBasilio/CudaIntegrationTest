#define  _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <malloc.h>
#include <string.h>
#include <cassert>

#include "TestSuite.h"

void CTestSuite::PrepareChunk(boards& _chunkBoards)
{
   int idxToadd = 0;
   int threadBegin = MAX_THREADS_IN_TEST - 1;
   for (int threadIndex = threadBegin; threadIndex >= 0; threadIndex--) {
      deal dl;
      int handno = 0;
      for (; handno < TEST_NUM_EXAMP_PKG; handno++) {
         FillDeal(dl, handno);
         _chunkBoards.deals[idxToadd] = dl;
         _chunkBoards.target[idxToadd] = -1;
         _chunkBoards.solutions[idxToadd] = 3;
         _chunkBoards.mode[idxToadd] = 0;
         idxToadd++;

         _chunkBoards.deals[idxToadd] = dl;
         _chunkBoards.target[idxToadd] = -1;
         _chunkBoards.solutions[idxToadd] = 2;
         _chunkBoards.mode[idxToadd] = 0;
         idxToadd++;
      }

      for (; handno < TEST_HOLDINGS_COUNT; handno++) {
         FillDeal(dl, handno);
         _chunkBoards.deals[idxToadd] = dl;
         _chunkBoards.target[idxToadd] = -1;
         _chunkBoards.solutions[idxToadd] = 1;
         _chunkBoards.mode[idxToadd] = 0;
         idxToadd++;

         dl.trump = 0;
         dl.first = 0;
         _chunkBoards.deals[idxToadd] = dl;
         _chunkBoards.target[idxToadd] = -1;
         _chunkBoards.solutions[idxToadd] = 1;
         _chunkBoards.mode[idxToadd] = 0;
         idxToadd++;
      }
   }
   assert(idxToadd == TOTAL_FUTURES_IN_TEST);
   _chunkBoards.noOfBoards = idxToadd;
}

void CTestSuite::CarpenterSolve()
{
   printf("Testing Carpenter()");
   bool isAllright = true;

   // prepare all boards
   static boards _chunkBoards;
   PrepareChunk(_chunkBoards);

   // run with CUDA
   Carpenter carp;
   carp.SmallTest();
   carp.SolveChunk(_chunkBoards);

   // compare
   ControlSolvedBoards(isAllright);
}

__global__ void kerCarpTest(void)
{
   int i = threadIdx.x;
   i++;
}

void Carpenter::SmallTest()
{
   printf("...");
   unsigned int size = MAX_THREADS_IN_TEST;
   kerCarpTest << <1, size >> > ();
}

