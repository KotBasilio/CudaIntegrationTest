/*
   DDS, a bridge double dummy solver.

   Copyright (C) 2006-2014 by Bo Haglund /
   2014-2018 by Bo Haglund & Soren Hein.

   See LICENSE and README.
*/


#include "../include/dll.h"
#include "Memory.h"
#include "System.h"
#include "Scheduler.h"
#include "ThreadMgr.h"

#define DDS_SYSTEM_THREAD_BASIC 0
#define DDS_SYSTEM_THREAD_SIZE 1

System sysdep;
Memory memory;
Scheduler scheduler;
ThreadMgr threadMgr;

Memory::Memory() {}
Memory::~Memory()
{
  Memory::Resize(0, DDS_TT_SMALL, 0, 0);
}

void Memory::ReturnThread(const unsigned thrId)
{
   memory[thrId]->transTable->ReturnAllMemory();
   memory[thrId]->memUsed = Memory::MemoryInUseMB(thrId);
}

void Memory::Resize(
  const unsigned n,
  const TTmemory flag,
  const int memDefault_MB,
  const int memMaximum_MB)
{
  if (memory.size() == n)
    return;

  if (memory.size() > n)
  {
    // Downsize.
    for (unsigned i = n; i < memory.size(); i++)
    {
      delete memory[i]->transTable;
      delete memory[i];
    }
    memory.resize(static_cast<unsigned>(n));
    threadSizes.resize(static_cast<unsigned>(n));
  }
  else
  {
    // Upsize.
    size_t oldSize = memory.size();
    memory.resize(n);
    threadSizes.resize(n);
    for (size_t i = oldSize; i < n; i++)
    {
      memory[i] = new ThreadData();
      if (flag == DDS_TT_SMALL)
      {
         memory[i]->transTable = new TransTableS;
         threadSizes[i] = "S";
      }
      else
      {
         memory[i]->transTable = new TransTableL;
         threadSizes[i] = "L";
      }

      memory[i]->transTable->SetMemoryDefault(memDefault_MB);
      memory[i]->transTable->SetMemoryMaximum(memMaximum_MB);

      memory[i]->transTable->MakeTT();
    }
  }
}

unsigned Memory::NumThreads() const
{
  return static_cast<unsigned>(memory.size());
}

ThreadData * Memory::GetPtr(const unsigned thrId)
{
  if (thrId >= memory.size())
  {
    printf("Memory::GetPtr: %lu  vs. %llu\n", thrId, memory.size());
    return nullptr;
  }
  return memory[thrId];
}


double Memory::MemoryInUseMB(const unsigned thrId) const
{
  return memory[thrId]->transTable->MemoryInUse() +
    8192. * sizeof(relRanksType) / static_cast<double>(1024.);
}


string Memory::ThreadSize(const unsigned thrId) const
{
  return threadSizes[thrId];
}

// ------------------------------------- from Init.cpp
int lho[DDS_HANDS] = { 1, 2, 3, 0 };
int rho[DDS_HANDS] = { 3, 0, 1, 2 };
int partner[DDS_HANDS] = { 2, 3, 0, 1 };

// There is no particular reason for the different types here,
// other than historical ones. They could all be char's for
// memory reasons, or all be int's for performance reasons.

int highestRank[8192];
int lowestRank[8192];
int counttable[8192];
char relRank[8192][15];
unsigned short int winRanks[8192][14];

moveGroupType groupData[8192];

void InitConstants()
{
   // highestRank[aggr] is the highest absolute rank in the
   // suit represented by aggr. The absolute rank is 2 .. 14.
   // Similarly for lowestRank.
   highestRank[0] = 0;
   lowestRank [0] = 0;
   for (int aggr = 1; aggr < 8192; aggr++)
   {
      for (int r = 14; r >= 2; r--)
      {
         if (aggr & bitMapRank[r])
         {
            highestRank[aggr] = r;
            break;
         }
      }
      for (int r = 2; r <= 14; r++)
      {
         if (aggr & bitMapRank[r])
         {
            lowestRank[aggr] = r;
            break;
         }
      }
   }

   // The use of the counttable to give the number of bits set to
   // one in an integer follows an implementation by Thomas Andrews.

   // counttable[aggr] is the number of '1' bits (binary weight)
   // in aggr.
   for (int aggr = 0; aggr < 8192; aggr++)
   {
      counttable[aggr] = 0;
      for (int r = 0; r < 13; r++)
      {
         if (aggr & (1 << r))
         {
            counttable[aggr]++;
         }
      }
   }

   // relRank[aggr][absolute rank] is the relative rank of
   // that absolute rank in the suit represented by aggr.
   // The relative rank is 2 .. 14.
   memset(relRank[0], 0, 15);
   for (int aggr = 1; aggr < 8192; aggr++)
   {
      char ord = 0;
      for (int r = 14; r >= 2; r--)
      {
         if (aggr & bitMapRank[r])
         {
            ord++;
            relRank[aggr][r] = ord;
         }
      }
   }

   // winRanks[aggr][leastWin] is the absolute suit represented
   // by aggr, but limited to its top "leastWin" bits.
   for (int aggr = 0; aggr < 8192; aggr++)
   {
      winRanks[aggr][0] = 0;
      for (int leastWin = 1; leastWin < 14; leastWin++)
      {
         int res = 0;
         int nextBitNo = 1;
         for (int r = 14; r >= 2; r--)
         {
            if (aggr & bitMapRank[r])
            {
               if (nextBitNo <= leastWin)
               {
                  res |= bitMapRank[r];
                  nextBitNo++;
               }
               else
                  break;
            }
         }
         winRanks[aggr][leastWin] = static_cast<unsigned short>(res);
      }
   }

   // groupData[ris] is a representation of the suit (ris is
   // "rank in suit") in terms of runs of adjacent bits.
   // 1 1100 1101 0110
   // has 4 runs, so lastGroup is 3, and the entries are
   // 0: 4 and 0x0002, gap 0x0000 (lowest gap unused, though)
   // 1: 6 and 0x0000, gap 0x0008
   // 2: 9 and 0x0040, gap 0x0020
   // 3: 14 and 0x0c00, gap 0x0300

   int topside[15] =
   {
      0x0000, 0x0000, 0x0000, 0x0001, // 2, 3,
      0x0003, 0x0007, 0x000f, 0x001f, // 4, 5, 6, 7,
      0x003f, 0x007f, 0x00ff, 0x01ff, // 8, 9, T, J,
      0x03ff, 0x07ff, 0x0fff          // Q, K, A
   };

   int botside[15] =
   {
      0xffff, 0xffff, 0x1ffe, 0x1ffc, // 2, 3,
      0x1ff8, 0x1ff0, 0x1fe0, 0x1fc0, // 4, 5, 6, 7,
      0x1f80, 0x1f00, 0x1e00, 0x1c00, // 8, 9, T, J,
      0x1800, 0x1000, 0x0000          // Q, K, A
   };

   // So the bit vector in the gap between a top card of K
   // and a bottom card of T is
   // topside[K] = 0x07ff &
   // botside[T] = 0x1e00
   // which is 0x0600, the binary code for QJ.

   groupData[0].lastGroup = -1;

   groupData[1].lastGroup = 0;
   groupData[1].rank[0] = 2;
   groupData[1].sequence[0] = 0;
   groupData[1].fullseq[0] = 1;
   groupData[1].gap[0] = 0;

   int topBitRank = 1;
   int nextBitRank = 0;
   int topBitNo = 2;
   int g;

   for (int ris = 2; ris < 8192; ris++)
   {
      if (ris >= (topBitRank << 1))
      {
         // Next top bit
         nextBitRank = topBitRank;
         topBitRank <<= 1;
         topBitNo++;
      }

      groupData[ris] = groupData[ris ^ topBitRank];

      if (ris & nextBitRank) // 11... Extend group
      {
         g = groupData[ris].lastGroup;
         groupData[ris].rank[g]++;
         groupData[ris].sequence[g] |= nextBitRank;
         groupData[ris].fullseq[g] |= topBitRank;
      }
      else // 10... New group
      {
         g = ++groupData[ris].lastGroup;
         groupData[ris].rank[g] = topBitNo;
         groupData[ris].sequence[g] = 0;
         groupData[ris].fullseq[g] = topBitRank;
         groupData[ris].gap[g] =
            topside[topBitNo] & botside[ groupData[ris].rank[g - 1] ];
      }
   }
}

void System::Reset()
{
   runCat = DDS_RUN_SOLVE;
   numThreads = 1;

   preferredSystem = DDS_SYSTEM_THREAD_BASIC;
   availableSystem.resize(DDS_SYSTEM_THREAD_SIZE);
   availableSystem[DDS_SYSTEM_THREAD_BASIC] = true;

   //RunPtrList.resize(DDS_SYSTEM_THREAD_SIZE);
   //RunPtrList[DDS_SYSTEM_THREAD_BASIC] = &System::RunThreadsBasic; 

   //CallbackSimpleList.resize(DDS_RUN_SIZE);
   //CallbackSimpleList[DDS_RUN_SOLVE] = SolveChunkCommon;
   //CallbackSimpleList[DDS_RUN_CALC] = CalcChunkCommon;
   //CallbackSimpleList[DDS_RUN_TRACE] = PlayChunkCommon;

   //CallbackDuplList.resize(DDS_RUN_SIZE);
   //CallbackDuplList[DDS_RUN_SOLVE] = DetectSolveDuplicates;
   //CallbackDuplList[DDS_RUN_CALC] = DetectCalcDuplicates;
   //CallbackDuplList[DDS_RUN_TRACE] = DetectPlayDuplicates;

   //CallbackSingleList.resize(DDS_RUN_SIZE);
   //CallbackSingleList[DDS_RUN_SOLVE] = SolveSingleCommon;
   //CallbackSingleList[DDS_RUN_CALC] = CalcSingleCommon;
   //CallbackSingleList[DDS_RUN_TRACE] = PlaySingleCommon;

   //CallbackCopyList.resize(DDS_RUN_SIZE);
   //CallbackCopyList[DDS_RUN_SOLVE] = CopySolveSingle;
   //CallbackCopyList[DDS_RUN_CALC] = CopyCalcSingle;
   //CallbackCopyList[DDS_RUN_TRACE] = CopyPlaySingle;
}


void SetResources()
{
   // Operate large threads only
   int thrMax = 12;
   int noOfThreads, noOfLargeThreads;
   noOfThreads = thrMax;
   noOfLargeThreads = thrMax;

   int memMaxMB = 11387;
   sysdep.RegisterParams(noOfThreads, memMaxMB);
   scheduler.RegisterThreads(noOfThreads);

   // Clear the thread memory and fill it up again.
   memory.Resize(0, DDS_TT_SMALL, 0, 0);
   memory.Resize(static_cast<unsigned>(noOfLargeThreads),
      DDS_TT_LARGE, THREADMEM_LARGE_DEF_MB, THREADMEM_LARGE_MAX_MB);

   threadMgr.Reset(noOfThreads);

   InitConstants();

   //runCat	DDS_RUN_SOLVE (0)	RunMode
   //   numThreads	12	int
   //   sysMem_MB	11387	int
}
