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

// from Init.cpp
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

void InitConstants();

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

string System::str(DDSInfo* info) const
{
   const string st = "ss.str()";
   strcpy_s(info->systemString, sizeof(info->systemString), st.c_str());
   return st;
}

bool System::ThreadOK(const int thrId) const
{
   return (thrId >= 0 && thrId < numThreads);
}

