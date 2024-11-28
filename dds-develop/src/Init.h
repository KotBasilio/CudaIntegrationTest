/*
   DDS, a bridge double dummy solver.

   Copyright (C) 2006-2014 by Bo Haglund /
   2014-2018 by Bo Haglund & Soren Hein.

   See LICENSE and README.

   2024 GPU acceleration by Serge Mironov
*/

#ifndef DDS_INIT_H
#define DDS_INIT_H

#include "dds.h"
#include "Memory.h"


void SetDeal(ThreadData * thrp);

void SetDealTables(ThreadData * thrp);

void InitWinners(
  const deal& dl,
  pos& posPoint,
  ThreadData const * thrp);

void ResetBestMoves(ThreadData * thrp);

inline void DumpInput(const int errCode, const deal& dl, const int target, const int solutions, const int mode) {}

enum class ErrorCode {
   SUCCESS = 0,
   FAILURE,
   OUT_OF_BOUNDS,
   INVALID_ARGUMENT,
   UNKNOWN_ERROR
};

const int LOG_BUFFER_SIZE = 20480; // 20KB 

class LogSubsystem {
private:
   char buffer[LOG_BUFFER_SIZE];
   int pos;

   __device__ void AppendToBuffer(const char* str);

public:
   void Initialize();
   void PrintLog() const;

   __device__ void Log(ErrorCode code, const char* module, int line);

   __device__ void Clear() {
      pos = 0;
      buffer[0] = '\0';
   }
};

// Macro for simplified logging
#define LOG(errorCode) log->Log(ErrorCode::errorCode, __FILE__, __LINE__)

#endif
