/*
   DDS, a bridge double dummy solver.

   Copyright (C) 2006-2014 by Bo Haglund /
   2014-2018 by Bo Haglund & Soren Hein.

   See LICENSE and README.
*/

#ifndef DDS_SYSTEM_H
#define DDS_SYSTEM_H

/*
   This class encapsulates all the system-dependent stuff.
 */

#include <string>
#include <vector>

#include "dds.h"

using namespace std;

class System
{
private:

   RunMode runCat; // SOLVE / CALC / PLAY

   int numThreads;
   int sysMem_MB;
   int thrDef_MB;
   int thrMax_MB;

public:
   System() { Reset(); }
   ~System() {}

   void Reset();

   int RegisterParams(const int nThreads, const int mem_usable_MB)
   {
      // No upper limit -- caveat emptor.
      if (nThreads < 1)
         return RETURN_THREAD_INDEX;

      numThreads = nThreads;
      sysMem_MB = mem_usable_MB;
      return RETURN_NO_FAULT;
   }

   bool ThreadOK(const int thrId) const;
};

#endif

