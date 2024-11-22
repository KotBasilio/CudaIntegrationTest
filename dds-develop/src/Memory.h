/*
   DDS, a bridge double dummy solver.

   Copyright (C) 2006-2014 by Bo Haglund /
   2014-2018 by Bo Haglund & Soren Hein.

   See LICENSE and README.

   2024 GPU acceleration by Serge Mironov
*/

#ifndef DDS_MEMORY_H
#define DDS_MEMORY_H

#include "TransTable.h"
#include "TransTableL.h"
#include "Moves.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

enum TTmemory
{
  DDS_TT_SMALL = 0,
  DDS_TT_LARGE = 1
};

struct WinnerEntryType
{
  int suit;
  int winnerRank;
  int winnerHand;
  int secondRank;
  int secondHand;
};

struct WinnersType
{
  int number;
  WinnerEntryType winner[4];
};

struct ThreadData
{
  int nodeTypeStore[DDS_HANDS];
  int iniDepth;
  bool val;

  unsigned short int suit[DDS_HANDS][DDS_SUITS];
  int trump;

  pos lookAheadPos; // Recursive alpha-beta data
  bool analysisFlag;
  unsigned short int lowestWin[50][DDS_SUITS];
  WinnersType winners[13];
  moveType forbiddenMoves[14];
  moveType bestMove[50];
  moveType bestMoveTT[50];

  double memUsed;
  int nodes;
  int trickNodes;

  // Constant for a given hand.
  // 960 KB
  relRanksType rel[8192];

  TransTable * transTable;

  Moves moves;
};


class Memory
{
  private:
    vector<ThreadData *> memory;

  public:
    Memory();
    ~Memory();

    void ReturnThread(int thrId);

    void Resize(
      const unsigned n,
      const TTmemory flag,
      const int memDefault_MB,
      const int memMaximum_MB);

    unsigned NumThreads() const;
    ThreadData * GetPtr(int thrId);
    double MemoryInUseMB(int thrId) const;
};

#endif
