/*
   DDS, a bridge double dummy solver.

   Copyright (C) 2006-2014 by Bo Haglund /
   2014-2018 by Bo Haglund & Soren Hein.

   See LICENSE and README.
*/

#ifndef DDS_SCHEDULER_H
#define DDS_SCHEDULER_H

#include <atomic>
#include <vector>

#include "dds.h"

using namespace std;

#define HASH_MAX 200

#define START_BLOCK_TIMER 1
#define END_BLOCK_TIMER 1
#define START_THREAD_TIMER(a) 1
#define END_THREAD_TIMER(a) 1


struct schedType
{
  int number;
  int repeatOf;
};


class Scheduler
{
  private:

    struct listType
    {
      int first;
      int last;
      int length;
    };

    struct groupType
    {
      int strain;
      int hash;
      int pred;
      int actual;
      int head;
      int repeatNo;
    };

    struct sortType
    {
      int number;
      int value;
    };

    struct handType
    {
      int next;
      int spareKey;
      unsigned remainCards[DDS_HANDS][DDS_SUITS];
      int NTflag;
      int first;
      int strain;
      int repeatNo;
      int depth;
      int strength;
      int fanout;
      int thread;
      int selectFlag;
      int time;
    };

    handType hands[MAXNOOFBOARDS];

    groupType group[MAXNOOFBOARDS];
    int numGroups;
    int extraGroups;

    atomic<int> currGroup;

    listType list[DDS_SUITS + 2][HASH_MAX];

    sortType sortList[MAXNOOFBOARDS];
    int sortLen;

    vector<int> threadGroup;
    vector<int> threadCurrGroup;
    vector<int> threadToHand;

    int numThreads;
    int numHands;

    vector<int> highCards;

    void InitHighCards();

    void SortHands(const enum RunMode mode);

    int Strength(const deal& dl) const;
    int Fanout(const deal& dl) const;

    void Reset();

    void MakeGroups(const boards& bds);

    void FinetuneGroups();

    bool SameHand(
      const int hno1,
      const int hno2) const;

    void SortSolve(),
         SortCalc(),
         SortTrace();

    int PredictedTime(
      deal& dl,
      int number) const;


  public:

    Scheduler();

    ~Scheduler();

    void RegisterThreads(
      const int n);

    void RegisterRun(
      const enum RunMode mode,
      const boards& bds,
      const playTracesBin& pl);

    void RegisterRun(
      const enum RunMode mode,
      const boards& bds);

    schedType GetNumber(const int thrId);

    int NumGroups() const;

};

#endif
