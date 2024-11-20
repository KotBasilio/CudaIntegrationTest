/*
   DDS, a bridge double dummy solver.

   Copyright (C) 2006-2014 by Bo Haglund /
   2014-2018 by Bo Haglund & Soren Hein.

   See LICENSE and README.
*/


/*
   See TimerList.h for some description.
*/

// #include <sstream>

#include "TimerList.h"


TimerList::TimerList()
{
  TimerList::Reset();
}


TimerList::~TimerList()
{
}


void TimerList::Reset()
{
  timerGroups.resize(TIMER_NO_SIZE);

  timerGroups[TIMER_NO_AB].SetNames("AB");
  timerGroups[TIMER_NO_MAKE].SetNames("Make");
  timerGroups[TIMER_NO_UNDO].SetNames("Undo");
  timerGroups[TIMER_NO_EVALUATE].SetNames("Evaluate");
  timerGroups[TIMER_NO_NEXTMOVE].SetNames("NextMove");
  timerGroups[TIMER_NO_QT].SetNames("QuickTricks");
  timerGroups[TIMER_NO_LT].SetNames("LaterTricks");
  timerGroups[TIMER_NO_MOVEGEN].SetNames("MoveGen");
  timerGroups[TIMER_NO_LOOKUP].SetNames("Lookup");
  timerGroups[TIMER_NO_BUILD].SetNames("Build");
}


void TimerList::Start(
  const ABTimerType groupno,
  const unsigned timerno)
{
  if (groupno >= TIMER_NO_SIZE)
    return;
  timerGroups[groupno].Start(timerno);
}


void TimerList::End(
  const ABTimerType groupno,
  const unsigned timerno)
{
  if (groupno >= TIMER_NO_SIZE)
    return;
  timerGroups[groupno].End(timerno);
}


bool TimerList::Used() const
{
  for (unsigned g = 0; g < TIMER_NO_SIZE; g++)
  {
    if (timerGroups[g].Used())
      return true;
  }
  return false;
}


void TimerList::PrintStats(ofstream& fout) const
{
}

