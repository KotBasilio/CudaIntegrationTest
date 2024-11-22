/*
   DDS, a bridge double dummy solver.

   Copyright (C) 2006-2014 by Bo Haglund /
   2014-2018 by Bo Haglund & Soren Hein.

   See LICENSE and README.

   2024 GPU acceleration by Serge Mironov
*/

#define  _CRT_SECURE_NO_WARNINGS

#include <algorithm>
#include <string.h>

#include "Init.h"
#include "System.h"
#include "Scheduler.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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


int _initialized = 0;


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


void SetDeal(
  ThreadData * thrp)
{
  /* Initialization of the rel structure is inspired by
     a solution given by Thomas Andrews */

  for (int s = 0; s < DDS_SUITS; s++)
  {
    thrp->lookAheadPos.aggr[s] = 0;
    for (int h = 0; h < DDS_HANDS; h++)
    {
      thrp->lookAheadPos.rankInSuit[h][s] = thrp->suit[h][s];
      thrp->lookAheadPos.aggr[s] |= thrp->suit[h][s];
    }
  }

  for (int s = 0; s < DDS_SUITS; s++)
  {
    for (int h = 0; h < DDS_HANDS; h++)
      thrp->lookAheadPos.length[h][s] = static_cast<unsigned char>(
        counttable[thrp->lookAheadPos.rankInSuit[h][s]]);
  }

  // Clubs are implicit, for a given trick number.
  for (int h = 0; h < DDS_HANDS; h++)
  {
    thrp->lookAheadPos.handDist[h] =
      static_cast<long long>(
        (thrp->lookAheadPos.length[h][0] << 8) |
        (thrp->lookAheadPos.length[h][1] << 4) |
        (thrp->lookAheadPos.length[h][2] ));
  }
}


void SetDealTables(
  ThreadData * thrp)
{
  unsigned int topBitRank = 1;
  unsigned int topBitNo = 2;

  // Initialization of the rel structure is inspired by
  // a solution given by Thomas Andrews.

  // rel[aggr].absRank[absolute rank][suit].hand is the hand
  // (N = 0, E = 1 etc.) which holds the absolute rank in
  // the suit characterized by aggr.
  // rel[aggr].absRank[absolute rank][suit].rank is the
  // relative rank of that card.

  for (int s = 0; s < DDS_SUITS; s++)
  {
    for (int ord = 1; ord <= 13; ord++)
    {
      thrp->rel[0].absRank[ord][s].hand = -1;
      thrp->rel[0].absRank[ord][s].rank = 0;
    }
  }

  // handLookup[suit][absolute rank] is the hand (N = 0 etc.)
  // holding the absolute rank in suit.

  int handLookup[DDS_SUITS][15];
  for (int s = 0; s < DDS_SUITS; s++)
  {
    for (int r = 14; r >= 2; r--)
    {
      handLookup[s][r] = 0;
      for (int h = 0; h < DDS_HANDS; h++)
      {
        if (thrp->suit[h][s] & bitMapRank[r])
        {
          handLookup[s][r] = h;
          break;
        }
      }
    }
  }

  thrp->transTable->Init(handLookup);

  relRanksType * relp;
  for (unsigned int aggr = 1; aggr < 8192; aggr++)
  {
    if (aggr >= (topBitRank << 1))
    {
      /* Next top bit */
      topBitRank <<= 1;
      topBitNo++;
    }

    thrp->rel[aggr] = thrp->rel[aggr ^ topBitRank];
    relp = &thrp->rel[aggr];

    int weight = counttable[aggr];
    for (int c = weight; c >= 2; c--)
    {
      for (int s = 0; s < DDS_SUITS; s++)
      {
        relp->absRank[c][s].hand = relp->absRank[c - 1][s].hand;
        relp->absRank[c][s].rank = relp->absRank[c - 1][s].rank;
      }
    }
    for (int s = 0; s < DDS_SUITS; s++)
    {
      relp->absRank[1][s].hand =
        static_cast<signed char>(handLookup[s][topBitNo]);
      relp->absRank[1][s].rank = static_cast<char>(topBitNo);
    }
  }
}


void InitWinners(
  const deal& dl,
  pos& posPoint,
  ThreadData const * thrp)
{
  int hand, suit, rank;
  unsigned short int startMovesBitMap[DDS_HANDS][DDS_SUITS];

  for (int h = 0; h < DDS_HANDS; h++)
    for (int s = 0; s < DDS_SUITS; s++)
      startMovesBitMap[h][s] = 0;

  for (int k = 0; k < posPoint.handRelFirst; k++)
  {
    hand = handId(dl.first, k);
    suit = dl.currentTrickSuit[k];
    rank = dl.currentTrickRank[k];
    startMovesBitMap[hand][suit] |= bitMapRank[rank];
  }

  int aggr;
  for (int s = 0; s < DDS_SUITS; s++)
  {
    aggr = 0;
    for (int h = 0; h < DDS_HANDS; h++)
      aggr |= startMovesBitMap[h][s] | thrp->suit[h][s];

    posPoint.winner[s].rank = thrp->rel[aggr].absRank[1][s].rank;
    posPoint.winner[s].hand = thrp->rel[aggr].absRank[1][s].hand;
    posPoint.secondBest[s].rank = thrp->rel[aggr].absRank[2][s].rank;
    posPoint.secondBest[s].hand = thrp->rel[aggr].absRank[2][s].hand;
  }
}


void ResetBestMoves(
  ThreadData * thrp)
{
  for (int d = 0; d <= 49; d++)
  {
    thrp->bestMove [d].rank = 0;
    thrp->bestMoveTT[d].rank = 0;
  }

  thrp->memUsed = thrp->transTable->MemoryInUse();
}

extern System sysdep;
extern Memory memory;

void  FreeMemory()
{
  for (unsigned thrId = 0; thrId < memory.NumThreads(); thrId++)
    memory.ReturnThread(thrId);
}


void  ErrorMessage(int code, char line[80])
{
  switch (code)
  {
    case RETURN_NO_FAULT:
      strcpy(line, TEXT_NO_FAULT);
      break;
    case RETURN_UNKNOWN_FAULT:
      strcpy(line, TEXT_UNKNOWN_FAULT);
      break;
    case RETURN_ZERO_CARDS:
      strcpy(line, TEXT_ZERO_CARDS);
      break;
    case RETURN_TARGET_TOO_HIGH:
      strcpy(line, TEXT_TARGET_TOO_HIGH);
      break;
    case RETURN_DUPLICATE_CARDS:
      strcpy(line, TEXT_DUPLICATE_CARDS);
      break;
    case RETURN_TARGET_WRONG_LO:
      strcpy(line, TEXT_TARGET_WRONG_LO);
      break;
    case RETURN_TARGET_WRONG_HI:
      strcpy(line, TEXT_TARGET_WRONG_HI);
      break;
    case RETURN_SOLNS_WRONG_LO:
      strcpy(line, TEXT_SOLNS_WRONG_LO);
      break;
    case RETURN_SOLNS_WRONG_HI:
      strcpy(line, TEXT_SOLNS_WRONG_HI);
      break;
    case RETURN_TOO_MANY_CARDS:
      strcpy(line, TEXT_TOO_MANY_CARDS);
      break;
    case RETURN_SUIT_OR_RANK:
      strcpy(line, TEXT_SUIT_OR_RANK);
      break;
    case RETURN_PLAYED_CARD:
      strcpy(line, TEXT_PLAYED_CARD);
      break;
    case RETURN_CARD_COUNT:
      strcpy(line, TEXT_CARD_COUNT);
      break;
    case RETURN_THREAD_INDEX:
      strcpy(line, TEXT_THREAD_INDEX);
      break;
    case RETURN_MODE_WRONG_LO:
      strcpy(line, TEXT_MODE_WRONG_LO);
      break;
    case RETURN_MODE_WRONG_HI:
      strcpy(line, TEXT_MODE_WRONG_HI);
      break;
    case RETURN_TRUMP_WRONG:
      strcpy(line, TEXT_TRUMP_WRONG);
      break;
    case RETURN_FIRST_WRONG:
      strcpy(line, TEXT_FIRST_WRONG);
      break;
    case RETURN_PLAY_FAULT:
      strcpy(line, TEXT_PLAY_FAULT);
      break;
    case RETURN_PBN_FAULT:
      strcpy(line, TEXT_PBN_FAULT);
      break;
    case RETURN_TOO_MANY_BOARDS:
      strcpy(line, TEXT_TOO_MANY_BOARDS);
      break;
    case RETURN_THREAD_CREATE:
      strcpy(line, TEXT_THREAD_CREATE);
      break;
    case RETURN_THREAD_WAIT:
      strcpy(line, TEXT_THREAD_WAIT);
      break;
    case RETURN_THREAD_MISSING:
      strcpy(line, TEXT_THREAD_MISSING);
      break;
    case RETURN_NO_SUIT:
      strcpy(line, TEXT_NO_SUIT);
      break;
    case RETURN_TOO_MANY_TABLES:
      strcpy(line, TEXT_TOO_MANY_TABLES);
      break;
    case RETURN_CHUNK_SIZE:
      strcpy(line, TEXT_CHUNK_SIZE);
      break;
    default:
      strcpy(line, "Not a DDS error code");
      break;
  }
}

